#!/usr/bin/env python3
"""
QMD Remote LLM Server - Embeddings & Reranking

FastAPI server providing OpenAI-compatible embedding and reranking endpoints.
Designed to run on NVIDIA DGX Spark alongside vLLM for text generation.

Endpoints:
  POST /v1/embeddings - Generate embeddings for text(s)
  POST /v1/rerank     - Rerank documents by relevance to a query
  GET  /health        - Health check

Usage:
  uvicorn embed_rerank:app --host 0.0.0.0 --port 8001

Environment variables:
  EMBED_MODEL    - HuggingFace model for embeddings (default: Qwen/Qwen3-Embedding-4B)
  RERANK_MODEL   - HuggingFace model for reranking (default: Qwen/Qwen3-Reranker-4B)
  DEVICE         - Device to use: cuda, cpu, mps (default: cuda)
  MAX_BATCH_SIZE - Maximum batch size for embeddings (default: 64)
  USE_FLASH_ATTN - Enable flash attention 2 (default: true)
"""

import os
import time
import logging
from typing import Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

EMBED_MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "Qwen/Qwen3-Reranker-4B")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "64"))
# Check if flash_attn is actually available (CUDA version must match)
_flash_attn_available = False
if os.environ.get("USE_FLASH_ATTN", "true").lower() == "true":
    try:
        import flash_attn  # noqa: F401
        _flash_attn_available = True
    except ImportError:
        pass
USE_FLASH_ATTN = _flash_attn_available

# =============================================================================
# Request/Response Models
# =============================================================================

class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request"""
    input: Union[str, list[str]] = Field(..., description="Text(s) to embed")
    model: str = Field(default="nomic-embed", description="Model name (ignored, uses server model)")


class EmbeddingData(BaseModel):
    """Single embedding result"""
    embedding: list[float]
    index: int
    object: str = "embedding"


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response"""
    data: list[EmbeddingData]
    model: str
    object: str = "list"
    usage: dict = Field(default_factory=lambda: {"prompt_tokens": 0, "total_tokens": 0})


class RerankRequest(BaseModel):
    """Rerank request"""
    query: str = Field(..., description="Query to rank documents against")
    documents: list[str] = Field(..., description="Documents to rerank")
    model: str = Field(default="bge-reranker", description="Model name (ignored, uses server model)")


class RerankResult(BaseModel):
    """Single rerank result"""
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    """Rerank response"""
    results: list[RerankResult]
    model: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models: dict
    device: str
    cuda_available: bool


# =============================================================================
# Model Management
# =============================================================================

# Check if we're using Qwen3 reranker (requires different loading approach)
IS_QWEN3_RERANKER = "qwen3-reranker" in RERANK_MODEL.lower()

class Models:
    """Lazy-loaded model container"""
    embed_model = None
    rerank_model = None
    rerank_tokenizer = None
    # Token IDs for Qwen3 reranker yes/no scoring
    token_true_id = None
    token_false_id = None

    @classmethod
    def get_embed_model(cls):
        if cls.embed_model is None:
            logger.info(f"Loading embedding model: {EMBED_MODEL}")
            start = time.time()
            from sentence_transformers import SentenceTransformer

            # Configure model kwargs for Qwen3 or other models
            model_kwargs = {"trust_remote_code": True}
            tokenizer_kwargs = {}

            if "qwen3" in EMBED_MODEL.lower():
                # Qwen3 benefits from flash attention and left padding
                if USE_FLASH_ATTN:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                tokenizer_kwargs["padding_side"] = "left"

                cls.embed_model = SentenceTransformer(
                    EMBED_MODEL,
                    device=DEVICE,
                    model_kwargs=model_kwargs,
                    tokenizer_kwargs=tokenizer_kwargs,
                    trust_remote_code=True,
                )
            else:
                # Legacy loading for non-Qwen3 models
                cls.embed_model = SentenceTransformer(
                    EMBED_MODEL,
                    device=DEVICE,
                    trust_remote_code=True,
                )
            logger.info(f"Embedding model loaded in {time.time() - start:.2f}s")
        return cls.embed_model

    @classmethod
    def get_rerank_model(cls):
        if cls.rerank_model is None:
            logger.info(f"Loading rerank model: {RERANK_MODEL}")
            start = time.time()

            if IS_QWEN3_RERANKER:
                # Qwen3 reranker uses CausalLM with yes/no token scoring
                from transformers import AutoModelForCausalLM, AutoTokenizer

                logger.info(f"Loading tokenizer for {RERANK_MODEL}...")
                cls.rerank_tokenizer = AutoTokenizer.from_pretrained(
                    RERANK_MODEL,
                    padding_side="left",
                    trust_remote_code=True,
                )
                logger.info(f"Tokenizer loaded, loading model weights...")
                cls.rerank_model = AutoModelForCausalLM.from_pretrained(
                    RERANK_MODEL,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                ).to(DEVICE).eval()
                logger.info(f"Model loaded to {DEVICE}")

                # Cache token IDs for yes/no
                cls.token_true_id = cls.rerank_tokenizer.convert_tokens_to_ids("yes")
                cls.token_false_id = cls.rerank_tokenizer.convert_tokens_to_ids("no")
                logger.info(f"Qwen3 reranker token IDs: yes={cls.token_true_id}, no={cls.token_false_id}")
            else:
                # CrossEncoder for traditional rerankers (bge, etc.)
                from sentence_transformers import CrossEncoder
                cls.rerank_model = CrossEncoder(
                    RERANK_MODEL,
                    device=DEVICE,
                    trust_remote_code=True,
                )

            logger.info(f"Rerank model loaded in {time.time() - start:.2f}s")
        return cls.rerank_model

    @classmethod
    def get_rerank_tokenizer(cls):
        """Get tokenizer for Qwen3 reranker"""
        if cls.rerank_tokenizer is None:
            cls.get_rerank_model()  # This loads both model and tokenizer
        return cls.rerank_tokenizer


# =============================================================================
# FastAPI Application
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload models on startup"""
    logger.info("Preloading models...")
    Models.get_embed_model()
    Models.get_rerank_model()
    logger.info("Models ready!")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="QMD Remote LLM Server",
    description="Embeddings and reranking for QMD",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        models={
            "embed": EMBED_MODEL,
            "rerank": RERANK_MODEL,
        },
        device=DEVICE,
        cuda_available=torch.cuda.is_available(),
    )


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embeddings(request: EmbeddingRequest):
    """
    Generate embeddings for input text(s).

    Compatible with OpenAI's /v1/embeddings endpoint.
    """
    try:
        model = Models.get_embed_model()

        # Normalize input to list
        texts = [request.input] if isinstance(request.input, str) else request.input

        if len(texts) == 0:
            raise HTTPException(status_code=400, detail="Input cannot be empty")

        if len(texts) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size {len(texts)} exceeds maximum {MAX_BATCH_SIZE}"
            )

        # Generate embeddings
        start = time.time()
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
            show_progress_bar=False,
        )
        elapsed = time.time() - start

        logger.info(f"Embedded {len(texts)} texts in {elapsed:.3f}s ({len(texts)/elapsed:.1f} texts/sec)")

        # Build response
        data = [
            EmbeddingData(
                embedding=vec.tolist(),
                index=i,
            )
            for i, vec in enumerate(vectors)
        ]

        return EmbeddingResponse(
            data=data,
            model=EMBED_MODEL,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Embedding error")
        raise HTTPException(status_code=500, detail=str(e))


def _format_qwen3_rerank_input(query: str, document: str, instruction: str | None = None) -> str:
    """Format input for Qwen3 reranker with conversation template."""
    if instruction is None:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"

    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    return f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}{suffix}"


@torch.no_grad()
def _qwen3_rerank_score(query: str, documents: list[str], instruction: str | None = None) -> list[float]:
    """Score documents using Qwen3 reranker's yes/no token probabilities."""
    model = Models.get_rerank_model()
    tokenizer = Models.get_rerank_tokenizer()

    # Format all inputs
    inputs_text = [_format_qwen3_rerank_input(query, doc, instruction) for doc in documents]

    # Tokenize
    inputs = tokenizer(
        inputs_text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=8192,
    ).to(model.device)

    # Get logits for the last token position
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]

    # Extract yes/no probabilities
    true_logits = logits[:, Models.token_true_id]
    false_logits = logits[:, Models.token_false_id]

    # Compute P(yes) / (P(yes) + P(no)) via log_softmax
    stacked = torch.stack([false_logits, true_logits], dim=1)
    log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
    scores = log_probs[:, 1].exp().tolist()

    return scores


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """
    Rerank documents by relevance to a query.

    Returns documents sorted by relevance score (highest first).
    """
    try:
        if len(request.documents) == 0:
            raise HTTPException(status_code=400, detail="Documents cannot be empty")

        start = time.time()

        if IS_QWEN3_RERANKER:
            # Use Qwen3's yes/no token scoring
            scores = _qwen3_rerank_score(request.query, request.documents)
        else:
            # Traditional CrossEncoder scoring
            model = Models.get_rerank_model()
            pairs = [[request.query, doc] for doc in request.documents]
            scores = model.predict(pairs, show_progress_bar=False)

        elapsed = time.time() - start
        logger.info(f"Reranked {len(request.documents)} documents in {elapsed:.3f}s")

        # Build results sorted by score (descending)
        results = [
            RerankResult(index=i, relevance_score=float(score))
            for i, score in enumerate(scores)
        ]
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        return RerankResponse(
            results=results,
            model=RERANK_MODEL,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Rerank error")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
