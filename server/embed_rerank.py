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
  EMBED_MODEL    - HuggingFace model for embeddings (default: nomic-ai/nomic-embed-text-v1.5)
  RERANK_MODEL   - HuggingFace model for reranking (default: BAAI/bge-reranker-v2-m3)
  DEVICE         - Device to use: cuda, cpu, mps (default: cuda)
  MAX_BATCH_SIZE - Maximum batch size for embeddings (default: 64)
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

EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "64"))

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

class Models:
    """Lazy-loaded model container"""
    embed_model = None
    rerank_model = None

    @classmethod
    def get_embed_model(cls):
        if cls.embed_model is None:
            logger.info(f"Loading embedding model: {EMBED_MODEL}")
            start = time.time()
            from sentence_transformers import SentenceTransformer
            cls.embed_model = SentenceTransformer(
                EMBED_MODEL,
                device=DEVICE,
                trust_remote_code=True,  # Required for nomic models
            )
            logger.info(f"Embedding model loaded in {time.time() - start:.2f}s")
        return cls.embed_model

    @classmethod
    def get_rerank_model(cls):
        if cls.rerank_model is None:
            logger.info(f"Loading rerank model: {RERANK_MODEL}")
            start = time.time()
            from sentence_transformers import CrossEncoder
            cls.rerank_model = CrossEncoder(
                RERANK_MODEL,
                device=DEVICE,
                trust_remote_code=True,
            )
            logger.info(f"Rerank model loaded in {time.time() - start:.2f}s")
        return cls.rerank_model


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


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """
    Rerank documents by relevance to a query.

    Returns documents sorted by relevance score (highest first).
    """
    try:
        model = Models.get_rerank_model()

        if len(request.documents) == 0:
            raise HTTPException(status_code=400, detail="Documents cannot be empty")

        # Build query-document pairs
        pairs = [[request.query, doc] for doc in request.documents]

        # Score all pairs
        start = time.time()
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
