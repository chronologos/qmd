# QMD Remote LLM Server

Reference server implementation for offloading QMD's GPU-intensive operations to a remote machine (e.g., NVIDIA DGX Spark).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DGX Spark                               │
│                                                              │
│  ┌──────────────────────┐    ┌──────────────────────────┐   │
│  │  NVIDIA vLLM (8000)  │    │  Embed/Rerank API (8001) │   │
│  │  /v1/completions     │    │  /v1/embeddings          │   │
│  │  /v1/chat/completions│    │  /v1/rerank              │   │
│  └──────────────────────┘    └──────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

Two services:
1. **NVIDIA vLLM** (port 8000) - Text generation for query expansion
2. **Embed/Rerank API** (port 8001) - Embeddings and document reranking

## Quick Start

### 1. Start the Embed/Rerank Server

```bash
cd server

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn embed_rerank:app --host 0.0.0.0 --port 8001
```

### 2. Start vLLM for Text Generation

Using the official NVIDIA container:

```bash
# Pull the container
docker pull nvcr.io/nvidia/vllm:25.12.post1-py3

# Run vLLM serving Qwen for query expansion
docker run -d --gpus all -p 8000:8000 \
  --name qmd-vllm \
  nvcr.io/nvidia/vllm:25.12.post1-py3 \
  vllm serve "Qwen/Qwen2.5-3B-Instruct"
```

Or with sentence-transformers models (smaller, good for testing):

```bash
docker run -d --gpus all -p 8000:8000 \
  --name qmd-vllm \
  nvcr.io/nvidia/vllm:25.12.post1-py3 \
  vllm serve "Qwen/Qwen2.5-1.5B-Instruct"
```

### 3. Configure QMD

Add to `~/.config/qmd/index.yml`:

```yaml
remote:
  generation_url: "http://dgx-spark:8000"  # vLLM
  embed_url: "http://dgx-spark:8001"       # Embed/Rerank
  # api_key: "${QMD_REMOTE_API_KEY}"       # Optional
  models:
    embed: "nomic-embed"
    generate: "Qwen/Qwen2.5-3B-Instruct"
    rerank: "bge-reranker"
```

### 4. Test Connection

```bash
qmd remote test
```

## Environment Variables

### Embed/Rerank Server

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | HuggingFace embedding model |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | HuggingFace reranking model |
| `DEVICE` | `cuda` | Device: cuda, cpu, mps |
| `MAX_BATCH_SIZE` | `64` | Maximum embedding batch size |

## API Endpoints

### POST /v1/embeddings

OpenAI-compatible embedding endpoint.

```bash
curl -X POST http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello world", "How are you?"],
    "model": "nomic-embed"
  }'
```

Response:
```json
{
  "data": [
    {"embedding": [0.1, 0.2, ...], "index": 0},
    {"embedding": [0.3, 0.4, ...], "index": 1}
  ],
  "model": "nomic-ai/nomic-embed-text-v1.5"
}
```

### POST /v1/rerank

Rerank documents by relevance to a query.

```bash
curl -X POST http://localhost:8001/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Python?",
    "documents": [
      "Python is a programming language",
      "Snakes are reptiles",
      "Python was created by Guido van Rossum"
    ]
  }'
```

Response:
```json
{
  "results": [
    {"index": 0, "relevance_score": 0.95},
    {"index": 2, "relevance_score": 0.87},
    {"index": 1, "relevance_score": 0.12}
  ],
  "model": "BAAI/bge-reranker-v2-m3"
}
```

### GET /health

Health check endpoint.

```bash
curl http://localhost:8001/health
```

Response:
```json
{
  "status": "ok",
  "models": {
    "embed": "nomic-ai/nomic-embed-text-v1.5",
    "rerank": "BAAI/bge-reranker-v2-m3"
  },
  "device": "cuda",
  "cuda_available": true
}
```

## Docker Deployment

### Build Image

```bash
docker build -t qmd-embed-rerank server/
```

### Run Container

```bash
docker run -d --gpus all -p 8001:8001 \
  --name qmd-embed-rerank \
  -e EMBED_MODEL=nomic-ai/nomic-embed-text-v1.5 \
  -e RERANK_MODEL=BAAI/bge-reranker-v2-m3 \
  qmd-embed-rerank
```

### Docker Compose

```yaml
version: '3.8'
services:
  vllm:
    image: nvcr.io/nvidia/vllm:25.12.post1-py3
    command: vllm serve "Qwen/Qwen2.5-3B-Instruct"
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  embed-rerank:
    build: ./server
    ports:
      - "8001:8001"
    environment:
      - EMBED_MODEL=nomic-ai/nomic-embed-text-v1.5
      - RERANK_MODEL=BAAI/bge-reranker-v2-m3
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## Model Recommendations

| Task | Model | Size | Notes |
|------|-------|------|-------|
| Embedding | `nomic-ai/nomic-embed-text-v1.5` | 137M | Good quality, 768-dim |
| Embedding | `BAAI/bge-base-en-v1.5` | 110M | Alternative, 768-dim |
| Reranking | `BAAI/bge-reranker-v2-m3` | 278M | Multilingual, excellent quality |
| Reranking | `BAAI/bge-reranker-base` | 109M | English-only, faster |
| Generation | `Qwen/Qwen2.5-3B-Instruct` | 3B | Good balance of quality/speed |
| Generation | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Faster, lower quality |

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or use smaller models:

```bash
export MAX_BATCH_SIZE=32
export EMBED_MODEL=BAAI/bge-base-en-v1.5
export RERANK_MODEL=BAAI/bge-reranker-base
```

### Connection Refused

Check that the server is running and accessible:

```bash
# Test embed/rerank server
curl http://localhost:8001/health

# Test vLLM server
curl http://localhost:8000/health
```

### Slow First Request

Models are loaded on first use. The `/health` endpoint triggers model loading at startup to warm up.
