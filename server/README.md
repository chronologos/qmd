# QMD Remote LLM Server

Server implementation for offloading QMD's GPU-intensive operations to a remote machine (e.g., NVIDIA DGX Spark), accessible via Tailscale.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DGX Spark                              │
│                                                             │
│  ┌──────────────────────┐    ┌──────────────────────────┐  │
│  │  NVIDIA vLLM (8000)  │    │  Embed/Rerank API (8001) │  │
│  │  /v1/completions     │    │  /v1/embeddings          │  │
│  │  /v1/chat/completions│    │  /v1/rerank              │  │
│  └──────────────────────┘    └──────────────────────────┘  │
│            │                            │                   │
│            └──────────┬─────────────────┘                   │
│                       │                                     │
│              Tailscale Serve (HTTPS)                        │
│                       │                                     │
└───────────────────────┼─────────────────────────────────────┘
                        │
            ┌───────────┴───────────┐
            │                       │
    qmd-vllm.tailnet.ts.net  qmd-embed.tailnet.ts.net
```

Two services:
1. **NVIDIA vLLM** (port 8000) - Text generation for query expansion
2. **Embed/Rerank API** (port 8001) - Embeddings and document reranking

## DGX Spark Deployment

### Prerequisites

1. Tailscale connected to your tailnet
2. Docker with GPU support
3. Python 3.12+ with uv

### Step 1: Set Up Embed/Rerank Service

```bash
cd server

# Create venv and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Install PyTorch with CUDA support (required for DGX Spark)
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### Step 2: Create Tailscale Services

In the [Tailscale admin console](https://login.tailscale.com/admin/services):

1. Create service `qmd-embed` with endpoint `tcp:443`
2. Create service `qmd-vllm` with endpoint `tcp:443`

### Step 3: Deploy Embed/Rerank Service

```bash
# Deploy with systemd + Tailscale Serve
sudo python3 deploy.py
```

This creates:
- systemd service `qmd-embed.service`
- Tailscale Serve config: `https://qmd-embed.<tailnet>.ts.net` → `localhost:8001`

Management commands:
```bash
sudo python3 deploy.py --status   # Check status
sudo python3 deploy.py --logs     # View logs
sudo python3 deploy.py --restart  # Restart service
sudo python3 deploy.py --stop     # Stop service
```

### Step 4: Deploy vLLM Service

```bash
# Pull the NVIDIA vLLM container (DGX Spark optimized)
docker pull nvcr.io/nvidia/vllm:25.12.post1-py3

# Start vLLM serving Qwen3-4B for query expansion
docker run -d --gpus all \
  -p 127.0.0.1:8000:8000 \
  --restart unless-stopped \
  --name qmd-vllm \
  nvcr.io/nvidia/vllm:25.12.post1-py3 \
  vllm serve "Qwen/Qwen3-4B"

# Set up Tailscale Serve for vLLM
sudo tailscale serve --service svc:qmd-vllm --bg --https=443 127.0.0.1:8000
```

### Step 5: Verify Services

```bash
# From another machine on the tailnet
curl https://qmd-embed.<tailnet>.ts.net/health
curl https://qmd-vllm.<tailnet>.ts.net/health
```

### Step 6: Configure QMD Client

Add to `~/.config/qmd/index.yml`:

```yaml
remote:
  generation_url: "https://qmd-vllm.<tailnet>.ts.net"
  embed_url: "https://qmd-embed.<tailnet>.ts.net"
  models:
    embed: "Qwen/Qwen3-Embedding-4B"
    generate: "Qwen/Qwen3-4B"
    rerank: "Qwen/Qwen3-Reranker-4B"
```

Test connection:
```bash
qmd remote test
```

## Environment Variables

### Embed/Rerank Server

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `Qwen/Qwen3-Embedding-4B` | HuggingFace embedding model |
| `RERANK_MODEL` | `Qwen/Qwen3-Reranker-4B` | HuggingFace reranking model |
| `DEVICE` | `cuda` | Device: cuda, cpu, mps |
| `MAX_BATCH_SIZE` | `64` | Maximum embedding batch size |
| `USE_FLASH_ATTN` | `true` | Enable flash attention 2 for faster embedding |

## API Endpoints

### POST /v1/embeddings

OpenAI-compatible embedding endpoint.

```bash
curl -X POST https://qmd-embed.<tailnet>.ts.net/v1/embeddings \
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
curl -X POST https://qmd-embed.<tailnet>.ts.net/v1/rerank \
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
curl https://qmd-embed.<tailnet>.ts.net/health
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

## Model Recommendations

**Recommended: Unified Qwen3 Stack** (~32GB total, 25% of DGX Spark's 128GB)

| Task | Model | Size | Notes |
|------|-------|------|-------|
| Embedding | `Qwen/Qwen3-Embedding-4B` | 4B | SOTA quality, 32K context, 100+ languages |
| Embedding | `Qwen/Qwen3-Embedding-0.6B` | 0.6B | Faster alternative, still excellent |
| Reranking | `Qwen/Qwen3-Reranker-4B` | 4B | +13% vs bge-reranker, excellent code retrieval |
| Reranking | `Qwen/Qwen3-Reranker-0.6B` | 0.6B | Drop-in speed, +6% quality |
| Generation | `Qwen/Qwen3-4B` | 4B | Default, good quality for query expansion, 128K context |
| Generation | `Qwen/Qwen3-8B` | 8B | Higher quality, use if VRAM allows |

**Legacy models** (for backward compatibility):

| Task | Model | Size | Notes |
|------|-------|------|-------|
| Embedding | `nomic-ai/nomic-embed-text-v1.5` | 137M | Good quality, 768-dim |
| Reranking | `BAAI/bge-reranker-v2-m3` | 278M | Multilingual, uses CrossEncoder |
| Generation | `Qwen/Qwen2.5-3B-Instruct` | 3B | Previous default |

## Troubleshooting

### CUDA Not Available

If `/health` shows `"cuda_available": false`, reinstall PyTorch with CUDA:

```bash
source .venv/bin/activate
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130
sudo python3 deploy.py --restart
```

### Tailscale Service Not Reachable

1. Verify service exists in admin console with endpoint `tcp:443`
2. Restart tailscaled: `sudo systemctl restart tailscaled`
3. Check serve config: `tailscale serve status --json`

### CUDA Out of Memory

Reduce batch size or use smaller models:

```bash
export MAX_BATCH_SIZE=32
export EMBED_MODEL=BAAI/bge-base-en-v1.5
export RERANK_MODEL=BAAI/bge-reranker-base
```

### vLLM Container Issues

```bash
# Check container logs
docker logs qmd-vllm

# Restart container
docker restart qmd-vllm

# Remove and recreate
docker rm -f qmd-vllm
docker run -d --gpus all -p 127.0.0.1:8000:8000 \
  --restart unless-stopped --name qmd-vllm \
  nvcr.io/nvidia/vllm:25.12.post1-py3 \
  vllm serve "Qwen/Qwen3-4B"
```
