# DGX Spark Model Upgrade Analysis

*January 2026*

## Executive Summary

Your DGX Spark with **128GB unified memory** is significantly underutilized. Current models (nomic-embed-text-v1.5, bge-reranker-v2-m3, Qwen2.5-3B-Instruct) total ~5GB VRAM. You could run models 10-20x larger with dramatically better quality.

**Recommended upgrade path**: Move to the Qwen3 family across all three components for a unified, SOTA stack.

---

## Current Hardware: DGX Spark

| Spec | Value |
|------|-------|
| GPU | Blackwell (6,144 CUDA cores) |
| Memory | 128GB unified LPDDR5x |
| Bandwidth | 273 GB/s |
| AI Performance | 1 PFLOP FP4 |
| Max Model Size | ~200B parameters |

The unified memory architecture means models can exceed traditional "VRAM limits"—the full 128GB is accessible to GPU compute.

---

## Component 1: Embedding Models

### Current: nomic-embed-text-v1.5
- **Parameters**: ~137M
- **Dimensions**: 768 (truncatable to 256)
- **Context**: 8,192 tokens
- **MTEB Score**: ~62.4

### Upgrade Options

| Model | Params | MTEB Score | Context | Dimensions | Memory |
|-------|--------|------------|---------|------------|--------|
| **Qwen3-Embedding-0.6B** | 600M | ~67 | 32K | 32-1024 | ~1.2GB |
| **Qwen3-Embedding-4B** ★ | 4B | ~69.5 | 32K | 32-1024 | ~8GB |
| **Qwen3-Embedding-8B** | 8B | **70.58** (#1 MTEB) | 32K | 32-1024 | ~16GB |
| F2LLM-4B | 4B | ~68 | 8K | 1024 | ~8GB |

### Recommendation: **Qwen3-Embedding-4B**

**Pros:**
- +7 points on MTEB vs nomic-embed (~11% relative improvement)
- 32K context (4x your current 8K)
- Flexible output dimensions (can use 768 for compatibility)
- Matches 8B performance on most tasks
- 100+ language support
- Apache 2.0 license

**Cons:**
- Slower inference (~4x parameters)
- Requires re-embedding your entire corpus
- Different embedding space (not compatible with existing vectors)

**Alternative**: Qwen3-Embedding-0.6B if you prioritize speed—still beats nomic-embed significantly.

---

## Component 2: Reranking Models

### Current: bge-reranker-v2-m3
- **Parameters**: ~560M
- **BEIR Score**: 56.51
- **MIRACL Score**: 69.32

### Upgrade Options

| Model | Params | MTEB-R | Code | Memory | Speed |
|-------|--------|--------|------|--------|-------|
| bge-reranker-v2-m3 (current) | 560M | ~60 | ~55 | ~1GB | Fast |
| **Qwen3-Reranker-0.6B** | 600M | ~66 | ~78 | ~1.2GB | Fast |
| **Qwen3-Reranker-4B** ★ | 4B | ~68 | ~80 | ~8GB | Medium |
| **Qwen3-Reranker-8B** | 8B | **69.02** | **81.22** | ~16GB | Slower |
| jina-reranker-v3 | 600M | ~62 | ~58 | ~1.2GB | Fast |

### Recommendation: **Qwen3-Reranker-0.6B** or **Qwen3-Reranker-4B**

**Qwen3-Reranker-0.6B Pros:**
- +3-4 points on MTEB-R vs bge-reranker (~6% improvement)
- Nearly identical size to current model
- Same inference speed
- Supports custom instructions (domain-specific prompting)
- Drop-in replacement

**Qwen3-Reranker-4B Pros:**
- +8 points on MTEB-R vs bge-reranker (~13% improvement)
- Excellent code retrieval (if you add code to your collection)
- Still fits easily in 128GB

**Cons:**
- Larger model = slower reranking
- For 40 documents: ~200ms (0.6B) vs ~800ms (4B)

---

## Component 3: Generation (Query Expansion)

### Current: Qwen2.5-3B-Instruct
- **Parameters**: 3B
- **Context**: 128K
- **Use case**: Query expansion, HyDE generation

### Upgrade Options

| Model | Params | vs Qwen2.5-7B | Context | Memory | Notes |
|-------|--------|---------------|---------|--------|-------|
| Qwen2.5-3B (current) | 3B | -30% | 128K | ~6GB | Baseline |
| **Qwen3-4B-Instruct** ★ | 4B | Equal | 32K | ~8GB | Best efficiency |
| **Qwen3-8B-Instruct** | 8B | +20% | 128K | ~16GB | Strong reasoning |
| **Qwen3-14B-Instruct** | 14B | +40% | 128K | ~28GB | Excellent quality |
| Phi-4-mini (3.8B) | 3.8B | Similar | 128K | ~8GB | Strong reasoning |

### Recommendation: **Qwen3-8B-Instruct**

**Pros:**
- Significantly better reasoning for query expansion
- Generates higher quality HyDE passages
- Better keyword extraction
- 128K context (vs 32K for Qwen3-4B)
- Still only uses ~16GB of your 128GB

**Cons:**
- ~2x slower than current 3B model
- May require prompt adjustments

**Alternative**: Qwen3-14B if latency isn't critical—it's still well within your memory budget.

---

## Unified Qwen3 Stack

Running all Qwen3 models together creates synergy:

```
┌─────────────────────────────────────────────────────────────┐
│                    DGX Spark (128GB)                        │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Qwen3-Embed-4B  │  │ Qwen3-Rerank-4B │  │ Qwen3-8B    │ │
│  │     ~8GB        │  │     ~8GB        │  │   ~16GB     │ │
│  │  Embeddings     │  │  Reranking      │  │  Generation │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
│                                                             │
│  Total: ~32GB / 128GB available (25% utilization)          │
└─────────────────────────────────────────────────────────────┘
```

### Benefits of Unified Stack

1. **Shared tokenizer**: All Qwen3 models use the same tokenizer
2. **Consistent multilingual support**: 100+ languages across all components
3. **Instruction-aware**: All models support custom task instructions
4. **Simplified deployment**: Same model family, same serving infrastructure
5. **Future upgrades**: Easy to swap 4B → 8B as needed

---

## Memory Budget Analysis

| Configuration | Embed | Rerank | Generate | Total | Headroom |
|--------------|-------|--------|----------|-------|----------|
| **Current** | 0.3GB | 1GB | 6GB | 7.3GB | 120GB |
| **Conservative** | 1.2GB | 1.2GB | 8GB | 10.4GB | 118GB |
| **Recommended** | 8GB | 8GB | 16GB | 32GB | 96GB |
| **Maximum** | 16GB | 16GB | 28GB | 60GB | 68GB |

Even the "Maximum" configuration uses less than half your available memory.

---

## Implementation Considerations

### Migration Path

1. **Start with generation** (Qwen3-8B): No reindexing needed
2. **Then reranking** (Qwen3-Reranker-4B): Drop-in replacement
3. **Finally embedding** (Qwen3-Embedding-4B): Requires full reindex

### Embedding Migration

Switching embedding models requires:
- Clearing existing vectors from SQLite
- Re-running `qmd embed --remote` on all documents
- Your 2,136 chunks took ~64 seconds; expect similar time

### Server Changes

Update `server/embed_rerank.py`:

```python
# Before
embed_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")
rerank_model = CrossEncoder("BAAI/bge-reranker-v2-m3")

# After (option 1: sentence-transformers)
embed_model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")
rerank_model = CrossEncoder("Qwen/Qwen3-Reranker-4B")

# After (option 2: transformers directly)
from transformers import AutoModel, AutoTokenizer
embed_model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-4B")
```

### vLLM Changes

```bash
# Before
vllm serve "Qwen/Qwen2.5-3B-Instruct"

# After
vllm serve "Qwen/Qwen3-8B-Instruct"
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Embedding incompatibility | High | Medium | Test on subset first |
| Slower inference | Medium | Low | 128GB gives massive batch headroom |
| Prompt changes needed | Medium | Low | Qwen3 uses similar prompts |
| Memory issues | Low | High | You have 4x headroom |
| Quality regression | Low | Medium | Benchmark before/after |

---

## Recommendations Summary

### Tier 1: Quick Wins (No Reindexing)

| Component | Upgrade To | Improvement | Effort |
|-----------|-----------|-------------|--------|
| Generation | Qwen3-8B-Instruct | +20% quality | Low |
| Reranking | Qwen3-Reranker-0.6B | +6% quality | Low |

### Tier 2: Full Upgrade (Requires Reindexing)

| Component | Upgrade To | Improvement | Effort |
|-----------|-----------|-------------|--------|
| Embedding | Qwen3-Embedding-4B | +11% quality | Medium |
| Reranking | Qwen3-Reranker-4B | +13% quality | Low |
| Generation | Qwen3-8B-Instruct | +20% quality | Low |

---

## Sources

### Hardware
- [DGX Spark Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [LMSYS DGX Spark Review](https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)
- [NVIDIA DGX Spark Optimizations](https://developer.nvidia.com/blog/new-software-and-model-optimizations-supercharge-nvidia-dgx-spark/)

### Embedding Models
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Qwen3 Embedding Blog](https://qwenlm.github.io/blog/qwen3-embedding/)
- [Qwen3-Embedding-4B on HuggingFace](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
- [Open Source Embedding Models Guide](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models)

### Reranking Models
- [Top Rerankers for RAG](https://www.analyticsvidhya.com/blog/2025/06/top-rerankers-for-rag/)
- [Choosing Reranking Models Guide](https://www.zeroentropy.dev/articles/ultimate-guide-to-choosing-the-best-reranking-model-in-2025)
- [Qwen3-Reranker-8B on HuggingFace](https://huggingface.co/Qwen/Qwen3-Reranker-8B)

### Generation Models
- [Qwen3 Technical Report](https://qwenlm.github.io/blog/qwen3/)
- [Small Language Models Guide](https://www.bentoml.com/blog/the-best-open-source-small-language-models)
- [Qwen3-8B Specifications](https://apxml.com/models/qwen3-8b)
