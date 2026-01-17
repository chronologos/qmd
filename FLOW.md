# QMD Search Flows

Sequence diagrams illustrating the data flow for each search command.

## `qmd search` - BM25 Keyword Search

Pure local operation using SQLite FTS5.

```mermaid
sequenceDiagram
    participant User
    participant CLI as qmd CLI
    participant SQLite as SQLite FTS5

    User->>CLI: qmd search "meditation"
    CLI->>SQLite: BM25 query
    SQLite-->>CLI: Ranked results (keyword match)
    CLI-->>User: Results with scores
```

## `qmd vsearch` - Vector Similarity Search

Uses remote GPU for query embedding, local SQLite for vector search.

```mermaid
sequenceDiagram
    participant User
    participant CLI as qmd CLI
    participant Remote as Remote Embed<br/>(nomic-embed)
    participant SQLite as SQLite + sqlite-vec

    User->>CLI: qmd vsearch "meditation"
    CLI->>Remote: POST /v1/embeddings<br/>{"input": "meditation"}
    Remote-->>CLI: embedding[768]
    CLI->>SQLite: Vector similarity search<br/>(cosine distance)
    SQLite-->>CLI: Nearest neighbors
    CLI-->>User: Results with similarity scores
```

## `qmd query` - Hybrid Search with Query Expansion + Reranking

Full pipeline using both remote services.

```mermaid
sequenceDiagram
    participant User
    participant CLI as qmd CLI
    participant vLLM as Remote vLLM<br/>(Qwen2.5-3B)
    participant Embed as Remote Embed<br/>(nomic-embed)
    participant Rerank as Remote Rerank<br/>(bge-reranker)
    participant SQLite as SQLite

    User->>CLI: qmd query "meditation"

    Note over CLI,vLLM: Phase 1: Query Expansion
    CLI->>vLLM: POST /v1/completions<br/>expand query prompt
    vLLM-->>CLI: lex: relaxation<br/>lex: mindfulness<br/>vec: meditation+practice<br/>hyde: Meditation involves...

    Note over CLI,SQLite: Phase 2: Multi-Query Search

    par Lexical Search
        CLI->>SQLite: BM25: "meditation"
        CLI->>SQLite: BM25: "relaxation"
        CLI->>SQLite: BM25: "mindfulness"
        SQLite-->>CLI: Lexical candidates
    and Vector Search
        CLI->>Embed: POST /v1/embeddings<br/>batch embed all vec/hyde queries
        Embed-->>CLI: embeddings[]
        CLI->>SQLite: Vector search (each embedding)
        SQLite-->>CLI: Vector candidates
    end

    Note over CLI: Phase 3: Fusion (RRF)
    CLI->>CLI: Reciprocal Rank Fusion<br/>merge lexical + vector results

    Note over CLI,Rerank: Phase 4: Reranking
    CLI->>Rerank: POST /v1/rerank<br/>{"query": "meditation",<br/> "documents": [top 40 chunks]}
    Rerank-->>CLI: Reranked scores

    CLI-->>User: Final results<br/>(sorted by rerank score)
```

## Service Architecture

```mermaid
flowchart TB
    subgraph Local["Local Machine"]
        CLI[qmd CLI]
        SQLite[(SQLite<br/>FTS5 + sqlite-vec)]
    end

    subgraph DGX["DGX Spark (Remote GPU)"]
        subgraph EmbedService["Embed/Rerank Service :8001"]
            Embed[nomic-embed-text-v1.5]
            Rerank[bge-reranker-v2-m3]
        end
        subgraph vLLMService["vLLM Service :8000"]
            LLM[Qwen2.5-3B-Instruct]
        end
    end

    CLI -->|Index + Search| SQLite
    CLI -->|/v1/embeddings| Embed
    CLI -->|/v1/rerank| Rerank
    CLI -->|/v1/completions| LLM
```

## Data Storage

All data remains local. Remote services are stateless compute.

```mermaid
flowchart LR
    subgraph Stored["Stored Locally (SQLite)"]
        Docs[Document text]
        Meta[Metadata + hashes]
        FTS[FTS5 index]
        Vec[Vector embeddings]
    end

    subgraph Computed["Computed Remotely (GPU)"]
        E[Embedding vectors]
        R[Rerank scores]
        Q[Query expansions]
    end

    Computed -->|returned & stored| Stored
```
