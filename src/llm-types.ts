/**
 * llm-types.ts - Shared types and utilities for LLM implementations
 *
 * This module contains types and pure functions shared between LlamaCpp and RemoteLLM
 * to avoid circular imports while eliminating code duplication.
 */

// =============================================================================
// Types
// =============================================================================

/**
 * Token with log probability
 */
export type TokenLogProb = {
  token: string;
  logprob: number;
};

/**
 * Embedding result
 */
export type EmbeddingResult = {
  embedding: number[];
  model: string;
};

/**
 * Generation result with optional logprobs
 */
export type GenerateResult = {
  text: string;
  model: string;
  logprobs?: TokenLogProb[];
  done: boolean;
};

/**
 * Rerank result for a single document
 */
export type RerankDocumentResult = {
  file: string;
  score: number;
  index: number;
};

/**
 * Batch rerank result
 */
export type RerankResult = {
  results: RerankDocumentResult[];
  model: string;
};

/**
 * Model info
 */
export type ModelInfo = {
  name: string;
  exists: boolean;
  path?: string;
};

/**
 * Options for embedding
 */
export type EmbedOptions = {
  model?: string;
  isQuery?: boolean;
  title?: string;
};

/**
 * Options for text generation
 */
export type GenerateOptions = {
  model?: string;
  maxTokens?: number;
  temperature?: number;
  stop?: string[];
};

/**
 * Options for reranking
 */
export type RerankOptions = {
  model?: string;
};

/**
 * Supported query types for different search backends
 */
export type QueryType = "lex" | "vec" | "hyde";

/**
 * A single query and its target backend type
 */
export type Queryable = {
  type: QueryType;
  text: string;
};

/**
 * Document to rerank
 */
export type RerankDocument = {
  file: string;
  text: string;
  title?: string;
};

// =============================================================================
// LLM Interface
// =============================================================================

/**
 * Abstract LLM interface - implement this for different backends
 */
export interface LLM {
  /**
   * Get embeddings for text
   */
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null>;

  /**
   * Batch embed multiple texts efficiently
   */
  embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]>;

  /**
   * Generate text completion
   */
  generate(
    prompt: string,
    options?: GenerateOptions
  ): Promise<GenerateResult | null>;

  /**
   * Check if a model exists/is available
   */
  modelExists(model: string): Promise<ModelInfo>;

  /**
   * Expand a search query into multiple variations for different backends.
   * Returns a list of Queryable objects.
   */
  expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean }
  ): Promise<Queryable[]>;

  /**
   * Rerank documents by relevance to a query
   * Returns list of documents with relevance scores (higher = more relevant)
   */
  rerank(
    query: string,
    documents: RerankDocument[],
    options?: RerankOptions
  ): Promise<RerankResult>;

  /**
   * Tokenize text into tokens (for chunking)
   * Returns an array of opaque token objects
   */
  tokenize(text: string): Promise<readonly unknown[]>;

  /**
   * Detokenize tokens back to text
   */
  detokenize(tokens: readonly unknown[]): Promise<string>;

  /**
   * Dispose of resources
   */
  dispose(): Promise<void>;
}

// =============================================================================
// Shared Utilities
// =============================================================================

/**
 * Parse and limit queryables from raw text output.
 * Deduplicates by text content and limits to max 3 lex, 3 vec, 1 hyde.
 */
export function parseQueryables(
  rawText: string,
  includeLexical: boolean
): Queryable[] {
  const lines = rawText.trim().split("\n");
  const seen = new Set<string>();
  const lex: Queryable[] = [];
  const vec: Queryable[] = [];
  const hyde: Queryable[] = [];

  for (const line of lines) {
    const colonIdx = line.indexOf(":");
    if (colonIdx === -1) continue;

    const type = line.slice(0, colonIdx).trim().toLowerCase();
    if (type !== "lex" && type !== "vec" && type !== "hyde") continue;

    const text = line.slice(colonIdx + 1).trim();
    if (!text || seen.has(text)) continue;
    seen.add(text);

    const q: Queryable = { type: type as QueryType, text };

    if (type === "lex" && lex.length < 3) {
      lex.push(q);
    } else if (type === "vec" && vec.length < 3) {
      vec.push(q);
    } else if (type === "hyde" && hyde.length < 1) {
      hyde.push(q);
    }
  }

  // Combine in order: lex, vec, hyde
  const result = [...lex, ...vec, ...hyde];

  // Filter out lex if not requested
  if (!includeLexical) {
    return result.filter((q) => q.type !== "lex");
  }
  return result;
}

// =============================================================================
// Embedding Formatting Functions
// =============================================================================

/**
 * Format a query for embedding.
 * Uses nomic-style task prefix format for embeddinggemma.
 */
export function formatQueryForEmbedding(query: string): string {
  return `task: search result | query: ${query}`;
}

/**
 * Format a document for embedding.
 * Uses nomic-style format with title and text fields.
 */
export function formatDocForEmbedding(text: string, title?: string): string {
  return `title: ${title || "none"} | text: ${text}`;
}
