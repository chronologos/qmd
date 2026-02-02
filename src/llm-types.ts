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
 * Options for parseQueryables
 */
export type ParseQueryablesOptions = {
  /** Include lex (lexical) queries in output (default: true) */
  includeLexical?: boolean;
  /**
   * Original query for term validation.
   * When provided, filters out expanded queries that don't contain
   * at least one term from the original query (guards against hallucinations).
   */
  originalQuery?: string;
};

/**
 * Extract alphanumeric terms from a query string for validation.
 */
function extractQueryTerms(query: string): string[] {
  return query
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
}

/**
 * Check if text contains at least one term from the original query.
 * Returns true if no terms provided (no validation needed).
 */
function hasQueryTerm(text: string, terms: string[]): boolean {
  if (terms.length === 0) return true;
  const lower = text.toLowerCase();
  return terms.some((term) => lower.includes(term));
}

/**
 * Create fallback queryables when parsing fails or returns nothing useful.
 * Returns: hyde, lex (if requested), vec - all using the original query.
 */
export function createFallbackQueryables(
  query: string,
  includeLexical: boolean
): Queryable[] {
  const fallback: Queryable[] = [
    { type: "hyde", text: `Information about ${query}` },
    { type: "vec", text: query },
  ];
  if (includeLexical) {
    fallback.splice(1, 0, { type: "lex", text: query });
  }
  return fallback;
}

/**
 * Parse and limit queryables from raw text output.
 * Deduplicates by text content and limits to max 3 lex, 3 vec, 1 hyde.
 *
 * When `originalQuery` is provided, validates that each parsed queryable
 * contains at least one term from the original query (filters hallucinations).
 *
 * Returns an empty array if no valid queryables are found - caller should
 * use createFallbackQueryables() to generate sensible defaults.
 */
export function parseQueryables(
  rawText: string,
  options?: ParseQueryablesOptions | boolean
): Queryable[] {
  // Support legacy signature: parseQueryables(text, includeLexical)
  const opts: ParseQueryablesOptions =
    typeof options === "boolean" ? { includeLexical: options } : options ?? {};

  const includeLexical = opts.includeLexical ?? true;
  const queryTerms = opts.originalQuery
    ? extractQueryTerms(opts.originalQuery)
    : [];

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

    // Validate against original query terms if provided
    if (queryTerms.length > 0 && !hasQueryTerm(text, queryTerms)) continue;

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
