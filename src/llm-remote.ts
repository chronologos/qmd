/**
 * llm-remote.ts — Remote LLM backend for QMD
 *
 * Fork-only file. Implements the duck-type surface expected by store.ts and qmd.ts,
 * delegating to a remote OpenAI-compatible API instead of local node-llama-cpp.
 *
 * Key design:
 * - Implements LLM interface (embed, generate, modelExists, expandQuery, rerank, dispose)
 * - Also implements embedBatch, tokenize, detokenize (called by store.ts/qmd.ts)
 * - Provides getDeviceInfo() for qmd.ts status display
 * - HTTP retry with exponential backoff
 * - Pseudo-tokenization for remote (~4 chars per token)
 */

import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  ModelInfo,
  RerankOptions,
  RerankResult,
  RerankDocument,
  Queryable,
  QueryType,
} from "./llm.js";

// =============================================================================
// Configuration
// =============================================================================

export interface RemoteLLMConfig {
  /** Base URL for generation/completions API (OpenAI-compatible) */
  generationUrl: string;
  /** Base URL for embedding/rerank API */
  embedUrl: string;
  /** Optional API key for authentication */
  apiKey?: string;
  /** Model names to use */
  models?: {
    embed?: string;
    generate?: string;
    rerank?: string;
  };
  /** Connection timeout in ms (default: 10000) */
  connectTimeoutMs?: number;
  /** Request timeout in ms (default: 120000) */
  requestTimeoutMs?: number;
  /** Max retry attempts (default: 3) */
  maxRetries?: number;
}

// =============================================================================
// HTTP Helpers
// =============================================================================

const DEFAULT_CONNECT_TIMEOUT_MS = 10_000;
const DEFAULT_REQUEST_TIMEOUT_MS = 120_000;
const DEFAULT_MAX_RETRIES = 3;

/** Pseudo-tokenization ratio: ~4 chars per token */
const CHARS_PER_TOKEN = 4;

interface RetryOptions {
  maxRetries: number;
  connectTimeoutMs: number;
  requestTimeoutMs: number;
  apiKey?: string;
}

/**
 * HTTP fetch with exponential backoff retry.
 * Retries on network errors and 5xx responses.
 */
async function fetchWithRetry(
  url: string,
  body: unknown,
  options: RetryOptions
): Promise<Response> {
  const { maxRetries, requestTimeoutMs, apiKey } = options;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (apiKey) {
    headers["Authorization"] = `Bearer ${apiKey}`;
  }

  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), requestTimeoutMs);

      const response = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      clearTimeout(timeout);

      // Don't retry on 4xx client errors
      if (response.status >= 400 && response.status < 500) {
        const text = await response.text().catch(() => "");
        throw new Error(
          `HTTP ${response.status} from ${url}: ${text}`
        );
      }

      // Retry on 5xx server errors
      if (response.status >= 500) {
        lastError = new Error(`HTTP ${response.status} from ${url}`);
        if (attempt < maxRetries) {
          await backoff(attempt);
          continue;
        }
        throw lastError;
      }

      return response;
    } catch (error: any) {
      // Don't retry 4xx
      if (error.message?.includes("HTTP 4")) {
        throw error;
      }
      lastError = error;
      if (attempt < maxRetries) {
        await backoff(attempt);
      }
    }
  }

  throw lastError || new Error(`Failed after ${maxRetries + 1} attempts`);
}

/** Exponential backoff: 500ms, 1s, 2s, ... with jitter */
function backoff(attempt: number): Promise<void> {
  const baseMs = 500 * Math.pow(2, attempt);
  const jitter = Math.random() * baseMs * 0.3;
  return new Promise((resolve) => setTimeout(resolve, baseMs + jitter));
}

// =============================================================================
// Query Expansion Helpers
// =============================================================================

/**
 * Parse raw text output into Queryable objects.
 *
 * Expected format (one per line):
 *   type: text
 * where type is 'lex', 'vec', or 'hyde'.
 */
function parseQueryables(text: string, includeLexical: boolean): Queryable[] {
  const lines = text.trim().split("\n");
  const queryables: Queryable[] = [];

  for (const line of lines) {
    const colonIdx = line.indexOf(":");
    if (colonIdx === -1) continue;

    const type = line.slice(0, colonIdx).trim();
    if (type !== "lex" && type !== "vec" && type !== "hyde") continue;

    const content = line.slice(colonIdx + 1).trim();
    if (!content) continue;

    queryables.push({ type: type as QueryType, text: content });
  }

  // Filter out lex entries if not requested
  return includeLexical
    ? queryables
    : queryables.filter((q) => q.type !== "lex");
}

/**
 * Filter queryables to retain only those that share at least one
 * meaningful term with the original query. This prevents hallucinated
 * expansions from polluting search results.
 *
 * Exported for testability.
 */
export function filterByQueryTerms(
  queryables: Queryable[],
  originalQuery: string
): Queryable[] {
  const queryLower = originalQuery.toLowerCase();
  const queryTerms = queryLower
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);

  if (queryTerms.length === 0) return queryables;

  return queryables.filter((q) => {
    const lower = q.text.toLowerCase();
    return queryTerms.some((term) => lower.includes(term));
  });
}

/**
 * Create fallback queryables when expansion produces nothing useful.
 *
 * Exported for testability.
 */
export function createFallbackQueryables(
  query: string,
  includeLexical: boolean
): Queryable[] {
  const fallback: Queryable[] = [
    { type: "hyde", text: `Information about ${query}` },
    { type: "lex", text: query },
    { type: "vec", text: query },
  ];
  return includeLexical
    ? fallback
    : fallback.filter((q) => q.type !== "lex");
}

// =============================================================================
// RemoteLLM Class
// =============================================================================

export class RemoteLLM implements LLM {
  private config: Required<
    Pick<RemoteLLMConfig, "generationUrl" | "embedUrl">
  > &
    RemoteLLMConfig;
  private retryOpts: RetryOptions;
  private disposed = false;

  constructor(config: RemoteLLMConfig) {
    this.config = {
      ...config,
      generationUrl: config.generationUrl.replace(/\/+$/, ""),
      embedUrl: config.embedUrl.replace(/\/+$/, ""),
    };
    this.retryOpts = {
      maxRetries: config.maxRetries ?? DEFAULT_MAX_RETRIES,
      connectTimeoutMs: config.connectTimeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS,
      requestTimeoutMs: config.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS,
      apiKey: config.apiKey,
    };
  }

  // ==========================================================================
  // LLM Interface Methods
  // ==========================================================================

  async embed(
    text: string,
    options: EmbedOptions = {}
  ): Promise<EmbeddingResult | null> {
    const model = options.model || this.config.models?.embed || "default";

    try {
      const response = await fetchWithRetry(
        `${this.config.embedUrl}/v1/embeddings`,
        {
          model,
          input: text,
        },
        this.retryOpts
      );

      const data = (await response.json()) as {
        data: { embedding: number[] }[];
        model: string;
      };

      if (!data.data?.[0]?.embedding) {
        console.error("Remote embed: unexpected response shape");
        return null;
      }

      return {
        embedding: data.data[0].embedding,
        model: data.model || model,
      };
    } catch (error) {
      console.error("Remote embed error:", error);
      return null;
    }
  }

  /**
   * Batch embed multiple texts. Sends all texts in a single API call.
   * Not on the LLM interface but called by store.ts.
   */
  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    const model = this.config.models?.embed || "default";

    try {
      const response = await fetchWithRetry(
        `${this.config.embedUrl}/v1/embeddings`,
        {
          model,
          input: texts,
        },
        this.retryOpts
      );

      const data = (await response.json()) as {
        data: { embedding: number[]; index: number }[];
        model: string;
      };

      if (!data.data || !Array.isArray(data.data)) {
        console.error("Remote embedBatch: unexpected response shape");
        return texts.map(() => null);
      }

      // The API returns results sorted by index, but be defensive
      const results: (EmbeddingResult | null)[] = new Array(texts.length).fill(
        null
      );
      for (const item of data.data) {
        if (item.index >= 0 && item.index < texts.length && item.embedding) {
          results[item.index] = {
            embedding: item.embedding,
            model: data.model || model,
          };
        }
      }

      return results;
    } catch (error) {
      console.error("Remote embedBatch error:", error);
      return texts.map(() => null);
    }
  }

  async generate(
    prompt: string,
    options: GenerateOptions = {}
  ): Promise<GenerateResult | null> {
    const model = options.model || this.config.models?.generate || "default";
    const maxTokens = options.maxTokens ?? 150;
    const temperature = options.temperature ?? 0.7;

    try {
      const response = await fetchWithRetry(
        `${this.config.generationUrl}/v1/completions`,
        {
          model,
          prompt,
          max_tokens: maxTokens,
          temperature,
          top_p: 0.8,
          top_k: 20,
        },
        this.retryOpts
      );

      const data = (await response.json()) as {
        choices: { text: string; finish_reason: string }[];
        model: string;
      };

      if (!data.choices?.[0]) {
        console.error("Remote generate: unexpected response shape");
        return null;
      }

      return {
        text: data.choices[0].text,
        model: data.model || model,
        done: data.choices[0].finish_reason === "stop",
      };
    } catch (error) {
      console.error("Remote generate error:", error);
      return null;
    }
  }

  async modelExists(model: string): Promise<ModelInfo> {
    // For remote, assume all models exist — the server manages them
    return { name: model, exists: true };
  }

  async expandQuery(
    query: string,
    options: { context?: string; includeLexical?: boolean } = {}
  ): Promise<Queryable[]> {
    const includeLexical = options.includeLexical ?? true;

    try {
      const prompt = `/no_think Expand this search query: ${query}`;
      const result = await this.generate(prompt, {
        maxTokens: 600,
        temperature: 0.7,
      });

      if (!result?.text) {
        return createFallbackQueryables(query, includeLexical);
      }

      // Parse the structured output
      const queryables = parseQueryables(result.text, includeLexical);

      // Filter to retain only expansions that share terms with the original query
      const filtered = filterByQueryTerms(queryables, query);

      // Fall back if nothing survived filtering
      if (filtered.length === 0) {
        return createFallbackQueryables(query, includeLexical);
      }

      return filtered;
    } catch (error) {
      console.error("Remote expandQuery error:", error);
      // Fallback to original query
      const fallback: Queryable[] = [{ type: "vec", text: query }];
      if (includeLexical) fallback.unshift({ type: "lex", text: query });
      return fallback;
    }
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options: RerankOptions = {}
  ): Promise<RerankResult> {
    const model = options.model || this.config.models?.rerank || "default";

    if (documents.length === 0) {
      return { results: [], model };
    }

    try {
      const response = await fetchWithRetry(
        `${this.config.embedUrl}/v1/rerank`,
        {
          model,
          query,
          documents: documents.map((d) => d.text),
        },
        this.retryOpts
      );

      const data = (await response.json()) as {
        results: { index: number; relevance_score: number }[];
        model: string;
      };

      if (!data.results || !Array.isArray(data.results)) {
        console.error("Remote rerank: unexpected response shape");
        // Return documents in original order with zero scores
        return {
          results: documents.map((doc, i) => ({
            file: doc.file,
            score: 0,
            index: i,
          })),
          model,
        };
      }

      // Sort by relevance score descending
      const sorted = [...data.results].sort(
        (a, b) => b.relevance_score - a.relevance_score
      );

      return {
        results: sorted.map((r) => ({
          file: documents[r.index]!.file,
          score: r.relevance_score,
          index: r.index,
        })),
        model: data.model || model,
      };
    } catch (error) {
      console.error("Remote rerank error:", error);
      // Return documents in original order with zero scores as fallback
      return {
        results: documents.map((doc, i) => ({
          file: doc.file,
          score: 0,
          index: i,
        })),
        model,
      };
    }
  }

  async dispose(): Promise<void> {
    this.disposed = true;
    // No local resources to clean up for remote backend
  }

  // ==========================================================================
  // Additional Methods (not on LLM interface, but called by store.ts/qmd.ts)
  // ==========================================================================

  /**
   * Pseudo-tokenize text for remote backend.
   * Returns an array of sequential integers representing ~4 chars/token.
   * The actual token IDs don't matter — store.ts only uses .length.
   */
  async tokenize(text: string): Promise<readonly number[]> {
    const count = Math.max(1, Math.ceil(text.length / CHARS_PER_TOKEN));
    return Array.from({ length: count }, (_, i) => i);
  }

  /**
   * Pseudo-detokenize: approximate reverse of pseudo-tokenize.
   * Not actually meaningful for remote — returns a placeholder.
   */
  async detokenize(tokens: readonly number[]): Promise<string> {
    // We can't reconstruct text from pseudo-tokens.
    // This is called rarely (if ever) in practice.
    return `[${tokens.length} tokens]`;
  }

  /**
   * Get device info for status display (qmd.ts).
   * Reports as a remote backend rather than a local GPU.
   */
  async getDeviceInfo(): Promise<{
    gpu: string | false;
    gpuOffloading: boolean;
    gpuDevices: string[];
    vram?: { total: number; used: number; free: number };
    cpuCores: number;
    remote?: {
      generationUrl: string;
      embedUrl: string;
      models: Record<string, string>;
    };
  }> {
    return {
      gpu: "remote",
      gpuOffloading: true,
      gpuDevices: ["remote"],
      cpuCores: 0,
      remote: {
        generationUrl: this.config.generationUrl,
        embedUrl: this.config.embedUrl,
        models: {
          embed: this.config.models?.embed || "default",
          generate: this.config.models?.generate || "default",
          rerank: this.config.models?.rerank || "default",
        },
      },
    };
  }

  // ==========================================================================
  // Diagnostic Methods
  // ==========================================================================

  /**
   * Test connectivity to the remote server.
   * Returns true if both endpoints respond.
   */
  async testConnection(): Promise<{
    generation: { ok: boolean; error?: string; latencyMs?: number };
    embed: { ok: boolean; error?: string; latencyMs?: number };
  }> {
    const testEndpoint = async (
      baseUrl: string
    ): Promise<{ ok: boolean; error?: string; latencyMs?: number }> => {
      const start = Date.now();
      try {
        const headers: Record<string, string> = {};
        if (this.config.apiKey) {
          headers["Authorization"] = `Bearer ${this.config.apiKey}`;
        }

        const controller = new AbortController();
        const timeout = setTimeout(
          () => controller.abort(),
          this.retryOpts.connectTimeoutMs
        );

        const response = await fetch(`${baseUrl}/health`, {
          headers,
          signal: controller.signal,
        });
        clearTimeout(timeout);

        const latencyMs = Date.now() - start;

        if (!response.ok) {
          return {
            ok: false,
            error: `HTTP ${response.status}`,
            latencyMs,
          };
        }

        return { ok: true, latencyMs };
      } catch (error: any) {
        return {
          ok: false,
          error: error.message || String(error),
          latencyMs: Date.now() - start,
        };
      }
    };

    const [generation, embed] = await Promise.all([
      testEndpoint(this.config.generationUrl),
      testEndpoint(this.config.embedUrl),
    ]);

    return { generation, embed };
  }

  /**
   * Get status information about the remote configuration.
   */
  getStatus(): {
    generationUrl: string;
    embedUrl: string;
    models: {
      embed: string;
      generate: string;
      rerank: string;
    };
    disposed: boolean;
  } {
    return {
      generationUrl: this.config.generationUrl,
      embedUrl: this.config.embedUrl,
      models: {
        embed: this.config.models?.embed || "default",
        generate: this.config.models?.generate || "default",
        rerank: this.config.models?.rerank || "default",
      },
      disposed: this.disposed,
    };
  }
}
