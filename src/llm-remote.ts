/**
 * llm-remote.ts - Remote LLM backend for QMD
 *
 * Offloads GPU-intensive operations (embedding, reranking, query expansion)
 * to a remote server via OpenAI-compatible HTTP APIs.
 *
 * Default models (Qwen3 stack for DGX Spark):
 * - Embedding: Qwen/Qwen3-Embedding-4B
 * - Reranking: Qwen/Qwen3-Reranker-4B
 * - Generation: Qwen/Qwen3-8B
 */

import {
  parseQueryables,
  type LLM,
  type EmbedOptions,
  type EmbeddingResult,
  type GenerateOptions,
  type GenerateResult,
  type ModelInfo,
  type RerankOptions,
  type RerankResult,
  type RerankDocument,
  type Queryable,
} from "./llm-types.js";

// =============================================================================
// Configuration
// =============================================================================

export interface RemoteLLMConfig {
  /** Single URL for all services (fallback if specific URLs not set) */
  url?: string;
  /** URL for embed + rerank service */
  embedUrl?: string;
  /** URL for vLLM generation service */
  generationUrl?: string;
  /** Optional Bearer token for authentication */
  apiKey?: string;
  /** Model names for each operation */
  models?: {
    embed?: string;
    generate?: string;
    rerank?: string;
  };
  /** Request timeout in milliseconds (default: 30000) */
  timeout?: number;
  /** Retry count for transient failures (default: 3) */
  retries?: number;
}

// =============================================================================
// API Request/Response Types
// =============================================================================

interface EmbeddingRequest {
  input: string | string[];
  model: string;
}

interface EmbeddingResponse {
  data: Array<{
    embedding: number[];
    index: number;
  }>;
  model: string;
}

interface CompletionRequest {
  prompt: string;
  model: string;
  max_tokens?: number;
  temperature?: number;
  stop?: string[];
}

interface CompletionResponse {
  choices: Array<{
    text: string;
    finish_reason?: string;
  }>;
  model: string;
}

interface RerankRequest {
  query: string;
  documents: string[];
  model: string;
}

interface RerankResponse {
  results: Array<{
    index: number;
    relevance_score: number;
  }>;
  model: string;
}

interface HealthResponse {
  status: string;
  models?: Record<string, unknown>;
}

// =============================================================================
// RemoteLLM Implementation
// =============================================================================

const DEFAULT_TIMEOUT = 30000;
const DEFAULT_RETRIES = 3;

export class RemoteLLM implements LLM {
  private embedUrl: string;
  private generationUrl: string;
  private apiKey?: string;
  private models: {
    embed: string;
    generate: string;
    rerank: string;
  };
  private timeout: number;
  private retries: number;

  constructor(config: RemoteLLMConfig) {
    // Resolve URLs - use specific URLs if provided, fallback to base URL
    const baseUrl = config.url || "http://localhost:8000";
    this.embedUrl = config.embedUrl || baseUrl;
    this.generationUrl = config.generationUrl || baseUrl;

    // Expand environment variables in API key
    this.apiKey = config.apiKey ? this.expandEnvVars(config.apiKey) : undefined;

    // Default model names (Qwen3 stack)
    this.models = {
      embed: config.models?.embed || "Qwen/Qwen3-Embedding-4B",
      generate: config.models?.generate || "Qwen/Qwen3-8B",
      rerank: config.models?.rerank || "Qwen/Qwen3-Reranker-4B",
    };

    this.timeout = config.timeout ?? DEFAULT_TIMEOUT;
    this.retries = config.retries ?? DEFAULT_RETRIES;
  }

  /**
   * Expand environment variable references like ${VAR_NAME}
   */
  private expandEnvVars(value: string): string {
    return value.replace(/\$\{([^}]+)\}/g, (_, varName) => {
      return process.env[varName] || "";
    });
  }

  /**
   * Build headers for API requests
   */
  private getHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }
    return headers;
  }

  /**
   * Make HTTP request with retry and exponential backoff
   */
  private async fetchWithRetry<T>(
    url: string,
    body: unknown,
    operation: string
  ): Promise<T> {
    let lastError: Error | null = null;

    for (let attempt = 0; attempt < this.retries; attempt++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);

        const response = await fetch(url, {
          method: "POST",
          headers: this.getHeaders(),
          body: JSON.stringify(body),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        return (await response.json()) as T;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));

        // Don't retry on certain errors
        if (
          lastError.name === "AbortError" ||
          lastError.message.includes("HTTP 4")
        ) {
          break;
        }

        // Exponential backoff
        if (attempt < this.retries - 1) {
          const delay = Math.min(1000 * Math.pow(2, attempt), 10000);
          await new Promise((resolve) => setTimeout(resolve, delay));
        }
      }
    }

    throw new Error(`${operation} failed after ${this.retries} attempts: ${lastError?.message}`);
  }

  // ===========================================================================
  // LLM Interface Implementation
  // ===========================================================================

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    try {
      const request: EmbeddingRequest = {
        input: text,
        model: this.models.embed,
      };

      const response = await this.fetchWithRetry<EmbeddingResponse>(
        `${this.embedUrl}/v1/embeddings`,
        request,
        "embed"
      );

      const embedding = response.data[0]?.embedding;
      if (!embedding) {
        return null;
      }

      return {
        embedding,
        model: response.model || this.models.embed,
      };
    } catch (error) {
      console.error("Remote embedding error:", error);
      return null;
    }
  }

  /**
   * Batch embed multiple texts efficiently
   * Sends all texts in a single request - server handles parallelism
   */
  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    try {
      const request: EmbeddingRequest = {
        input: texts,
        model: this.models.embed,
      };

      const response = await this.fetchWithRetry<EmbeddingResponse>(
        `${this.embedUrl}/v1/embeddings`,
        request,
        "embedBatch"
      );

      // Map response back to input order
      const results: (EmbeddingResult | null)[] = new Array(texts.length).fill(null);
      for (const item of response.data) {
        if (item.embedding && item.index >= 0 && item.index < texts.length) {
          results[item.index] = {
            embedding: item.embedding,
            model: response.model || this.models.embed,
          };
        }
      }

      return results;
    } catch (error) {
      console.error("Remote batch embedding error:", error);
      return texts.map(() => null);
    }
  }

  async generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> {
    try {
      const request: CompletionRequest = {
        prompt,
        model: this.models.generate,
        max_tokens: options?.maxTokens ?? 150,
        temperature: options?.temperature ?? 0,
        stop: options?.stop,
      };

      const response = await this.fetchWithRetry<CompletionResponse>(
        `${this.generationUrl}/v1/completions`,
        request,
        "generate"
      );

      const text = response.choices[0]?.text;
      if (text === undefined) {
        return null;
      }

      return {
        text,
        model: response.model || this.models.generate,
        done: true,
      };
    } catch (error) {
      console.error("Remote generation error:", error);
      return null;
    }
  }

  async modelExists(model: string): Promise<ModelInfo> {
    // For remote models, we assume they exist if the server is reachable
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

      const response = await fetch(`${this.embedUrl}/health`, {
        method: "GET",
        headers: this.getHeaders(),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        return { name: model, exists: true };
      }
    } catch {
      // Server unreachable
    }

    return { name: model, exists: false };
  }

  async expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean }
  ): Promise<Queryable[]> {
    const includeLexical = options?.includeLexical ?? true;
    const context = options?.context;

    // Simpler prompt for remote - less chain-of-thought to reduce over-generation
    const prompt = `Generate search queries for: "${query}"
${context ? `\nContext: ${context}\n` : ""}
Output format (one per line):
${includeLexical ? "lex: keyword phrase\n" : ""}vec: semantic query
hyde: A complete sentence that would appear in a relevant document.

Rules:
- Output ONLY the formatted lines, no explanations
- Maximum: ${includeLexical ? "2 lex, " : ""}2 vec, 1 hyde
- Each line must be unique
- Use plain text, no URL encoding

Output:`;

    try {
      const result = await this.generate(prompt, {
        maxTokens: 300,  // Reduced from 1000
        temperature: 0.7,  // Reduced from 1.0 for more focused output
        stop: ["\n\n", "---", "Rules:", "Note:"],  // Stop on double newline or common continuation patterns
      });

      if (!result?.text) {
        // Fallback
        const fallback: Queryable[] = [{ type: "vec", text: query }];
        if (includeLexical) fallback.unshift({ type: "lex", text: query });
        return fallback;
      }

      // Use shared parser with deduplication and limits
      return parseQueryables(result.text, includeLexical);
    } catch (error) {
      console.error("Remote query expansion failed:", error);
      // Fallback to original query
      const fallback: Queryable[] = [{ type: "vec", text: query }];
      if (includeLexical) fallback.unshift({ type: "lex", text: query });
      return fallback;
    }
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options?: RerankOptions
  ): Promise<RerankResult> {
    const request: RerankRequest = {
      query,
      documents: documents.map((doc) => doc.text),
      model: this.models.rerank,
    };

    const response = await this.fetchWithRetry<RerankResponse>(
      `${this.embedUrl}/v1/rerank`,
      request,
      "rerank"
    );

    // Map results back to documents with file info
    const results = response.results.map((r) => {
      const doc = documents[r.index];
      return {
        file: doc?.file || "",
        score: r.relevance_score,
        index: r.index,
      };
    });

    return {
      results,
      model: response.model || this.models.rerank,
    };
  }

  async dispose(): Promise<void> {
    // No local resources to clean up
  }

  /**
   * Tokenize text into pseudo-tokens for chunking.
   * Uses a character-based approximation (~4 chars per token) since
   * remote servers don't typically expose tokenization endpoints.
   */
  async tokenize(text: string): Promise<readonly string[]> {
    // Approximate tokenization: split into ~4 character chunks
    // This is a rough approximation - real tokenizers are subword-based
    const CHARS_PER_TOKEN = 4;
    const tokens: string[] = [];
    for (let i = 0; i < text.length; i += CHARS_PER_TOKEN) {
      tokens.push(text.slice(i, i + CHARS_PER_TOKEN));
    }
    return tokens;
  }

  /**
   * Detokenize pseudo-tokens back to text.
   */
  async detokenize(tokens: readonly unknown[]): Promise<string> {
    return (tokens as string[]).join("");
  }

  // ===========================================================================
  // Remote-specific methods
  // ===========================================================================

  /**
   * Test connection to remote server
   */
  async testConnection(): Promise<{
    embedService: { ok: boolean; error?: string };
    generationService: { ok: boolean; error?: string };
  }> {
    const testService = async (
      url: string
    ): Promise<{ ok: boolean; error?: string }> => {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);

        const response = await fetch(`${url}/health`, {
          method: "GET",
          headers: this.getHeaders(),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (response.ok) {
          return { ok: true };
        }
        return { ok: false, error: `HTTP ${response.status}` };
      } catch (error) {
        return {
          ok: false,
          error: error instanceof Error ? error.message : String(error),
        };
      }
    };

    const [embedService, generationService] = await Promise.all([
      testService(this.embedUrl),
      testService(this.generationUrl),
    ]);

    return { embedService, generationService };
  }

  /**
   * Get server status and model info
   */
  async getStatus(): Promise<{
    embedUrl: string;
    generationUrl: string;
    health?: HealthResponse;
    error?: string;
  }> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

      const response = await fetch(`${this.embedUrl}/health`, {
        method: "GET",
        headers: this.getHeaders(),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        const health = (await response.json()) as HealthResponse;
        return {
          embedUrl: this.embedUrl,
          generationUrl: this.generationUrl,
          health,
        };
      }
      return {
        embedUrl: this.embedUrl,
        generationUrl: this.generationUrl,
        error: `HTTP ${response.status}`,
      };
    } catch (error) {
      return {
        embedUrl: this.embedUrl,
        generationUrl: this.generationUrl,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /**
   * Get configured URLs (for display)
   */
  getConfig(): { embedUrl: string; generationUrl: string; models: typeof this.models } {
    return {
      embedUrl: this.embedUrl,
      generationUrl: this.generationUrl,
      models: { ...this.models },
    };
  }
}
