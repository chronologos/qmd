/**
 * llm-remote.test.ts - Tests for the remote LLM backend
 *
 * Fork-only file. Tests pure helper functions (always run) and
 * RemoteLLM integration (requires QMD_TEST_REMOTE=1 + running server).
 */

import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { filterByQueryTerms, createFallbackQueryables, RemoteLLM } from "../src/llm-remote.js";
import type { Queryable, RerankDocument } from "../src/llm.js";
import {
  setupTestLLM,
  cleanupTestLLM,
  getTestEmbedDimensions,
  getTestLLMDescription,
  shouldUseRemoteLLM,
  shouldSkipLLMTests,
} from "../src/test-config.js";
import { getDefaultLlamaCpp } from "../src/llm.js";

// =============================================================================
// Pure Function Tests (always run, no LLM needed)
// =============================================================================

describe("filterByQueryTerms", () => {
  test("filters queries not containing original terms", () => {
    const queryables: Queryable[] = [
      { type: "vec", text: "rust programming" },
      { type: "vec", text: "python development" },
      { type: "hyde", text: "A document about rust" },
    ];

    const result = filterByQueryTerms(queryables, "rust");

    expect(result).toHaveLength(2);
    expect(result.map((q) => q.text)).toEqual([
      "rust programming",
      "A document about rust",
    ]);
  });

  test("matches any term from multi-word query", () => {
    const queryables: Queryable[] = [
      { type: "vec", text: "memory safety" },
      { type: "lex", text: "garbage collection" },
      { type: "hyde", text: "Systems programming with rust" },
    ];

    const result = filterByQueryTerms(queryables, "rust memory management");

    expect(result).toHaveLength(2);
    expect(result.map((q) => q.text)).toEqual([
      "memory safety",
      "Systems programming with rust",
    ]);
  });

  test("case-insensitive matching", () => {
    const queryables: Queryable[] = [
      { type: "vec", text: "RUST language" },
      { type: "lex", text: "Go concurrency" },
    ];

    const result = filterByQueryTerms(queryables, "rust");

    expect(result).toHaveLength(1);
    expect(result[0]!.text).toBe("RUST language");
  });

  test("strips punctuation from query for term extraction", () => {
    const queryables: Queryable[] = [
      { type: "vec", text: "async await" },
      { type: "lex", text: "promises callbacks" },
    ];

    const result = filterByQueryTerms(queryables, "async/await");

    expect(result).toHaveLength(1);
    expect(result[0]!.text).toBe("async await");
  });

  test("without original query, all pass", () => {
    const queryables: Queryable[] = [
      { type: "vec", text: "anything goes" },
      { type: "lex", text: "all pass through" },
      { type: "hyde", text: "everything included" },
    ];

    const result = filterByQueryTerms(queryables, "");

    expect(result).toHaveLength(3);
    expect(result).toEqual(queryables);
  });

  test("returns empty array when no matches", () => {
    const queryables: Queryable[] = [
      { type: "vec", text: "python programming" },
      { type: "lex", text: "javascript frameworks" },
    ];

    const result = filterByQueryTerms(queryables, "rust");

    expect(result).toHaveLength(0);
    expect(result).toEqual([]);
  });
});

describe("createFallbackQueryables", () => {
  test("creates fallback with lex when includeLexical is true", () => {
    const result = createFallbackQueryables("test query", true);

    expect(result).toHaveLength(3);

    const types = result.map((q) => q.type);
    expect(types).toContain("hyde");
    expect(types).toContain("lex");
    expect(types).toContain("vec");

    const hyde = result.find((q) => q.type === "hyde")!;
    expect(hyde.text).toBe("Information about test query");

    const lex = result.find((q) => q.type === "lex")!;
    expect(lex.text).toBe("test query");

    const vec = result.find((q) => q.type === "vec")!;
    expect(vec.text).toBe("test query");
  });

  test("creates fallback without lex when includeLexical is false", () => {
    const result = createFallbackQueryables("test query", false);

    expect(result).toHaveLength(2);

    const types = result.map((q) => q.type);
    expect(types).toContain("hyde");
    expect(types).toContain("vec");
    expect(types).not.toContain("lex");
  });
});

// =============================================================================
// RemoteLLM Integration Tests (require running remote server)
// =============================================================================

describe("RemoteLLM integration", () => {
  let llm: ReturnType<typeof getDefaultLlamaCpp> | null;

  beforeAll(async () => {
    llm = await setupTestLLM();
    if (llm) {
      console.log(`[llm-remote.test.ts] Running with ${getTestLLMDescription()}`);
    }
  });

  afterAll(async () => {
    await cleanupTestLLM();
  });

  test("embed returns embedding with correct dimensions", async () => {
    if (!llm || !shouldUseRemoteLLM()) return;

    const result = await llm.embed("Hello world");

    expect(result).not.toBeNull();
    expect(result!.embedding).toBeInstanceOf(Array);
    expect(result!.embedding.length).toBe(getTestEmbedDimensions());
  });

  test("embedBatch returns embeddings for multiple texts", async () => {
    if (!llm || !shouldUseRemoteLLM()) return;

    const texts = ["Hello world", "Test text", "Another document"];
    const results = await llm.embedBatch(texts);

    expect(results).toHaveLength(3);
    for (const result of results) {
      expect(result).not.toBeNull();
      expect(result!.embedding.length).toBe(getTestEmbedDimensions());
    }
  });

  test("expandQuery returns queryables with correct types", async () => {
    if (!llm || !shouldUseRemoteLLM()) return;

    const result = await llm.expandQuery("test query");

    expect(result.length).toBeGreaterThanOrEqual(1);
    for (const q of result) {
      expect(["lex", "vec", "hyde"]).toContain(q.type);
      expect(q.text.length).toBeGreaterThan(0);
    }
  }, 30000);

  test("rerank scores relevant documents higher", async () => {
    if (!llm || !shouldUseRemoteLLM()) return;

    const query = "What is the capital of France?";
    const documents: RerankDocument[] = [
      { file: "france.txt", text: "The capital of France is Paris." },
      { file: "butterflies.txt", text: "Butterflies indeed fly through the garden." },
      { file: "canada.txt", text: "The capital of Canada is Ottawa." },
    ];

    const result = await llm.rerank(query, documents);

    expect(result.results).toHaveLength(3);
    expect(result.results[0]!.file).toBe("france.txt");
    expect(result.results[0]!.score).toBeGreaterThan(0); // Catches silent error fallback
  });

  test("can exclude lexical queries from expandQuery", async () => {
    if (!llm || !shouldUseRemoteLLM()) return;

    const result = await llm.expandQuery("authentication setup", {
      includeLexical: false,
    });

    const lexEntries = result.filter((q) => q.type === "lex");
    expect(lexEntries).toHaveLength(0);
  }, 30000);
});
