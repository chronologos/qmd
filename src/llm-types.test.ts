/**
 * llm-types.test.ts - Unit tests for shared LLM utilities
 *
 * These tests don't require LLM backends - they test pure functions.
 */

import { describe, test, expect } from "bun:test";
import { parseQueryables, createFallbackQueryables, type Queryable } from "./llm-types.js";

describe("parseQueryables", () => {
  describe("basic parsing", () => {
    test("parses valid queryables from text", () => {
      const input = `lex: keyword search
vec: semantic search query
hyde: This is a hypothetical document.`;

      const result = parseQueryables(input, { includeLexical: true });

      expect(result).toEqual([
        { type: "lex", text: "keyword search" },
        { type: "vec", text: "semantic search query" },
        { type: "hyde", text: "This is a hypothetical document." },
      ]);
    });

    test("filters lex when includeLexical is false", () => {
      const input = `lex: keyword search
vec: semantic search query
hyde: This is a hypothetical document.`;

      const result = parseQueryables(input, { includeLexical: false });

      expect(result).toEqual([
        { type: "vec", text: "semantic search query" },
        { type: "hyde", text: "This is a hypothetical document." },
      ]);
    });

    test("supports legacy boolean signature", () => {
      const input = `lex: test
vec: test2`;

      const result = parseQueryables(input, false);

      expect(result).toEqual([{ type: "vec", text: "test2" }]);
    });

    test("deduplicates identical text", () => {
      const input = `lex: same text
vec: same text
lex: different text`;

      const result = parseQueryables(input, { includeLexical: true });

      expect(result).toEqual([
        { type: "lex", text: "same text" },
        { type: "lex", text: "different text" },
      ]);
    });

    test("limits: max 3 lex, 3 vec, 1 hyde", () => {
      const input = `lex: a
lex: b
lex: c
lex: d
vec: 1
vec: 2
vec: 3
vec: 4
hyde: first
hyde: second`;

      const result = parseQueryables(input, { includeLexical: true });

      const lexCount = result.filter((q) => q.type === "lex").length;
      const vecCount = result.filter((q) => q.type === "vec").length;
      const hydeCount = result.filter((q) => q.type === "hyde").length;

      expect(lexCount).toBe(3);
      expect(vecCount).toBe(3);
      expect(hydeCount).toBe(1);
    });

    test("ignores malformed lines", () => {
      const input = `lex: valid
not a queryable
: missing type
vec: another valid
unknown: wrong type`;

      const result = parseQueryables(input, { includeLexical: true });

      expect(result).toEqual([
        { type: "lex", text: "valid" },
        { type: "vec", text: "another valid" },
      ]);
    });

    test("returns empty array for empty input", () => {
      expect(parseQueryables("", { includeLexical: true })).toEqual([]);
      expect(parseQueryables("  \n\n  ", { includeLexical: true })).toEqual([]);
    });
  });

  describe("term validation (originalQuery)", () => {
    test("filters queries not containing original terms", () => {
      const input = `lex: rust programming
vec: python development
hyde: A document about rust language features.`;

      const result = parseQueryables(input, {
        includeLexical: true,
        originalQuery: "rust",
      });

      // Only lines containing "rust" should pass
      expect(result).toEqual([
        { type: "lex", text: "rust programming" },
        { type: "hyde", text: "A document about rust language features." },
      ]);
    });

    test("matches any term from multi-word query", () => {
      const input = `lex: memory safety
vec: garbage collection
hyde: Systems programming with rust.`;

      const result = parseQueryables(input, {
        includeLexical: true,
        originalQuery: "rust memory management",
      });

      // "memory" matches "memory safety", "rust" matches "rust"
      expect(result).toEqual([
        { type: "lex", text: "memory safety" },
        { type: "hyde", text: "Systems programming with rust." },
      ]);
    });

    test("case-insensitive matching", () => {
      const input = `lex: RUST language
vec: Go concurrency`;

      const result = parseQueryables(input, {
        includeLexical: true,
        originalQuery: "rust",
      });

      expect(result).toEqual([{ type: "lex", text: "RUST language" }]);
    });

    test("strips punctuation from query for term extraction", () => {
      const input = `lex: async await
vec: promises callbacks`;

      const result = parseQueryables(input, {
        includeLexical: true,
        originalQuery: "async/await",
      });

      // "async" should match despite "/" in query
      expect(result).toEqual([{ type: "lex", text: "async await" }]);
    });

    test("without originalQuery, all valid lines pass", () => {
      const input = `lex: unrelated topic
vec: something else`;

      const result = parseQueryables(input, {
        includeLexical: true,
        // no originalQuery
      });

      expect(result.length).toBe(2);
    });

    test("returns empty array when no lines match terms", () => {
      const input = `lex: python programming
vec: javascript frameworks`;

      const result = parseQueryables(input, {
        includeLexical: true,
        originalQuery: "rust",
      });

      expect(result).toEqual([]);
    });
  });
});

describe("createFallbackQueryables", () => {
  test("creates fallback with lex when includeLexical is true", () => {
    const result = createFallbackQueryables("rust programming", true);

    expect(result).toEqual([
      { type: "hyde", text: "Information about rust programming" },
      { type: "lex", text: "rust programming" },
      { type: "vec", text: "rust programming" },
    ]);
  });

  test("creates fallback without lex when includeLexical is false", () => {
    const result = createFallbackQueryables("rust programming", false);

    expect(result).toEqual([
      { type: "hyde", text: "Information about rust programming" },
      { type: "vec", text: "rust programming" },
    ]);
  });
});
