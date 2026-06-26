/**
 * anki-provider.test.ts - Tests for the Anki overlay
 *
 * Pure function tests always run. Integration tests require:
 *   QMD_TEST_ANKI=1 bun test test/anki-provider.test.ts
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach } from "vitest";
import type { Database } from "../src/db.js";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import YAML from "yaml";
import { createStore } from "../src/store.js";
import {
  isAnkiCollection,
  getAnkiCollectionConfig,
  listAnkiCollections,
  addAnkiCollection,
  ensureAnkiTables,
  getAnkiMetadata,
  getAllAnkiMetadata,
  upsertAnkiMetadata,
  deleteAnkiMetadata,
  getDeletedAnkiNoteIds,
  clearAnkiMetadata,
  formatAnkiCollectionInfo,
} from "../src/anki-provider.js";

// =============================================================================
// Test Utilities
// =============================================================================

let testDir: string;
let testConfigDir: string;

async function setupTestConfig(collections: Record<string, unknown> = {}): Promise<void> {
  const configPrefix = join(testDir, `config-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  testConfigDir = await mkdtemp(configPrefix);
  process.env.QMD_CONFIG_DIR = testConfigDir;

  const config = { collections };
  await writeFile(join(testConfigDir, "index.yml"), YAML.stringify(config));
}

function createTestDb(): Database {
  const dbPath = join(testDir, `test-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`);
  const store = createStore(dbPath);
  return store.db;
}

async function cleanupConfigDir(): Promise<void> {
  if (!testConfigDir) return;
  await rm(testConfigDir, { recursive: true, force: true }).catch(() => {});
}

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-anki-test-"));
});

afterAll(async () => {
  delete process.env.QMD_CONFIG_DIR;
  await rm(testDir, { recursive: true, force: true }).catch(() => {});
});

// =============================================================================
// Config helpers
// =============================================================================

describe("config helpers", () => {
  beforeEach(async () => {
    await cleanupConfigDir();
  });

  test("isAnkiCollection returns true for anki source", async () => {
    await setupTestConfig({
      flashcards: { source: "anki", decks: ["Default"] },
    });
    expect(isAnkiCollection("flashcards")).toBe(true);
  });

  test("isAnkiCollection returns false for filesystem collection", async () => {
    await setupTestConfig({
      docs: { path: "/tmp/docs", pattern: "**/*.md" },
    });
    expect(isAnkiCollection("docs")).toBe(false);
  });

  test("isAnkiCollection returns false for nonexistent collection", async () => {
    await setupTestConfig({});
    expect(isAnkiCollection("nope")).toBe(false);
  });

  test("getAnkiCollectionConfig returns config for anki collection", async () => {
    await setupTestConfig({
      flashcards: { source: "anki", decks: ["Default", "Japanese"], note_types: ["Basic"], tags: ["study"] },
    });

    const config = getAnkiCollectionConfig("flashcards");
    expect(config).not.toBeNull();
    expect(config!.source).toBe("anki");
    expect(config!.decks).toEqual(["Default", "Japanese"]);
    expect(config!.note_types).toEqual(["Basic"]);
    expect(config!.tags).toEqual(["study"]);
  });

  test("getAnkiCollectionConfig returns null for filesystem collection", async () => {
    await setupTestConfig({
      docs: { path: "/tmp/docs", pattern: "**/*.md" },
    });
    expect(getAnkiCollectionConfig("docs")).toBeNull();
  });

  test("listAnkiCollections returns only anki collections", async () => {
    await setupTestConfig({
      flashcards: { source: "anki", decks: ["Default"] },
      docs: { path: "/tmp/docs", pattern: "**/*.md" },
      study: { source: "anki", tags: ["exam"] },
    });

    const result = listAnkiCollections();
    expect(result).toHaveLength(2);
    expect(result.map(r => r.name).sort()).toEqual(["flashcards", "study"]);
  });

  test("addAnkiCollection adds to YAML config", async () => {
    await setupTestConfig({});

    addAnkiCollection("flashcards", { decks: ["Default"] });

    expect(isAnkiCollection("flashcards")).toBe(true);
    const config = getAnkiCollectionConfig("flashcards");
    expect(config!.decks).toEqual(["Default"]);
  });

  test("addAnkiCollection throws if name exists", async () => {
    await setupTestConfig({
      existing: { path: "/tmp", pattern: "**/*.md" },
    });

    expect(() => addAnkiCollection("existing", {})).toThrow("already exists");
  });
});

// =============================================================================
// Database helpers (anki_metadata)
// =============================================================================

describe("anki_metadata database helpers", () => {
  let db: Database;

  beforeEach(() => {
    db = createTestDb();
    ensureAnkiTables(db);
  });

  test("ensureAnkiTables is idempotent", () => {
    // Should not throw on second call
    ensureAnkiTables(db);
    ensureAnkiTables(db);
  });

  test("upsert and get round-trip", () => {
    upsertAnkiMetadata(db, "flashcards", 12345, 1700000000, "abc123");

    const result = getAnkiMetadata(db, "flashcards", 12345);
    expect(result).toEqual({ mod_time: 1700000000, hash: "abc123" });
  });

  test("get returns null for nonexistent entry", () => {
    expect(getAnkiMetadata(db, "flashcards", 99999)).toBeNull();
  });

  test("upsert updates existing entry", () => {
    upsertAnkiMetadata(db, "flashcards", 12345, 1700000000, "abc123");
    upsertAnkiMetadata(db, "flashcards", 12345, 1700001000, "def456");

    const result = getAnkiMetadata(db, "flashcards", 12345);
    expect(result).toEqual({ mod_time: 1700001000, hash: "def456" });
  });

  test("getAllAnkiMetadata returns all entries for a collection", () => {
    upsertAnkiMetadata(db, "flashcards", 100, 1700000000, "aaa");
    upsertAnkiMetadata(db, "flashcards", 200, 1700000001, "bbb");
    upsertAnkiMetadata(db, "other", 300, 1700000002, "ccc");

    const result = getAllAnkiMetadata(db, "flashcards");
    expect(result.size).toBe(2);
    expect(result.get(100)).toEqual({ mod_time: 1700000000, hash: "aaa" });
    expect(result.get(200)).toEqual({ mod_time: 1700000001, hash: "bbb" });
  });

  test("deleteAnkiMetadata removes entry", () => {
    upsertAnkiMetadata(db, "flashcards", 12345, 1700000000, "abc123");
    deleteAnkiMetadata(db, "flashcards", 12345);

    expect(getAnkiMetadata(db, "flashcards", 12345)).toBeNull();
  });

  test("getDeletedAnkiNoteIds finds notes not in current set", () => {
    upsertAnkiMetadata(db, "flashcards", 100, 1700000000, "aaa");
    upsertAnkiMetadata(db, "flashcards", 200, 1700000001, "bbb");
    upsertAnkiMetadata(db, "flashcards", 300, 1700000002, "ccc");

    const current = new Set([100, 300]);
    const deleted = getDeletedAnkiNoteIds(db, "flashcards", current);
    expect(deleted).toEqual([200]);
  });

  test("getDeletedAnkiNoteIds returns empty when all present", () => {
    upsertAnkiMetadata(db, "flashcards", 100, 1700000000, "aaa");

    const current = new Set([100]);
    const deleted = getDeletedAnkiNoteIds(db, "flashcards", current);
    expect(deleted).toEqual([]);
  });

  test("clearAnkiMetadata removes all entries for collection", () => {
    upsertAnkiMetadata(db, "flashcards", 100, 1700000000, "aaa");
    upsertAnkiMetadata(db, "flashcards", 200, 1700000001, "bbb");
    upsertAnkiMetadata(db, "other", 300, 1700000002, "ccc");

    clearAnkiMetadata(db, "flashcards");

    expect(getAllAnkiMetadata(db, "flashcards").size).toBe(0);
    // Other collection unaffected
    expect(getAllAnkiMetadata(db, "other").size).toBe(1);
  });
});

// =============================================================================
// Display helpers
// =============================================================================

describe("formatAnkiCollectionInfo", () => {
  const noColors = { bold: "", reset: "", dim: "", cyan: "", green: "", yellow: "", red: "" };

  test("formats basic Anki collection info", () => {
    const result = formatAnkiCollectionInfo(
      "flashcards",
      { source: "anki", decks: ["Default"] },
      42,
      null,
      noColors
    );

    expect(result).toContain("flashcards");
    expect(result).toContain("source: anki");
    expect(result).toContain("Default");
    expect(result).toContain("42");
  });

  test("includes excluded tag when includeByDefault is false", () => {
    const result = formatAnkiCollectionInfo(
      "flashcards",
      { source: "anki", includeByDefault: false },
      0,
      null,
      noColors
    );

    expect(result).toContain("[excluded]");
  });
});

// =============================================================================
// Integration tests (require Anki + AnkiConnect running)
// =============================================================================

const SKIP_ANKI = !process.env.QMD_TEST_ANKI;

describe.skipIf(SKIP_ANKI)("Anki integration", () => {
  test("testConnection connects to AnkiConnect", async () => {
    const { testConnection } = await import("../src/anki.js");
    const connected = await testConnection();
    expect(connected).toBe(true);
  });

  test("listDecks returns deck names", async () => {
    const { listDecks } = await import("../src/anki.js");
    const decks = await listDecks();
    expect(Array.isArray(decks)).toBe(true);
    expect(decks.length).toBeGreaterThan(0);
  });
});
