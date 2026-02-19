/**
 * Anki Collection Provider (fork-only overlay)
 *
 * Manages Anki collections as a fork overlay: config, database, indexing, CLI.
 * All Anki-specific logic lives here; qmd.ts has only ~30 lines of hooks.
 *
 * Architecture:
 *   - Reads/writes same YAML config as collections.ts (source: "anki" entries)
 *   - Creates anki_metadata table lazily via CREATE TABLE IF NOT EXISTS
 *   - Uses store.ts exports for document CRUD (insertContent, insertDocument, etc.)
 *   - Uses anki.ts for AnkiConnect communication
 */

import type { Database } from "./db.js";
import {
  hashContent,
  insertContent,
  insertDocument,
  findActiveDocument,
  updateDocument,
  deactivateDocument,
  getActiveDocumentPaths,
} from "./store.js";
import {
  loadConfig,
  saveConfig,
} from "./collections.js";
import {
  testConnection,
  listDecks,
  fetchNotesForIndexing,
} from "./anki.js";

// =============================================================================
// Types
// =============================================================================

export interface AnkiCollectionConfig {
  source: "anki";
  decks?: string[];
  note_types?: string[];
  tags?: string[];
  context?: Record<string, string>;
  includeByDefault?: boolean;
}

// =============================================================================
// Config helpers
// =============================================================================

/** Cast a raw YAML collection entry to a loosely-typed record. */
function rawEntry(name: string): Record<string, unknown> | undefined {
  const config = loadConfig();
  return config.collections[name] as Record<string, unknown> | undefined;
}

/** Convert a raw YAML entry into a typed AnkiCollectionConfig. */
function toAnkiConfig(raw: Record<string, unknown>): AnkiCollectionConfig {
  return {
    source: "anki",
    decks: raw.decks as string[] | undefined,
    note_types: raw.note_types as string[] | undefined,
    tags: raw.tags as string[] | undefined,
    context: raw.context as Record<string, string> | undefined,
    includeByDefault: raw.includeByDefault as boolean | undefined,
  };
}

/**
 * Check if a collection is an Anki collection by reading YAML config.
 */
export function isAnkiCollection(collectionName: string): boolean {
  return rawEntry(collectionName)?.source === "anki";
}

/**
 * Get the Anki-specific config for a collection.
 * Returns null if not found or not an Anki collection.
 */
export function getAnkiCollectionConfig(name: string): AnkiCollectionConfig | null {
  const raw = rawEntry(name);
  if (!raw || raw.source !== "anki") return null;
  return toAnkiConfig(raw);
}

/**
 * List all Anki collections from YAML config.
 */
export function listAnkiCollections(): Array<{ name: string; config: AnkiCollectionConfig }> {
  const config = loadConfig();
  const results: Array<{ name: string; config: AnkiCollectionConfig }> = [];

  for (const [name, coll] of Object.entries(config.collections)) {
    const raw = coll as unknown as Record<string, unknown>;
    if (raw.source === "anki") {
      results.push({ name, config: toAnkiConfig(raw) });
    }
  }

  return results;
}

/**
 * Add an Anki collection to YAML config.
 */
export function addAnkiCollection(
  name: string,
  opts: { decks?: string[]; noteTypes?: string[]; tags?: string[] }
): void {
  const config = loadConfig();

  if (config.collections[name]) {
    throw new Error(`Collection "${name}" already exists`);
  }

  // Store as a raw object — YAML doesn't enforce TS types
  const entry: Record<string, unknown> = { source: "anki" };
  if (opts.decks?.length) entry.decks = opts.decks;
  if (opts.noteTypes?.length) entry.note_types = opts.noteTypes;
  if (opts.tags?.length) entry.tags = opts.tags;

  config.collections[name] = entry as any;
  saveConfig(config);
}

// =============================================================================
// Database helpers (anki_metadata table)
// =============================================================================

/**
 * Ensure the anki_metadata table exists. Safe to call multiple times.
 */
export function ensureAnkiTables(db: Database): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS anki_metadata (
      collection TEXT NOT NULL,
      note_id INTEGER NOT NULL,
      mod_time INTEGER NOT NULL,
      hash TEXT NOT NULL,
      PRIMARY KEY (collection, note_id)
    )
  `);
}

export function getAnkiMetadata(
  db: Database,
  collection: string,
  noteId: number
): { mod_time: number; hash: string } | null {
  const row = db.prepare(
    `SELECT mod_time, hash FROM anki_metadata WHERE collection = ? AND note_id = ?`
  ).get(collection, noteId) as { mod_time: number; hash: string } | undefined;
  return row ?? null;
}

export function getAllAnkiMetadata(
  db: Database,
  collection: string
): Map<number, { mod_time: number; hash: string }> {
  const rows = db.prepare(
    `SELECT note_id, mod_time, hash FROM anki_metadata WHERE collection = ?`
  ).all(collection) as Array<{ note_id: number; mod_time: number; hash: string }>;

  const map = new Map<number, { mod_time: number; hash: string }>();
  for (const row of rows) {
    map.set(row.note_id, { mod_time: row.mod_time, hash: row.hash });
  }
  return map;
}

export function upsertAnkiMetadata(
  db: Database,
  collection: string,
  noteId: number,
  modTime: number,
  hash: string
): void {
  db.prepare(`
    INSERT INTO anki_metadata (collection, note_id, mod_time, hash)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(collection, note_id) DO UPDATE SET
      mod_time = excluded.mod_time,
      hash = excluded.hash
  `).run(collection, noteId, modTime, hash);
}

export function deleteAnkiMetadata(db: Database, collection: string, noteId: number): void {
  db.prepare(
    `DELETE FROM anki_metadata WHERE collection = ? AND note_id = ?`
  ).run(collection, noteId);
}

/**
 * Find note IDs that exist in anki_metadata but are NOT in the current set
 * (i.e., deleted from Anki since last index).
 */
export function getDeletedAnkiNoteIds(
  db: Database,
  collection: string,
  currentNoteIds: Set<number>
): number[] {
  const allMeta = getAllAnkiMetadata(db, collection);
  return [...allMeta.keys()].filter(id => !currentNoteIds.has(id));
}

export function clearAnkiMetadata(db: Database, collection: string): void {
  db.prepare(`DELETE FROM anki_metadata WHERE collection = ?`).run(collection);
}

// =============================================================================
// Indexing
// =============================================================================

export async function indexAnkiCollection(
  db: Database,
  collectionName: string,
  config: AnkiCollectionConfig,
  progressCallback?: (msg: string) => void
): Promise<{ indexed: number; updated: number; unchanged: number; removed: number }> {
  ensureAnkiTables(db);

  // Fetch notes from Anki
  const fetchedNotes = await fetchNotesForIndexing(
    {
      decks: config.decks,
      noteTypes: config.note_types,
      tags: config.tags,
    },
    progressCallback
  );

  // Load existing metadata for comparison
  const existingMeta = getAllAnkiMetadata(db, collectionName);
  const now = new Date().toISOString();

  let indexed = 0;
  let updated = 0;
  let unchanged = 0;

  const currentNoteIds = new Set<number>();

  for (const fetched of fetchedNotes) {
    const noteId = fetched.note.noteId;
    currentNoteIds.add(noteId);

    const { content, title } = fetched.markdown;
    const path = fetched.path;
    const contentHash = await hashContent(content);
    const modTime = fetched.note.mod;

    const existing = existingMeta.get(noteId);

    if (existing && existing.hash === contentHash && existing.mod_time === modTime) {
      unchanged++;
      continue;
    }

    insertContent(db, contentHash, content, now);

    const existingDoc = findActiveDocument(db, collectionName, path);
    if (existingDoc) {
      updateDocument(db, existingDoc.id, title, contentHash, now);
      updated++;
    } else {
      insertDocument(db, collectionName, path, title, contentHash, now, now);
      indexed++;
    }

    upsertAnkiMetadata(db, collectionName, noteId, modTime, contentHash);

    progressCallback?.(`Indexed ${indexed + updated}/${fetchedNotes.length} notes...`);
  }

  // Handle deleted notes
  const deletedNoteIds = getDeletedAnkiNoteIds(db, collectionName, currentNoteIds);
  const activePaths = deletedNoteIds.length > 0
    ? getActiveDocumentPaths(db, collectionName)
    : [];

  for (const noteId of deletedNoteIds) {
    const notePath = activePaths.find(p => p.endsWith(`/${noteId}.anki`));
    if (notePath) {
      deactivateDocument(db, collectionName, notePath);
    }
    deleteAnkiMetadata(db, collectionName, noteId);
  }

  return { indexed, updated, unchanged, removed: deletedNoteIds.length };
}

// =============================================================================
// CLI handlers
// =============================================================================

interface Colors {
  bold: string;
  reset: string;
  dim: string;
  cyan: string;
  green: string;
  yellow: string;
  red: string;
}

export async function handleAnkiCommand(args: string[], c: Colors): Promise<void> {
  const subcommand = args[0];

  switch (subcommand) {
    case "test": {
      try {
        const connected = await testConnection();
        if (connected) {
          console.log(`${c.green}✓${c.reset} Connected to AnkiConnect`);
        } else {
          console.log(`${c.red}✗${c.reset} AnkiConnect is not responding`);
          process.exit(1);
        }
      } catch (err) {
        console.error(`${c.red}✗${c.reset} Failed to connect: ${err}`);
        process.exit(1);
      }
      break;
    }

    case "decks": {
      try {
        const decks = await listDecks();
        console.log(`${c.bold}Anki Decks (${decks.length}):${c.reset}`);
        for (const deck of decks) {
          console.log(`  ${deck}`);
        }
      } catch (err) {
        console.error(`${c.red}✗${c.reset} Failed to list decks: ${err}`);
        console.error(`${c.dim}Is Anki running with AnkiConnect installed?${c.reset}`);
        process.exit(1);
      }
      break;
    }

    default:
      console.log("Usage:");
      console.log("  qmd anki test   - Test AnkiConnect connection");
      console.log("  qmd anki decks  - List available Anki decks");
      break;
  }
}

export async function handleAnkiCollectionAdd(
  name: string,
  opts: { decks?: string[]; noteTypes?: string[]; tags?: string[] },
  db: Database,
  c: Colors
): Promise<void> {
  // Test connection first
  try {
    const connected = await testConnection();
    if (!connected) {
      console.error(`${c.red}✗${c.reset} AnkiConnect is not responding. Is Anki running?`);
      process.exit(1);
    }
  } catch (err) {
    console.error(`${c.red}✗${c.reset} Failed to connect to Anki: ${err}`);
    process.exit(1);
  }

  // Add to YAML config
  try {
    addAnkiCollection(name, opts);
  } catch (err) {
    console.error(`${c.red}✗${c.reset} ${err}`);
    process.exit(1);
  }

  const config = getAnkiCollectionConfig(name)!;

  // Show what was created
  console.log(`${c.green}✓${c.reset} Created Anki collection "${c.cyan}${name}${c.reset}"`);
  if (config.decks?.length) {
    console.log(`  ${c.dim}Decks:${c.reset}      ${config.decks.join(", ")}`);
  }
  if (config.note_types?.length) {
    console.log(`  ${c.dim}Note types:${c.reset} ${config.note_types.join(", ")}`);
  }
  if (config.tags?.length) {
    console.log(`  ${c.dim}Tags:${c.reset}       ${config.tags.join(", ")}`);
  }

  // Run initial indexing
  console.log(`\n${c.dim}Indexing...${c.reset}`);
  const result = await indexAnkiCollection(db, name, config, (msg) => {
    process.stderr.write(`\r${msg}        `);
  });
  process.stderr.write("\r" + " ".repeat(60) + "\r");
  console.log(`${c.green}✓${c.reset} Indexed ${result.indexed} notes`);
}

export function formatAnkiCollectionInfo(
  name: string,
  config: AnkiCollectionConfig,
  activeCount: number,
  lastModified: string | null,
  c: Colors
): string {
  const excluded = config.includeByDefault === false;
  const excludeTag = excluded ? ` ${c.yellow}[excluded]${c.reset}` : "";

  const lines: string[] = [];
  lines.push(`${c.cyan}${name}${c.reset} ${c.dim}(source: anki)${c.reset}${excludeTag}`);

  if (config.decks?.length) {
    lines.push(`  ${c.dim}Decks:${c.reset}      ${config.decks.join(", ")}`);
  }
  if (config.note_types?.length) {
    lines.push(`  ${c.dim}Note types:${c.reset} ${config.note_types.join(", ")}`);
  }
  if (config.tags?.length) {
    lines.push(`  ${c.dim}Tags:${c.reset}       ${config.tags.join(", ")}`);
  }

  lines.push(`  ${c.dim}Notes:${c.reset}    ${activeCount}`);

  if (lastModified) {
    const timeAgo = formatTimeAgo(new Date(lastModified));
    lines.push(`  ${c.dim}Updated:${c.reset}  ${timeAgo}`);
  }

  return lines.join("\n");
}

// Simple time-ago formatter (avoids importing from qmd.ts)
function formatTimeAgo(date: Date): string {
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHour = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHour / 24);

  if (diffDay > 0) return `${diffDay}d ago`;
  if (diffHour > 0) return `${diffHour}h ago`;
  if (diffMin > 0) return `${diffMin}m ago`;
  return "just now";
}
