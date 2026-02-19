/**
 * AnkiConnect Client
 *
 * This module provides a TypeScript client for the AnkiConnect HTTP API.
 * AnkiConnect is an Anki plugin that exposes Anki functionality via a REST API.
 *
 * Protocol: JSON-RPC over HTTP at localhost:8765
 * Each request must include:
 *   - action: the API method name
 *   - version: API version (we use 6)
 *   - params: optional method parameters
 */

// =============================================================================
// Types
// =============================================================================

/**
 * A single field in an Anki note
 */
export interface AnkiField {
  value: string;
  order: number;
}

/**
 * Full note information returned by notesInfo
 */
export interface AnkiNote {
  noteId: number;
  modelName: string;
  tags: string[];
  fields: Record<string, AnkiField>;
  mod: number;  // Modification timestamp (Unix seconds)
}

/**
 * AnkiConnect API response wrapper
 */
interface AnkiResponse<T> {
  result: T;
  error: string | null;
}

/**
 * Configuration for AnkiConnect connection
 */
export interface AnkiConnectConfig {
  host?: string;
  port?: number;
  timeout?: number;
}

// =============================================================================
// Constants
// =============================================================================

const DEFAULT_HOST = "localhost";
const DEFAULT_PORT = 8765;
const DEFAULT_TIMEOUT = 30000;
const API_VERSION = 6;

// =============================================================================
// Core API Functions
// =============================================================================

/**
 * Make a request to the AnkiConnect API
 */
async function ankiRequest<T>(
  action: string,
  params?: Record<string, unknown>,
  config: AnkiConnectConfig = {}
): Promise<T> {
  const host = config.host ?? DEFAULT_HOST;
  const port = config.port ?? DEFAULT_PORT;
  const timeout = config.timeout ?? DEFAULT_TIMEOUT;
  const url = `http://${host}:${port}`;

  const body = JSON.stringify({
    action,
    version: API_VERSION,
    params: params ?? {},
  });

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`AnkiConnect HTTP error: ${response.status} ${response.statusText}`);
    }

    const data = (await response.json()) as AnkiResponse<T>;

    if (data.error) {
      throw new Error(`AnkiConnect error: ${data.error}`);
    }

    return data.result;
  } catch (error) {
    clearTimeout(timeoutId);

    if (error instanceof Error) {
      if (error.name === "AbortError") {
        throw new Error(`AnkiConnect timeout after ${timeout}ms`);
      }
      // Connection refused - Anki not running
      if (error.message.includes("ECONNREFUSED") || error.message.includes("fetch failed")) {
        throw new Error(
          "Could not connect to AnkiConnect. Make sure:\n" +
          "  1. Anki desktop is running\n" +
          "  2. AnkiConnect plugin is installed (code: 2055492159)\n" +
          "  3. AnkiConnect is listening on localhost:8765"
        );
      }
    }
    throw error;
  }
}

/**
 * Test connection to AnkiConnect
 * Returns true if connection is successful
 */
export async function testConnection(config?: AnkiConnectConfig): Promise<boolean> {
  try {
    // requestPermission is a simple no-op that tests connectivity
    await ankiRequest("requestPermission", {}, config);
    return true;
  } catch {
    return false;
  }
}

/**
 * Get AnkiConnect version information
 */
export async function getVersion(config?: AnkiConnectConfig): Promise<number> {
  return ankiRequest<number>("version", {}, config);
}

/**
 * List all deck names
 */
export async function listDecks(config?: AnkiConnectConfig): Promise<string[]> {
  return ankiRequest<string[]>("deckNames", {}, config);
}

/**
 * List all note type (model) names
 */
export async function listNoteTypes(config?: AnkiConnectConfig): Promise<string[]> {
  return ankiRequest<string[]>("modelNames", {}, config);
}

/**
 * Find note IDs matching an Anki search query
 *
 * Query syntax examples:
 *   - "deck:MyDeck"
 *   - "deck:MyDeck note:Basic"
 *   - "deck:MyDeck tag:important"
 *   - "deck:*" (all decks)
 */
export async function findNotes(
  query: string,
  config?: AnkiConnectConfig
): Promise<number[]> {
  return ankiRequest<number[]>("findNotes", { query }, config);
}

/**
 * Get detailed information about notes by their IDs
 *
 * AnkiConnect can handle ~1000 notes per call efficiently.
 * For larger batches, use getNotesInfoBatched.
 */
export async function getNotesInfo(
  noteIds: number[],
  config?: AnkiConnectConfig
): Promise<AnkiNote[]> {
  if (noteIds.length === 0) return [];
  return ankiRequest<AnkiNote[]>("notesInfo", { notes: noteIds }, config);
}

/**
 * Get notes info in batches to handle large collections
 *
 * @param noteIds - Array of note IDs to fetch
 * @param batchSize - Number of notes per batch (default: 500)
 * @param onProgress - Optional callback for progress updates
 */
export async function getNotesInfoBatched(
  noteIds: number[],
  batchSize: number = 500,
  onProgress?: (fetched: number, total: number) => void,
  config?: AnkiConnectConfig
): Promise<AnkiNote[]> {
  const results: AnkiNote[] = [];

  for (let i = 0; i < noteIds.length; i += batchSize) {
    const batch = noteIds.slice(i, i + batchSize);
    const notes = await getNotesInfo(batch, config);
    results.push(...notes);

    if (onProgress) {
      onProgress(results.length, noteIds.length);
    }
  }

  return results;
}

/**
 * Get all notes from a specific deck
 */
export async function getNotesByDeck(
  deckName: string,
  config?: AnkiConnectConfig
): Promise<AnkiNote[]> {
  // Escape special characters in deck name for Anki query
  const escapedDeck = deckName.replace(/"/g, '\\"');
  const noteIds = await findNotes(`"deck:${escapedDeck}"`, config);
  return getNotesInfo(noteIds, config);
}

/**
 * Build an Anki search query from filter options
 */
export function buildAnkiQuery(options: {
  decks?: string[];
  noteTypes?: string[];
  tags?: string[];
}): string {
  const parts: string[] = [];

  // Deck filter (OR logic for multiple decks)
  if (options.decks && options.decks.length > 0) {
    if (options.decks.length === 1) {
      parts.push(`"deck:${options.decks[0]}"`);
    } else {
      // Multiple decks: (deck:A OR deck:B)
      const deckParts = options.decks.map(d => `"deck:${d}"`);
      parts.push(`(${deckParts.join(" OR ")})`);
    }
  }

  // Note type filter (OR logic)
  if (options.noteTypes && options.noteTypes.length > 0) {
    if (options.noteTypes.length === 1) {
      parts.push(`"note:${options.noteTypes[0]}"`);
    } else {
      const noteParts = options.noteTypes.map(n => `"note:${n}"`);
      parts.push(`(${noteParts.join(" OR ")})`);
    }
  }

  // Tag filter (OR logic, supports wildcards)
  if (options.tags && options.tags.length > 0) {
    if (options.tags.length === 1) {
      parts.push(`"tag:${options.tags[0]}"`);
    } else {
      const tagParts = options.tags.map(t => `"tag:${t}"`);
      parts.push(`(${tagParts.join(" OR ")})`);
    }
  }

  // If no filters, return all notes
  return parts.length > 0 ? parts.join(" ") : "deck:*";
}

// =============================================================================
// Content Conversion Utilities
// =============================================================================

/**
 * Strip HTML tags and decode HTML entities
 * Preserves line breaks where meaningful
 */
export function stripHtml(html: string): string {
  if (!html) return "";

  let text = html;

  // Convert <br> and block elements to newlines
  text = text.replace(/<br\s*\/?>/gi, "\n");
  text = text.replace(/<\/(p|div|li|tr)>/gi, "\n");
  text = text.replace(/<(p|div|ul|ol|table)[^>]*>/gi, "\n");

  // Remove remaining HTML tags
  text = text.replace(/<[^>]+>/g, "");

  // Decode common HTML entities
  text = text.replace(/&nbsp;/g, " ");
  text = text.replace(/&lt;/g, "<");
  text = text.replace(/&gt;/g, ">");
  text = text.replace(/&amp;/g, "&");
  text = text.replace(/&quot;/g, '"');
  text = text.replace(/&#39;/g, "'");
  text = text.replace(/&apos;/g, "'");

  // Normalize whitespace (but preserve intentional line breaks)
  text = text.replace(/[ \t]+/g, " ");
  text = text.replace(/\n[ \t]+/g, "\n");
  text = text.replace(/[ \t]+\n/g, "\n");
  text = text.replace(/\n{3,}/g, "\n\n");

  return text.trim();
}

/**
 * Get the deck path as a filesystem-like path
 * Converts Anki's "::" separator to "/"
 */
export function deckToPath(deckName: string): string {
  return deckName.replace(/::/g, "/");
}

/**
 * Convert an Anki note to markdown format for indexing
 *
 * Format:
 * ```markdown
 * # {First field as title}
 *
 * **Tags:** tag1, tag2
 *
 * ## {Field 1 Name}
 * {Field 1 content}
 *
 * ## {Field 2 Name}
 * {Field 2 content}
 * ```
 */
export function ankiNoteToMarkdown(note: AnkiNote): { content: string; title: string } {
  // Sort fields by their order
  const sortedFields = Object.entries(note.fields)
    .sort(([, a], [, b]) => a.order - b.order);

  // First field is typically the "front" or question
  const firstField = sortedFields[0];
  const title = firstField ? stripHtml(firstField[1].value).split("\n")[0]?.slice(0, 100) || "Untitled" : "Untitled";

  const lines: string[] = [];

  // Title (first line of first field, truncated)
  lines.push(`# ${title}`);
  lines.push("");

  // Model/note type info
  lines.push(`**Type:** ${note.modelName}`);

  // Tags
  if (note.tags.length > 0) {
    lines.push(`**Tags:** ${note.tags.join(", ")}`);
  }
  lines.push("");

  // All fields as sections
  for (const [fieldName, field] of sortedFields) {
    const content = stripHtml(field.value);
    if (content) {
      lines.push(`## ${fieldName}`);
      lines.push(content);
      lines.push("");
    }
  }

  return {
    content: lines.join("\n").trim(),
    title,
  };
}

/**
 * Generate a document path for an Anki note
 *
 * Format: {deck-path}/{noteId}.anki
 * Example: "Programming/Rust/1234567890.anki"
 */
export function getAnkiNotePath(note: AnkiNote, deckName: string): string {
  const deckPath = deckToPath(deckName);
  return `${deckPath}/${note.noteId}.anki`;
}

/**
 * Get the deck name for a note by querying its cards
 * (Notes don't directly store deck info; cards do)
 *
 * For simplicity in indexing, we use the query's deck filter
 * rather than looking up each note's actual deck.
 */
export async function getNoteDeck(
  noteId: number,
  config?: AnkiConnectConfig
): Promise<string | null> {
  // Get cards for this note
  const cardIds = await ankiRequest<number[]>("findCards", { query: `nid:${noteId}` }, config);
  if (cardIds.length === 0) return null;

  // Get card info (includes deck name)
  const cardInfo = await ankiRequest<Array<{ deckName: string }>>(
    "cardsInfo",
    { cards: [cardIds[0]] },
    config
  );

  return cardInfo[0]?.deckName ?? null;
}

// =============================================================================
// Batch Operations for Indexing
// =============================================================================

/**
 * Result of fetching notes for indexing
 */
export interface FetchedNote {
  note: AnkiNote;
  deckName: string;
  path: string;
  markdown: { content: string; title: string };
}

/**
 * Fetch all notes matching filter criteria, enriched with deck info and markdown
 *
 * This is the main entry point for indexing Anki collections.
 */
export async function fetchNotesForIndexing(
  options: {
    decks?: string[];
    noteTypes?: string[];
    tags?: string[];
  },
  onProgress?: (message: string) => void,
  config?: AnkiConnectConfig
): Promise<FetchedNote[]> {
  // Build and execute query
  const query = buildAnkiQuery(options);
  onProgress?.(`Searching Anki: ${query}`);

  const noteIds = await findNotes(query, config);
  if (noteIds.length === 0) {
    return [];
  }

  onProgress?.(`Found ${noteIds.length} notes, fetching details...`);

  // Fetch notes in batches
  const notes = await getNotesInfoBatched(
    noteIds,
    500,
    (fetched, total) => onProgress?.(`Fetched ${fetched}/${total} notes`),
    config
  );

  // For each note, we need to determine its deck
  // This is expensive (one API call per note), so we use a heuristic:
  // If filtering by specific decks, use that; otherwise query per note
  onProgress?.(`Processing ${notes.length} notes...`);

  const results: FetchedNote[] = [];

  // If we filtered by a single deck, use that as the deck name
  if (options.decks && options.decks.length === 1) {
    const deckName = options.decks[0]!;
    for (const note of notes) {
      const path = getAnkiNotePath(note, deckName);
      const markdown = ankiNoteToMarkdown(note);
      results.push({ note, deckName, path, markdown });
    }
  } else {
    // Need to look up deck for each note
    // Batch queries to avoid SQLite expression tree depth limit (max ~1000 OR clauses)
    const BATCH_SIZE = 500;
    const noteToDeck = new Map<number, string>();

    for (let i = 0; i < noteIds.length; i += BATCH_SIZE) {
      const batchIds = noteIds.slice(i, i + BATCH_SIZE);
      const cardQuery = batchIds.map(id => `nid:${id}`).join(" OR ");

      onProgress?.(`Looking up decks: ${Math.min(i + BATCH_SIZE, noteIds.length)}/${noteIds.length}`);

      const cardIds = await ankiRequest<number[]>("findCards", { query: cardQuery }, config);

      // Get card info for this batch
      const cardsInfo = await ankiRequest<Array<{ note: number; deckName: string }>>(
        "cardsInfo",
        { cards: cardIds },
        config
      );

      // Build note -> deck mapping (use first card's deck)
      for (const card of cardsInfo) {
        if (!noteToDeck.has(card.note)) {
          noteToDeck.set(card.note, card.deckName);
        }
      }
    }

    for (const note of notes) {
      const deckName = noteToDeck.get(note.noteId) ?? "Unknown";
      const path = getAnkiNotePath(note, deckName);
      const markdown = ankiNoteToMarkdown(note);
      results.push({ note, deckName, path, markdown });
    }
  }

  onProgress?.(`Processed ${results.length} notes`);
  return results;
}
