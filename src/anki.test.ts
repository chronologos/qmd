/**
 * anki.test.ts - Unit tests for the AnkiConnect client
 *
 * Run with: bun test src/anki.test.ts
 *
 * These tests are split into two categories:
 *   - Unit tests: Test content conversion, query building (no Anki required)
 *   - Integration tests: Require Anki + AnkiConnect (QMD_TEST_ANKI=1)
 *
 * To run integration tests:
 *   1. Start Anki desktop
 *   2. Install AnkiConnect plugin (code: 2055492159)
 *   3. QMD_TEST_ANKI=1 bun test src/anki.test.ts
 */

import { describe, test, expect, beforeAll } from "bun:test";
import {
  stripHtml,
  deckToPath,
  ankiNoteToMarkdown,
  getAnkiNotePath,
  buildAnkiQuery,
  testConnection,
  listDecks,
  listNoteTypes,
  findNotes,
  getNotesInfo,
  type AnkiNote,
} from "./anki.js";

// =============================================================================
// Test Configuration
// =============================================================================

function shouldRunAnkiTests(): boolean {
  return process.env.QMD_TEST_ANKI === "1";
}

// =============================================================================
// Unit Tests: HTML Stripping
// =============================================================================

describe("stripHtml", () => {
  test("strips basic HTML tags", () => {
    expect(stripHtml("<p>Hello</p>")).toBe("Hello");
    expect(stripHtml("<b>Bold</b> text")).toBe("Bold text");
    expect(stripHtml("<span class='foo'>Text</span>")).toBe("Text");
  });

  test("converts br tags to newlines", () => {
    expect(stripHtml("Line 1<br>Line 2")).toBe("Line 1\nLine 2");
    expect(stripHtml("Line 1<br/>Line 2")).toBe("Line 1\nLine 2");
    expect(stripHtml("Line 1<br />Line 2")).toBe("Line 1\nLine 2");
  });

  test("converts block elements to newlines", () => {
    expect(stripHtml("<p>Para 1</p><p>Para 2</p>")).toBe("Para 1\n\nPara 2");
    expect(stripHtml("<div>Div 1</div><div>Div 2</div>")).toBe("Div 1\n\nDiv 2");
  });

  test("decodes HTML entities", () => {
    expect(stripHtml("&lt;code&gt;")).toBe("<code>");
    expect(stripHtml("&amp;&amp;")).toBe("&&");
    expect(stripHtml("&quot;quoted&quot;")).toBe('"quoted"');
    expect(stripHtml("it&#39;s")).toBe("it's");
    // Note: &nbsp; becomes space, which gets trimmed at edges
    expect(stripHtml("hello&nbsp;world")).toBe("hello world");
  });

  test("normalizes whitespace", () => {
    expect(stripHtml("too   many    spaces")).toBe("too many spaces");
    expect(stripHtml("  leading and trailing  ")).toBe("leading and trailing");
    expect(stripHtml("line 1\n\n\n\nline 2")).toBe("line 1\n\nline 2");
  });

  test("handles empty input", () => {
    expect(stripHtml("")).toBe("");
    expect(stripHtml("   ")).toBe("");
  });

  test("handles complex Anki card content", () => {
    const html = `
      <div style="font-size: 20px;">
        <b>What is a closure?</b>
      </div>
      <br>
      <p>A function that has access to variables from its outer scope.</p>
    `;
    const result = stripHtml(html);
    expect(result).toContain("What is a closure?");
    expect(result).toContain("A function that has access to variables");
    expect(result).not.toContain("<div");
    expect(result).not.toContain("font-size");
  });
});

// =============================================================================
// Unit Tests: Path Conversion
// =============================================================================

describe("deckToPath", () => {
  test("converts single-level deck names", () => {
    expect(deckToPath("Default")).toBe("Default");
    expect(deckToPath("MyDeck")).toBe("MyDeck");
  });

  test("converts hierarchical deck names", () => {
    expect(deckToPath("Programming::Rust")).toBe("Programming/Rust");
    expect(deckToPath("CS::Algorithms::Sorting")).toBe("CS/Algorithms/Sorting");
  });

  test("handles empty string", () => {
    expect(deckToPath("")).toBe("");
  });
});

describe("getAnkiNotePath", () => {
  const mockNote: AnkiNote = {
    noteId: 1234567890,
    modelName: "Basic",
    tags: [],
    fields: {},
    mod: 1000000,
  };

  test("generates correct path for simple deck", () => {
    const path = getAnkiNotePath(mockNote, "Default");
    expect(path).toBe("Default/1234567890.anki");
  });

  test("generates correct path for hierarchical deck", () => {
    const path = getAnkiNotePath(mockNote, "Programming::Rust");
    expect(path).toBe("Programming/Rust/1234567890.anki");
  });
});

// =============================================================================
// Unit Tests: Query Building
// =============================================================================

describe("buildAnkiQuery", () => {
  test("returns deck:* when no filters specified", () => {
    expect(buildAnkiQuery({})).toBe("deck:*");
  });

  test("builds single deck query", () => {
    const query = buildAnkiQuery({ decks: ["MyDeck"] });
    expect(query).toBe('"deck:MyDeck"');
  });

  test("builds multiple deck query with OR", () => {
    const query = buildAnkiQuery({ decks: ["Deck1", "Deck2"] });
    expect(query).toBe('("deck:Deck1" OR "deck:Deck2")');
  });

  test("builds note type query", () => {
    const query = buildAnkiQuery({ noteTypes: ["Basic"] });
    expect(query).toBe('"note:Basic"');
  });

  test("builds tag query", () => {
    const query = buildAnkiQuery({ tags: ["important"] });
    expect(query).toBe('"tag:important"');
  });

  test("combines multiple filters with AND", () => {
    const query = buildAnkiQuery({
      decks: ["MyDeck"],
      noteTypes: ["Basic"],
      tags: ["important"],
    });
    expect(query).toContain('"deck:MyDeck"');
    expect(query).toContain('"note:Basic"');
    expect(query).toContain('"tag:important"');
  });

  test("supports wildcard tags", () => {
    const query = buildAnkiQuery({ tags: ["thinking::*"] });
    expect(query).toBe('"tag:thinking::*"');
  });
});

// =============================================================================
// Unit Tests: Markdown Conversion
// =============================================================================

// Mock data based on real AnkiConnect responses
const MOCK_BASIC_NOTE: AnkiNote = {
  noteId: 1558046207039,
  modelName: "Basic",
  tags: ["thinking::models"],
  fields: {
    Front: {
      value: "Ray Dalio's 5 step approach to achieve goals",
      order: 0,
    },
    Back: {
      value: "1. Have clear <b>goals</b><div>2. <b>Identify</b> and don't tolerate problems</div><div>3. <b>Diagnose</b> and find root causes</div><div>4. <b>Design</b> a plan</div><div>5. <b>Push</b> through</div>",
      order: 1,
    },
  },
  mod: 1701756745,
};

const MOCK_CLOZE_NOTE: AnkiNote = {
  noteId: 1557810413077,
  modelName: "Cloze",
  tags: ["p::go"],
  fields: {
    Text: {
      value: "A map is a {{c1::reference}} to a hashtable. Thus, the zero value for a map type is {{c1::nil}}. Most operations on maps, including lookup, delete, len, and range loops, are {{c1::safe}} to perform on the zero value.",
      order: 0,
    },
    Text2: {
      value: "",
      order: 1,
    },
    Extra: {
      value: "But storing to a nil map causes a panic: You must allocate the map before you can store into it.<br>",
      order: 2,
    },
  },
  mod: 1756252328,
};

describe("ankiNoteToMarkdown", () => {
  test("converts real Basic note to markdown", () => {
    const result = ankiNoteToMarkdown(MOCK_BASIC_NOTE);

    expect(result.title).toBe("Ray Dalio's 5 step approach to achieve goals");
    expect(result.content).toContain("# Ray Dalio's 5 step approach to achieve goals");
    expect(result.content).toContain("**Type:** Basic");
    expect(result.content).toContain("**Tags:** thinking::models");
    expect(result.content).toContain("## Front");
    expect(result.content).toContain("## Back");
    // HTML should be stripped
    expect(result.content).toContain("Have clear goals");
    expect(result.content).toContain("Identify and don't tolerate problems");
    expect(result.content).not.toContain("<b>");
    expect(result.content).not.toContain("<div>");
  });

  test("converts real Cloze note to markdown", () => {
    const result = ankiNoteToMarkdown(MOCK_CLOZE_NOTE);

    // Title is truncated to 100 chars from first line
    expect(result.title.length).toBeLessThanOrEqual(100);
    expect(result.title).toContain("A map is a {{c1::reference}}");
    expect(result.content).toContain("**Type:** Cloze");
    expect(result.content).toContain("**Tags:** p::go");
    // Cloze syntax should be preserved for searchability
    expect(result.content).toContain("{{c1::reference}}");
    expect(result.content).toContain("{{c1::nil}}");
    expect(result.content).toContain("{{c1::safe}}");
    // Empty Text2 field should not create a section
    expect(result.content).not.toContain("## Text2");
    // Extra field content should be included
    expect(result.content).toContain("But storing to a nil map causes a panic");
  });

  test("handles HTML content in fields", () => {
    const note: AnkiNote = {
      noteId: 456,
      modelName: "Basic",
      tags: [],
      fields: {
        Front: { value: "<b>Bold question?</b>", order: 0 },
        Back: { value: "<p>Answer with <code>code</code></p>", order: 1 },
      },
      mod: 1000000,
    };

    const result = ankiNoteToMarkdown(note);

    expect(result.title).toBe("Bold question?");
    expect(result.content).toContain("Bold question?");
    expect(result.content).toContain("Answer with code");
    expect(result.content).not.toContain("<b>");
    expect(result.content).not.toContain("<code>");
  });

  test("handles empty fields", () => {
    const note: AnkiNote = {
      noteId: 101,
      modelName: "Basic",
      tags: [],
      fields: {
        Front: { value: "Question", order: 0 },
        Back: { value: "", order: 1 },
      },
      mod: 1000000,
    };

    const result = ankiNoteToMarkdown(note);

    expect(result.content).toContain("## Front");
    // Empty Back field should not create a section
    expect(result.content).not.toContain("## Back");
  });

  test("handles notes without tags", () => {
    const note: AnkiNote = {
      noteId: 102,
      modelName: "Basic",
      tags: [],
      fields: {
        Front: { value: "Question", order: 0 },
        Back: { value: "Answer", order: 1 },
      },
      mod: 1000000,
    };

    const result = ankiNoteToMarkdown(note);

    expect(result.content).not.toContain("**Tags:**");
  });

  test("truncates long titles", () => {
    const longQuestion = "A".repeat(150);
    const note: AnkiNote = {
      noteId: 103,
      modelName: "Basic",
      tags: [],
      fields: {
        Front: { value: longQuestion, order: 0 },
        Back: { value: "Short answer", order: 1 },
      },
      mod: 1000000,
    };

    const result = ankiNoteToMarkdown(note);

    expect(result.title.length).toBeLessThanOrEqual(100);
  });

  test("generates correct path for real note", () => {
    const path = getAnkiNotePath(MOCK_BASIC_NOTE, "Salience");
    expect(path).toBe("Salience/1558046207039.anki");
  });

  test("generates correct path for hierarchical deck", () => {
    const path = getAnkiNotePath(MOCK_CLOZE_NOTE, "Great Works of Art::Artists");
    expect(path).toBe("Great Works of Art/Artists/1557810413077.anki");
  });
});

// =============================================================================
// Integration Tests: AnkiConnect API
// =============================================================================

describe("AnkiConnect Integration", () => {
  beforeAll(() => {
    if (!shouldRunAnkiTests()) {
      console.log("[anki.test.ts] Skipping integration tests (set QMD_TEST_ANKI=1 to enable)");
    }
  });

  test("testConnection returns true when Anki is running", async () => {
    if (!shouldRunAnkiTests()) return;

    const connected = await testConnection();
    expect(connected).toBe(true);
  });

  test("listDecks returns array of deck names", async () => {
    if (!shouldRunAnkiTests()) return;

    const decks = await listDecks();
    expect(Array.isArray(decks)).toBe(true);
    // Default deck should always exist
    expect(decks.includes("Default")).toBe(true);
  });

  test("listNoteTypes returns array of model names", async () => {
    if (!shouldRunAnkiTests()) return;

    const noteTypes = await listNoteTypes();
    expect(Array.isArray(noteTypes)).toBe(true);
    // Basic and Cloze are built-in note types
    expect(noteTypes.includes("Basic")).toBe(true);
  });

  test("findNotes returns array of note IDs", async () => {
    if (!shouldRunAnkiTests()) return;

    // Search for all notes (may return empty if no notes exist)
    const noteIds = await findNotes("deck:*");
    expect(Array.isArray(noteIds)).toBe(true);
    // All IDs should be numbers
    noteIds.forEach(id => expect(typeof id).toBe("number"));
  });

  test("getNotesInfo returns note details", async () => {
    if (!shouldRunAnkiTests()) return;

    // First find some notes
    const noteIds = await findNotes("deck:*");
    if (noteIds.length === 0) {
      console.log("[anki.test.ts] No notes found, skipping getNotesInfo test");
      return;
    }

    // Get info for first note
    const firstNoteId = noteIds[0]!;
    const notes = await getNotesInfo([firstNoteId]);
    expect(notes.length).toBe(1);

    const note = notes[0]!;
    expect(note.noteId).toBe(firstNoteId);
    expect(typeof note.modelName).toBe("string");
    expect(Array.isArray(note.tags)).toBe(true);
    expect(typeof note.fields).toBe("object");
    expect(typeof note.mod).toBe("number");
  });

  test("handles empty noteIds array gracefully", async () => {
    if (!shouldRunAnkiTests()) return;

    const notes = await getNotesInfo([]);
    expect(notes).toEqual([]);
  });
});
