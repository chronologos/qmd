#!/usr/bin/env bun
/**
 * QMD MCP Server (Shell Implementation)
 *
 * Alternative MCP server that shells out to the QMD CLI for all operations.
 * This ensures feature parity with CLI and avoids code duplication bugs.
 *
 * Benefits:
 * - Single source of truth (CLI implementation)
 * - No divergence bugs between MCP and CLI
 * - CLI improvements automatically benefit MCP
 * - Simpler MCP code to maintain
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { spawn } from "bun";

// =============================================================================
// Types
// =============================================================================

type SearchResultItem = {
  docid: string;
  file: string;
  title: string;
  score: number;
  context: string | null;
  snippet?: string;
  body?: string;
};

type StatusResult = {
  totalDocuments: number;
  needsEmbedding: number;
  hasVectorIndex: boolean;
  collections: {
    name: string;
    path: string;
    pattern: string;
    documents: number;
    lastUpdated: string;
  }[];
};

type DocumentResult = {
  file: string;
  title: string;
  context?: string;
  body?: string;
  skipped?: boolean;
  reason?: string;
};

// =============================================================================
// Shell Execution Helper
// =============================================================================

/**
 * Execute qmd CLI command and return parsed JSON output
 */
async function execQmd(args: string[]): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  // Get the directory where this script is located
  const scriptDir = import.meta.dir;
  const qmdPath = `${scriptDir}/qmd.ts`;

  const proc = spawn({
    cmd: ["bun", qmdPath, ...args],
    stdout: "pipe",
    stderr: "pipe",
  });

  const stdout = await new Response(proc.stdout).text();
  const stderr = await new Response(proc.stderr).text();
  const exitCode = await proc.exited;

  return { stdout, stderr, exitCode };
}

/**
 * Execute qmd command expecting JSON output
 */
async function execQmdJson<T>(args: string[]): Promise<T> {
  const { stdout, stderr, exitCode } = await execQmd(args);

  if (exitCode !== 0) {
    throw new Error(`qmd command failed: ${stderr || stdout}`);
  }

  try {
    return JSON.parse(stdout) as T;
  } catch {
    throw new Error(`Failed to parse JSON output: ${stdout.slice(0, 200)}`);
  }
}

// =============================================================================
// Helper Functions
// =============================================================================

function formatSearchSummary(results: SearchResultItem[], query: string): string {
  if (results.length === 0) {
    return `No results found for "${query}"`;
  }
  const lines = [`Found ${results.length} result${results.length === 1 ? '' : 's'} for "${query}":\n`];
  for (const r of results) {
    lines.push(`${r.docid} ${Math.round(r.score * 100)}% ${r.file} - ${r.title}`);
  }
  return lines.join('\n');
}

// =============================================================================
// MCP Server
// =============================================================================

export async function startMcpServer(): Promise<void> {
  const server = new McpServer({
    name: "qmd",
    version: "1.0.0",
  });

  // ---------------------------------------------------------------------------
  // Prompt: query guide
  // ---------------------------------------------------------------------------

  server.registerPrompt(
    "query",
    {
      title: "QMD Query Guide",
      description: "How to effectively search your knowledge base with QMD",
    },
    () => ({
      messages: [
        {
          role: "user",
          content: {
            type: "text",
            text: `# QMD - Quick Markdown Search

QMD is your on-device search engine for markdown knowledge bases.

## Available Tools

### 1. search (Fast keyword search)
Best for: Finding documents with specific keywords or phrases.

### 2. vsearch (Semantic search)
Best for: Finding conceptually related content even without exact keyword matches.

### 3. query (Hybrid search - highest quality)
Best for: Important searches where you want the best results.
Combines keyword + semantic search with LLM reranking.

### 4. get (Retrieve document)
Best for: Getting the full content of a document you found.

### 5. multi_get (Retrieve multiple documents)
Best for: Getting content from multiple files at once.

### 6. status (Index info)
Shows collection info, document counts, and embedding status.

## Search Strategy

1. **Start with search** for quick keyword lookups
2. **Use vsearch** when keywords aren't working
3. **Use query** for important searches
4. **Use get/multi_get** to retrieve full documents`,
          },
        },
      ],
    })
  );

  // ---------------------------------------------------------------------------
  // Tool: search (BM25 full-text)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "search",
    {
      title: "Search (BM25)",
      description: "Fast keyword-based full-text search using BM25. Best for finding documents with specific words or phrases.",
      inputSchema: {
        query: z.string().describe("Search query - keywords or phrases to find"),
        limit: z.number().optional().default(10).describe("Maximum number of results (default: 10)"),
        minScore: z.number().optional().default(0).describe("Minimum relevance score 0-1 (default: 0)"),
        collection: z.string().optional().describe("Filter to a specific collection by name"),
      },
    },
    async ({ query, limit, minScore, collection }) => {
      const args = ["search", "--json", "-l", String(limit || 10)];
      if (minScore) args.push("--min-score", String(minScore));
      if (collection) args.push("--collection", collection);
      args.push(query);

      try {
        const results = await execQmdJson<SearchResultItem[]>(args);
        return {
          content: [{ type: "text", text: formatSearchSummary(results, query) }],
          structuredContent: { results },
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `Search failed: ${error instanceof Error ? error.message : String(error)}` }],
          isError: true,
        };
      }
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: vsearch (Vector semantic search)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "vsearch",
    {
      title: "Vector Search (Semantic)",
      description: "Semantic similarity search using vector embeddings. Finds conceptually related content even without exact keyword matches.",
      inputSchema: {
        query: z.string().describe("Natural language query - describe what you're looking for"),
        limit: z.number().optional().default(10).describe("Maximum number of results (default: 10)"),
        minScore: z.number().optional().default(0.3).describe("Minimum relevance score 0-1 (default: 0.3)"),
        collection: z.string().optional().describe("Filter to a specific collection by name"),
      },
    },
    async ({ query, limit, minScore, collection }) => {
      const args = ["vsearch", "--json", "-l", String(limit || 10)];
      if (minScore) args.push("--min-score", String(minScore));
      if (collection) args.push("--collection", collection);
      args.push(query);

      try {
        const results = await execQmdJson<SearchResultItem[]>(args);
        return {
          content: [{ type: "text", text: formatSearchSummary(results, query) }],
          structuredContent: { results },
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `Vector search failed: ${error instanceof Error ? error.message : String(error)}` }],
          isError: true,
        };
      }
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: query (Hybrid with reranking)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "query",
    {
      title: "Hybrid Query (Best Quality)",
      description: "Highest quality search combining BM25 + vector + query expansion + LLM reranking. Slower but most accurate. Use for important searches.",
      inputSchema: {
        query: z.string().describe("Natural language query - describe what you're looking for"),
        limit: z.number().optional().default(10).describe("Maximum number of results (default: 10)"),
        minScore: z.number().optional().default(0).describe("Minimum relevance score 0-1 (default: 0)"),
        collection: z.string().optional().describe("Filter to a specific collection by name"),
      },
    },
    async ({ query, limit, minScore, collection }) => {
      const args = ["query", "--json", "-l", String(limit || 10)];
      if (minScore) args.push("--min-score", String(minScore));
      if (collection) args.push("--collection", collection);
      args.push(query);

      try {
        const results = await execQmdJson<SearchResultItem[]>(args);
        return {
          content: [{ type: "text", text: formatSearchSummary(results, query) }],
          structuredContent: { results },
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `Query failed: ${error instanceof Error ? error.message : String(error)}` }],
          isError: true,
        };
      }
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: get (Retrieve document)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "get",
    {
      title: "Get Document",
      description: "Retrieve the full content of a document by its file path or docid. Use paths or docids (#abc123) from search results. Suggests similar files if not found.",
      inputSchema: {
        file: z.string().describe("File path or docid from search results (e.g., 'pages/meeting.md', '#abc123', or 'pages/meeting.md:100' to start at line 100)"),
        fromLine: z.number().optional().describe("Start from this line number (1-indexed)"),
        maxLines: z.number().optional().describe("Maximum number of lines to return"),
        lineNumbers: z.boolean().optional().default(false).describe("Add line numbers to output (format: 'N: content')"),
      },
    },
    async ({ file, fromLine, maxLines, lineNumbers }) => {
      const args = ["get", "--json"];
      if (lineNumbers) args.push("--line-numbers");

      // Handle :line suffix
      let lookup = file;
      if (fromLine) {
        lookup = `${file}:${fromLine}`;
      }
      args.push(lookup);

      try {
        const result = await execQmdJson<DocumentResult>(args);

        let text = result.body || "";
        if (result.context) {
          text = `<!-- Context: ${result.context} -->\n\n` + text;
        }

        return {
          content: [{
            type: "resource",
            resource: {
              uri: `qmd://${encodeURIComponent(result.file)}`,
              name: result.file,
              title: result.title,
              mimeType: "text/markdown",
              text,
            },
          }],
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `Get failed: ${error instanceof Error ? error.message : String(error)}` }],
          isError: true,
        };
      }
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: multi_get (Retrieve multiple documents)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "multi_get",
    {
      title: "Multi-Get Documents",
      description: "Retrieve multiple documents by glob pattern (e.g., 'journals/2025-05*.md') or comma-separated list. Skips files larger than maxBytes.",
      inputSchema: {
        pattern: z.string().describe("Glob pattern or comma-separated list of file paths"),
        maxLines: z.number().optional().describe("Maximum lines per file"),
        maxBytes: z.number().optional().default(10240).describe("Skip files larger than this (default: 10240 = 10KB)"),
        lineNumbers: z.boolean().optional().default(false).describe("Add line numbers to output (format: 'N: content')"),
      },
    },
    async ({ pattern, maxLines, maxBytes, lineNumbers }) => {
      const args = ["multi-get", "--json"];
      if (maxLines) args.push("-l", String(maxLines));
      if (maxBytes) args.push("--max-bytes", String(maxBytes));
      if (lineNumbers) args.push("--line-numbers");
      args.push(pattern);

      try {
        const results = await execQmdJson<DocumentResult[]>(args);

        const content: ({ type: "text"; text: string } | { type: "resource"; resource: { uri: string; name: string; title?: string; mimeType: string; text: string } })[] = [];

        for (const result of results) {
          if (result.skipped) {
            content.push({
              type: "text",
              text: `[SKIPPED: ${result.file} - ${result.reason}. Use 'get' with file="${result.file}" to retrieve.]`,
            });
            continue;
          }

          let text = result.body || "";
          if (result.context) {
            text = `<!-- Context: ${result.context} -->\n\n` + text;
          }

          content.push({
            type: "resource",
            resource: {
              uri: `qmd://${encodeURIComponent(result.file)}`,
              name: result.file,
              title: result.title,
              mimeType: "text/markdown",
              text,
            },
          });
        }

        return { content };
      } catch (error) {
        return {
          content: [{ type: "text", text: `Multi-get failed: ${error instanceof Error ? error.message : String(error)}` }],
          isError: true,
        };
      }
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: status (Index status)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "status",
    {
      title: "Index Status",
      description: "Show the status of the QMD index: collections, document counts, and health information.",
      inputSchema: {},
    },
    async () => {
      try {
        const { stdout, exitCode } = await execQmd(["status", "--json"]);

        if (exitCode !== 0) {
          throw new Error("Status command failed");
        }

        // Parse JSON if available, otherwise return text
        let status: StatusResult;
        try {
          status = JSON.parse(stdout);
        } catch {
          // Fallback: return the CLI text output
          return {
            content: [{ type: "text", text: stdout }],
          };
        }

        const summary = [
          `QMD Index Status:`,
          `  Total documents: ${status.totalDocuments}`,
          `  Needs embedding: ${status.needsEmbedding}`,
          `  Vector index: ${status.hasVectorIndex ? 'yes' : 'no'}`,
          `  Collections: ${status.collections.length}`,
        ];

        for (const col of status.collections) {
          summary.push(`    - ${col.path} (${col.documents} docs)`);
        }

        return {
          content: [{ type: "text", text: summary.join('\n') }],
          structuredContent: status,
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `Status failed: ${error instanceof Error ? error.message : String(error)}` }],
          isError: true,
        };
      }
    }
  );

  // ---------------------------------------------------------------------------
  // Connect via stdio
  // ---------------------------------------------------------------------------

  const transport = new StdioServerTransport();
  await server.connect(transport);

  // Keep the process alive
  await new Promise<void>((resolve) => {
    process.stdin.on('close', resolve);
    process.stdin.on('end', resolve);
  });
}

// Run if this is the main module
if (import.meta.main) {
  startMcpServer().catch(console.error);
}
