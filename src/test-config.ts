/**
 * test-config.ts — Test environment configuration for LLM tests
 *
 * Fork-only file. Helps tests select between local and remote LLM.
 *
 * Environment variables:
 *   QMD_TEST_REMOTE=1            — Use remote LLM backend
 *   QMD_TEST_LOCAL=1             — Use local LLM backend (downloads models)
 *   QMD_TEST_GENERATION_URL=<url> — Override generation service URL
 *   QMD_TEST_EMBED_URL=<url>     — Override embed service URL
 *   Neither set                  — Skip LLM tests
 *
 * When QMD_TEST_REMOTE=1 without explicit URLs, reads ~/.config/qmd/index.yml
 * (same config file used by the CLI).
 */

import { getDefaultLlamaCpp, setDefaultLlamaCpp, disposeDefaultLlamaCpp } from "./llm.js";
import { RemoteLLM } from "./llm-remote.js";
import { initLLMProvider, resetLLMProvider } from "./llm-provider.js";

export function shouldUseRemoteLLM(): boolean {
  return process.env.QMD_TEST_REMOTE === "1";
}

export function shouldSkipLLMTests(): boolean {
  return !process.env.QMD_TEST_REMOTE && !process.env.QMD_TEST_LOCAL;
}

export function getTestEmbedDimensions(): number {
  // embeddinggemma (local) = 768, Qwen3-Embedding-4B (remote) = 2560
  return shouldUseRemoteLLM() ? 2560 : 768;
}

export function getTestLLMDescription(): string {
  if (shouldUseRemoteLLM()) return "Remote LLM backend";
  if (process.env.QMD_TEST_LOCAL) return "Local LlamaCpp backend";
  return "No LLM backend (tests skipped)";
}

export async function setupTestLLM(): Promise<ReturnType<typeof getDefaultLlamaCpp> | null> {
  if (shouldSkipLLMTests()) return null;

  if (shouldUseRemoteLLM()) {
    const hasExplicitUrls = process.env.QMD_TEST_GENERATION_URL || process.env.QMD_TEST_EMBED_URL;

    if (hasExplicitUrls) {
      // Explicit URLs override everything
      const remoteLLM = new RemoteLLM({
        generationUrl: process.env.QMD_TEST_GENERATION_URL || "http://localhost:8000",
        embedUrl: process.env.QMD_TEST_EMBED_URL || "http://localhost:8001",
      });
      setDefaultLlamaCpp(remoteLLM as any);
    } else {
      // Use config file (same as CLI: ~/.config/qmd/index.yml)
      resetLLMProvider();
      initLLMProvider({ forceRemote: true });
    }
  }

  return getDefaultLlamaCpp();
}

export async function cleanupTestLLM(): Promise<void> {
  await disposeDefaultLlamaCpp();
}
