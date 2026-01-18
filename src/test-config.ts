/**
 * test-config.ts - Shared test configuration for LLM backend selection
 *
 * Environment variables:
 *   QMD_TEST_REMOTE=1        - Use remote LLM backend (reads config from ~/.config/qmd/index.yml)
 *   QMD_TEST_LOCAL=1         - Use local LLM backend (downloads models)
 *   QMD_REMOTE_URL           - Override remote embed/rerank URL
 *   QMD_REMOTE_GENERATION_URL - Override remote generation URL
 *
 * If neither QMD_TEST_REMOTE nor QMD_TEST_LOCAL is set, LLM tests will be skipped.
 *
 * Usage:
 *   import { setupTestLLM, getTestEmbedDimensions, shouldSkipLLMTests } from "./test-config";
 *
 *   describe("LLM tests", () => {
 *     beforeAll(async () => {
 *       if (shouldSkipLLMTests()) return;
 *       await setupTestLLM();
 *     });
 *
 *     test("embed works", async () => {
 *       if (shouldSkipLLMTests()) return;
 *       // ... test code
 *     });
 *   });
 */

import {
  setDefaultLLMConfig,
  getDefaultLlamaCpp,
  disposeDefaultLlamaCpp,
  type LLM,
} from "./llm.js";
import { getRemoteConfig } from "./collections.js";
import type { RemoteLLMConfig } from "./llm-remote.js";

// Model dimension mapping
const MODEL_DIMENSIONS: Record<string, number> = {
  // Local models
  embeddinggemma: 768,
  "hf:phr00g/embeddinggemma-GGUF/embeddinggemma_4.Q8_0.gguf": 768,

  // Remote models (Qwen3 stack)
  "Qwen/Qwen3-Embedding-4B": 2560,
  "Qwen/Qwen3-Embedding-0.6B": 1024,

  // Alternatives
  "nomic-ai/nomic-embed-text-v1.5": 768,
};

// Default dimensions by provider
const DEFAULT_LOCAL_DIM = 768;
const DEFAULT_REMOTE_DIM = 2560;

let testLLMInitialized = false;
let testEmbedModel: string | undefined;

/**
 * Check if LLM tests should be skipped (no explicit configuration)
 */
export function shouldSkipLLMTests(): boolean {
  const hasRemote = process.env.QMD_TEST_REMOTE === "1" || process.env.QMD_TEST_REMOTE === "true";
  const hasLocal = process.env.QMD_TEST_LOCAL === "1" || process.env.QMD_TEST_LOCAL === "true";
  return !hasRemote && !hasLocal;
}

/**
 * Check if tests should use remote LLM
 */
export function shouldUseRemoteLLM(): boolean {
  return process.env.QMD_TEST_REMOTE === "1" || process.env.QMD_TEST_REMOTE === "true";
}

/**
 * Get the embed model name for current configuration
 */
export function getTestEmbedModel(): string {
  if (testEmbedModel) return testEmbedModel;

  if (shouldUseRemoteLLM()) {
    const remoteConfig = getRemoteConfig();
    return remoteConfig?.models?.embed || "Qwen/Qwen3-Embedding-4B";
  }

  return "embeddinggemma";
}

/**
 * Get the embedding dimensions for the current test configuration
 */
export function getTestEmbedDimensions(): number {
  const model = getTestEmbedModel();
  return MODEL_DIMENSIONS[model] || (shouldUseRemoteLLM() ? DEFAULT_REMOTE_DIM : DEFAULT_LOCAL_DIM);
}

/**
 * Setup the LLM singleton for tests.
 * Call this in beforeAll() before any LLM operations.
 * Returns null if LLM tests should be skipped.
 */
export async function setupTestLLM(): Promise<LLM | null> {
  if (shouldSkipLLMTests()) {
    console.log("[test-config] LLM tests SKIPPED - set QMD_TEST_REMOTE=1 or QMD_TEST_LOCAL=1 to run");
    return null;
  }

  if (testLLMInitialized) {
    return getDefaultLlamaCpp();
  }

  if (shouldUseRemoteLLM()) {
    const yamlConfig = getRemoteConfig();

    const remoteConfig: RemoteLLMConfig = {
      url: process.env.QMD_REMOTE_URL || yamlConfig?.url,
      embedUrl: process.env.QMD_REMOTE_URL || yamlConfig?.embed_url,
      generationUrl: process.env.QMD_REMOTE_GENERATION_URL || yamlConfig?.generation_url,
      apiKey: yamlConfig?.api_key,
      models: {
        embed: yamlConfig?.models?.embed || "Qwen/Qwen3-Embedding-4B",
        generate: yamlConfig?.models?.generate || "Qwen/Qwen3-8B",
        rerank: yamlConfig?.models?.rerank || "Qwen/Qwen3-Reranker-4B",
      },
      timeout: 60000, // Longer timeout for tests
      retries: 2,
    };

    testEmbedModel = remoteConfig.models?.embed;

    console.log(`[test-config] Using REMOTE LLM: ${remoteConfig.embedUrl}`);
    setDefaultLLMConfig({ provider: "remote", remoteConfig });
  } else {
    console.log("[test-config] Using LOCAL LLM (LlamaCpp)");
    setDefaultLLMConfig({ provider: "local" });
  }

  testLLMInitialized = true;
  return getDefaultLlamaCpp();
}

/**
 * Cleanup LLM resources. Call in afterAll().
 */
export async function cleanupTestLLM(): Promise<void> {
  if (testLLMInitialized) {
    await disposeDefaultLlamaCpp();
  }
  testLLMInitialized = false;
  testEmbedModel = undefined;
}

/**
 * Get a description of the current test LLM configuration (for logging)
 */
export function getTestLLMDescription(): string {
  if (shouldSkipLLMTests()) {
    return "SKIPPED (no config)";
  }
  if (shouldUseRemoteLLM()) {
    const model = getTestEmbedModel();
    return `Remote (${model}, ${getTestEmbedDimensions()}d)`;
  }
  return `Local (embeddinggemma, 768d)`;
}
