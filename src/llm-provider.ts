/**
 * llm-provider.ts — Backend selection for QMD
 *
 * Fork-only file. Reads ~/.config/qmd/index.yml for a `remote:` section
 * and injects RemoteLLM via setDefaultLlamaCpp().
 *
 * Called once at CLI startup before any LLM operations.
 *
 * Config format in index.yml:
 * ```yaml
 * remote:
 *   url: "http://host:8000"           # fallback if specific URLs not set
 *   embed_url: "http://host:8001"     # embed + rerank service
 *   generation_url: "http://host:8000" # vLLM for text generation
 *   api_key: "${QMD_REMOTE_API_KEY}"  # optional, supports env var expansion
 *   models:
 *     embed: Qwen/Qwen3-Embedding-4B
 *     generate: Qwen/Qwen3-4B
 *     rerank: Qwen/Qwen3-Reranker-4B
 * ```
 */

import { setDefaultLlamaCpp } from "./llm.js";
import { RemoteLLM } from "./llm-remote.js";
import type { RemoteLLMConfig } from "./llm-remote.js";
import { existsSync, readFileSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import YAML from "yaml";

// =============================================================================
// Types
// =============================================================================

export interface LLMProviderOptions {
  /** Force local backend regardless of config */
  forceLocal?: boolean;
  /** Force remote backend (fails if no config) */
  forceRemote?: boolean;
  /** Override remote URL at runtime */
  remoteUrl?: string;
  /** Config index name (default: "index") */
  indexName?: string;
}

interface RemoteYamlConfig {
  url?: string;
  embed_url?: string;
  generation_url?: string;
  api_key?: string;
  models?: {
    embed?: string;
    generate?: string;
    rerank?: string;
  };
}

// =============================================================================
// State
// =============================================================================

let initialized = false;
let activeBackend: "local" | "remote" = "local";

// =============================================================================
// Helpers
// =============================================================================

/**
 * Expand environment variable references in a string.
 * Supports ${VAR_NAME} syntax.
 */
function expandEnvVars(value: string): string {
  return value.replace(/\$\{([^}]+)\}/g, (_, varName) => {
    return process.env[varName] || "";
  });
}

/**
 * Get the config file path for a given index name.
 */
function getConfigFilePath(indexName: string): string {
  const configDir =
    process.env.QMD_CONFIG_DIR ||
    (process.env.XDG_CONFIG_HOME
      ? join(process.env.XDG_CONFIG_HOME, "qmd")
      : join(homedir(), ".config", "qmd"));
  return join(configDir, `${indexName}.yml`);
}

/**
 * Read and parse the remote config section from the YAML file.
 * Returns null if no remote config exists.
 */
function readRemoteConfig(indexName: string): RemoteYamlConfig | null {
  const configPath = getConfigFilePath(indexName);

  if (!existsSync(configPath)) {
    return null;
  }

  try {
    const content = readFileSync(configPath, "utf-8");
    const config = YAML.parse(content);

    if (!config?.remote) {
      return null;
    }

    return config.remote as RemoteYamlConfig;
  } catch {
    // Silently ignore parse errors — fall back to local
    return null;
  }
}

/**
 * Convert YAML remote config to RemoteLLMConfig.
 */
function buildRemoteLLMConfig(
  yaml: RemoteYamlConfig,
  overrideUrl?: string
): RemoteLLMConfig {
  // Resolve URLs: specific > override > fallback `url` field
  const baseUrl = overrideUrl || yaml.url || "";
  const generationUrl = yaml.generation_url || baseUrl;
  const embedUrl = yaml.embed_url || baseUrl;

  if (!generationUrl || !embedUrl) {
    throw new Error(
      "Remote config requires at least 'url' or both 'generation_url' and 'embed_url'"
    );
  }

  // Expand env vars in API key
  const apiKey = yaml.api_key ? expandEnvVars(yaml.api_key) : undefined;

  return {
    generationUrl,
    embedUrl,
    apiKey: apiKey || undefined, // Convert empty string to undefined
    models: yaml.models,
  };
}

// =============================================================================
// Public API
// =============================================================================

/**
 * Initialize the LLM provider. Called once at CLI startup.
 *
 * Provider selection priority:
 * 1. --local flag → always local
 * 2. --remote or --remote-url flag → remote
 * 3. Config file remote: section → remote
 * 4. Default → local
 *
 * Safe to call multiple times — subsequent calls are no-ops.
 */
export function initLLMProvider(options: LLMProviderOptions = {}): void {
  // Guard against double-init
  if (initialized) {
    return;
  }
  initialized = true;

  const indexName = options.indexName || "index";

  // Priority 1: --local always wins
  if (options.forceLocal) {
    activeBackend = "local";
    return;
  }

  // Priority 2: --remote or --remote-url forces remote
  if (options.forceRemote || options.remoteUrl) {
    const yamlConfig = readRemoteConfig(indexName);
    if (!yamlConfig && !options.remoteUrl) {
      throw new Error(
        "No remote configuration found in config file. " +
          "Add a 'remote:' section to ~/.config/qmd/index.yml or use --remote-url."
      );
    }

    const config = buildRemoteLLMConfig(
      yamlConfig || {},
      options.remoteUrl
    );
    const remoteLlm = new RemoteLLM(config);
    setDefaultLlamaCpp(remoteLlm as any);
    activeBackend = "remote";
    return;
  }

  // Priority 3: Check config file for remote section
  const yamlConfig = readRemoteConfig(indexName);
  if (yamlConfig) {
    try {
      const config = buildRemoteLLMConfig(yamlConfig);
      const remoteLlm = new RemoteLLM(config);
      setDefaultLlamaCpp(remoteLlm as any);
      activeBackend = "remote";
    } catch (error: any) {
      // Invalid remote config — fall back to local with a warning
      process.stderr.write(
        `QMD Warning: Invalid remote config, falling back to local: ${error.message}\n`
      );
      activeBackend = "local";
    }
    return;
  }

  // Priority 4: Default to local
  activeBackend = "local";
}

/**
 * Get the currently active backend type.
 */
export function getActiveBackend(): "local" | "remote" {
  return activeBackend;
}

/**
 * Reset provider state (for testing).
 */
export function resetLLMProvider(): void {
  initialized = false;
  activeBackend = "local";
}
