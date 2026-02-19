# Merging Upstream Changes (jj workflow)

This repo is a fork of https://github.com/tobi/qmd with remote LLM support. This guide documents how to merge upstream changes using Jujutsu (jj).

## Overlay Architecture

Fork-specific code lives in **fork-only files** that upstream doesn't have. This means upstream merges should produce **zero conflicts** (or at most a trivial one in `qmd.ts`).

**Fork-only files (never conflict):**

| File | Purpose |
|------|---------|
| `src/llm-remote.ts` | RemoteLLM class — imports from upstream's `llm.ts` |
| `src/llm-provider.ts` | Backend selection — reads config, injects via `setDefaultLlamaCpp()` |
| `src/test-config.ts` | Test environment helpers (local vs remote LLM) |
| `test/llm-remote.test.ts` | Fork-specific tests (pure functions + integration) |
| `src/anki.ts` | AnkiConnect JSON-RPC client |
| `src/anki-provider.ts` | Anki collection management, indexing, CLI handlers |
| `test/anki-provider.test.ts` | Anki overlay tests |
| `server/deploy.py` | Tailscale+systemd deployment for vLLM + embed/rerank |
| `MERGE_UPSTREAM.md` | This file |

**Minimally modified upstream file:**

| File | Changes |
|------|---------|
| `src/qmd.ts` | 2 imports, 7 parseArgs options, `initLLMProvider()` call, Anki command routing + hooks |
| `.gitignore` | 1 line: `!MERGE_UPSTREAM.md` |

## Prerequisites

Ensure the upstream remote is configured:

```sh
jj git remote list
# Should show: upstream https://github.com/tobi/qmd.git
# If not:
jj git remote add upstream https://github.com/tobi/qmd.git
```

## Step 1: Fetch upstream

```sh
jj git fetch --remote upstream
```

This creates/updates `main@upstream` bookmark tracking upstream's main branch.

## Step 2: Analyze divergence

```sh
# Commits in upstream not in your fork
jj log -r 'main@upstream ~ main'

# Commits in your fork not in upstream
jj log -r 'main ~ main@upstream'
```

## Step 3: Create merge commit

```sh
jj new main main@upstream -m "Merge upstream/main"
```

If there are conflicts, they should only be in `qmd.ts` (the one file we modify). Resolve by keeping both: upstream's changes plus our 3 fork additions (import, parseArgs options, initLLMProvider call).

## Step 4: Resolve conflicts (if any)

The only likely conflict is in `src/qmd.ts`. Look for our fork additions:

1. **LLM import** (after `from "./llm.js"`): `import { initLLMProvider } from "./llm-provider.js";`
2. **Anki import** (after LLM import): `import { isAnkiCollection, ... } from "./anki-provider.js";`
3. **parseArgs options** (in `parseCLI()`): `remote`, `local`, `remote-url`, `anki`, `deck`, `note-type`, `tag`
4. **Init call** (after `setConfigIndexName`): `initLLMProvider({ ... })`
5. **Anki hooks**: `case "anki"` routing, `--anki` in collection add, Anki branch in `updateCollections()`, Anki display in `collectionList()`

Keep both upstream's changes and our additions.

```sh
# Verify no conflict markers remain
rg '<<<<<<< conflict' src/

# Check jj status shows no conflicts
jj status
```

## Step 5: Verify the merge

```sh
bun install
bun src/qmd.ts --help

# Run upstream tests
npx vitest run --reporter=verbose test/

# Run fork tests
npx vitest run --reporter=verbose test/llm-remote.test.ts
```

## Step 6: Update bookmark and push

```sh
jj bookmark set main -r @
jj git push
```

## Tips

### Abandoning a bad merge

```sh
jj abandon
jj new main  # Go back to clean state
```

### If upstream changes the LLM interface

If upstream modifies `llm.ts` types that `llm-remote.ts` imports (e.g., `LLM`, `EmbedOptions`, `GenerateOptions`), update `llm-remote.ts` to match. Since we import types rather than re-export them, this is straightforward.

### Keeping track of divergent features

| This Fork | Upstream |
|-----------|----------|
| Remote LLM provider (`llm-remote.ts`) | Local-only LlamaCpp |
| Provider selection (`llm-provider.ts`) | Single LlamaCpp singleton |
| Server deployment (`server/deploy.py`) | No server infrastructure |
| Anki collection support (`anki.ts`, `anki-provider.ts`) | No Anki integration |
| Test config (`test-config.ts`) | CI-skip via `process.env.CI` |
| Minimal `qmd.ts` modification (LLM + Anki hooks) | No remote/Anki flags |
