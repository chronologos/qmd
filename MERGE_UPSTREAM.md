# Merging Upstream Changes (jj workflow)

This repo is a fork of https://github.com/tobi/qmd. This guide documents how to merge upstream changes using Jujutsu (jj).

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

# Find common ancestor
jj log -r 'heads(ancestors(main) & ancestors(main@upstream))'
```

The `~` operator means "ancestors of left excluding ancestors of right" - useful for seeing what's new on each side.

## Step 3: Create merge commit

```sh
jj new main main@upstream -m "Merge upstream/main"
```

This creates a new commit with two parents. If there are conflicts, jj will list them.

## Step 4: Resolve conflicts

jj uses a unique conflict format with these markers:

```
<<<<<<< conflict N of M
+++++++ commit_id "commit message"     # One side's addition
content from one side
%%%%%%% diff from: base_commit         # Shows what changed
\\\\\\\        to: other_commit
-removed line
+added line
>>>>>>> conflict N of M ends
```

**Reading the markers:**
- `+++++++` block: content added by one commit
- `%%%%%%%` block: a diff showing what the other side changed from the base
- Sometimes you see both additions (both sides added different content)

**Resolution strategy:**
1. Read both sides' changes carefully
2. Determine which changes to keep (often both, merged together)
3. Delete all conflict markers and write the resolved content
4. Use `jj status` to verify no conflicts remain

**Common patterns:**
- **Import conflicts**: Usually merge both sets of imports
- **Function conflicts**: Keep your refactored version if it's intentional design
- **Config/constant conflicts**: Choose based on your requirements (e.g., batch size)
- **New code from upstream**: Usually keep it (tests, new features)

## Step 5: Verify the merge

```sh
# Check no conflict markers remain
rg '<<<<<<< conflict' src/

# Check jj status shows no conflicts
jj status

# Install deps and test
bun install
bun src/qmd.ts --help

# Optionally run tests
bun test
```

## Step 6: Update bookmark and push

```sh
# Move main bookmark to the merge commit
jj bookmark set main -r @

# Push to origin
jj git push
```

## Tips

### Abandoning a bad merge

If the merge goes wrong:

```sh
jj abandon
jj new main  # Go back to clean state
```

### Viewing what upstream changed in specific files

```sh
# See the diff for a specific commit
jj diff -r main@upstream

# Compare branches at specific files
jj diff --from main --to main@upstream -- src/llm.ts
```

### Session management note

This fork adds remote LLM support. Upstream's `withLLMSession` is tightly coupled to `LlamaCpp`. When merging session management code, add guards for remote LLM compatibility or adapt the session manager to work with the `LLM` interface.

### Keeping track of divergent features

| This Fork | Upstream |
|-----------|----------|
| Remote LLM provider (`llm-remote.ts`) | Session management (`withLLMSession`) |
| `parseQueryables` helper in `llm-types.ts` | Inline parsing with `hasQueryTerm` validation |
| Batch size 64 | Batch size 32 |
| Anki collection support | Finetune infrastructure |

When merging, preserve fork-specific features while adopting upstream improvements.
