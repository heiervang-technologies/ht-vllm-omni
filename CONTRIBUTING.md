# Contributing to vLLM-Omni

You may find information about contributing to vLLM-Omni on [Contributing](https://vllm-omni.readthedocs.io/en/latest/contributing/)

---

# HT Fork Management Guide

This document describes how Heiervang Technologies maintains forks of upstream open-source projects.

## Branch Structure

Each fork has two key branches:

| Branch | Purpose | Default? |
|--------|---------|----------|
| **`ht`** | HT-specific changes on top of upstream | **Yes** (default branch) |
| **`main`** | Clean mirror of upstream `main` | No |

### `main` — Upstream Mirror

- Always a clean fast-forward of the upstream `main` branch
- **Never** commit HT-specific changes to `main`
- Enable GitHub's "Sync fork" button or use `git fetch upstream && git merge --ff-only upstream/main`
- Used as the merge base when syncing upstream changes into `ht`

### `ht` — Default Branch

- Contains all HT-specific features, fixes, and configuration on top of `main`
- Set as the repo's **default branch** on GitHub so that clones, PRs, and the README all reference it
- All PRs target `ht` unless they're upstream contributions

## Commit Hygiene

The `ht` branch should maintain a **clean, linear history** of logical feature commits on top of `main`:

- One commit per feature or logical change
- No merge commits from sync operations (rebase instead)
- No "fix review feedback" or "address comments" commits — squash them
- No commits for changes that are now in upstream — drop them during rebase

### Commit Message Convention

```
feat(<scope>): short description

Longer explanation if needed.

Co-Authored-By: <author> <email>
```

Scopes match the upstream project's module structure (e.g., `qwen3-tts`, `serving`, `diffusion`).

## Inspecting the Fork Delta

To see what HT-specific changes currently exist on top of upstream:

```bash
# List HT commits
git log main..ht --oneline

# Full diff against upstream
git diff main..ht

# Diff with stat summary
git diff main..ht --stat
```

Run these before syncs and rewrites to audit the current state of HT changes.

## Syncing with Upstream

When upstream `main` advances:

```bash
# 1. Update local main
git checkout main
git fetch upstream
git merge --ff-only upstream/main
git push origin main

# 2. Rebase ht onto updated main
git checkout ht
git rebase main

# 3. Resolve any conflicts, then force-push
git push --force-with-lease origin ht
```

If the rebase is complex, create a backup branch first:

```bash
git branch ht-backup-$(date +%Y%m%d) ht
git push origin ht-backup-$(date +%Y%m%d)
```

### Sync Cadence

- **Before starting a new feature** — rebase `ht` onto the latest `main` so your feature branch starts from a current base
- **When upstream changes affect your work** — if upstream lands changes in files or modules you've modified, sync promptly to catch conflicts early
- **Periodically** — even without immediate need, sync at least weekly to avoid large, painful rebases

### After a Force-Push: Recovering Local Branches

When `ht` is force-pushed (after a rebase or rewrite), teammates with local branches based on the old `ht` need to reset:

```bash
git fetch origin
git checkout ht
git reset --hard origin/ht

# Rebase any local feature branches onto the new ht
git checkout feat/my-feature
git rebase ht
```

## Feature Development

### New Features

1. Create a feature branch from `ht`:
   ```bash
   git checkout -b feat/my-feature ht
   ```
2. Develop, commit, push
3. Open a PR targeting `ht`
4. Squash-merge the PR (GitHub's "Squash and merge" button)
5. Delete the feature branch after merge

### PR Checklist

Each fork should have a PR template with a checkbox:

```markdown
- [ ] If this PR adds or changes HT-specific functionality, update the **HT Fork Changes** section in `README.md`.
```

## README Convention

The `ht` branch README should include an **HT Fork Changes** section near the top, documenting all changes relative to upstream. Group changes by feature area and annotate items that have been upstreamed:

```markdown
## HT Fork Changes

This is the [Heiervang Technologies](https://github.com/heiervang-technologies) fork of <project>. The `ht` branch contains the following changes on top of upstream `main`:

### Feature Area 1
- Description of change
- ~~Change now in upstream~~ *(now in upstream)*

### Feature Area 2
- Description of change
```

## Using Draft PRs as Issues

Since GitHub Issues may be disabled on fork repos, use **draft PRs** to track RFC documents, design discussions, and work items:

- Create a branch with the document (e.g., `docs/my-rfc`)
- Open a **draft PR** targeting `ht`
- Use the PR body for the full document content
- Use PR comments for discussion
- The draft PR is **never intended to be merged** — it serves as a discussion thread
- Close (don't merge) when the discussion is resolved or the work is complete

This gives us issue-like tracking with inline commenting, without needing GitHub Issues enabled.

## History Rewrites

When the `ht` branch history becomes messy (accumulated merge commits, fix-up commits, superseded changes), perform a history rewrite:

### Procedure

1. **Create a protected backup branch**:
   ```bash
   git branch ht-backup-pre-rewrite ht
   git push origin ht-backup-pre-rewrite
   # Add branch protection: no force push, no delete, enforce admins
   gh api repos/<org>/<repo>/branches/ht-backup-pre-rewrite/protection \
     -X PUT --input - <<< '{"required_status_checks":null,"enforce_admins":true,"required_pull_request_reviews":null,"restrictions":null}'
   ```

2. **Create a working branch from `main`**:
   ```bash
   git checkout -b ht-rewrite main
   ```

3. **Build clean commits** by feature group. For each group:
   - Check out unique files from `ht`: `git checkout ht -- <file>`
   - For shared files modified by multiple features, build intermediate states incrementally
   - Commit with a clean message

4. **Verify the final state matches**:
   ```bash
   git diff ht-rewrite..ht  # Should be empty (or only intentionally dropped files)
   ```

5. **Force-push**:
   ```bash
   git checkout ht
   git reset --hard ht-rewrite
   git push --force-with-lease origin ht
   ```

6. **Clean up**: Delete the working branch. Keep the backup branch as a safety net.

### What to Drop During Rewrites

- Merge/sync commits (artifacts of the merge workflow)
- "Fix review feedback" and "address comments" commits (squash into parent)
- Changes now in upstream `main` (redundant after sync)
- Planning documents (move to draft PRs instead)

## Agent Safety Rules

When AI agents (Claude Code, Copilot, etc.) work on HT fork repositories:

- **NEVER** push to, create PRs against, or modify the **upstream** repository unless explicitly instructed by a human. The upstream remote is read-only for syncing purposes.
- **NEVER** push to `main` — it is an upstream mirror and should only be updated via fast-forward merge from `upstream/main`.
- **ALWAYS** target `ht` (not `main`) when creating PRs.
- **ALWAYS** create a backup branch before destructive operations (force push, history rewrite, branch deletion).
- **NEVER** delete protected branches or backup branches.
- When in doubt about whether an action affects upstream or shared state, **ask first**.

## Contributing Back Upstream

When an HT-specific change is general enough to benefit the upstream project:

1. **Create a branch from `main`** (not `ht`):
   ```bash
   git checkout -b upstream/my-contribution main
   ```

2. **Cherry-pick or rewrite** the relevant commits cleanly — no HT-specific references, no unrelated changes:
   ```bash
   git cherry-pick <commit-from-ht>
   # or manually apply the change
   ```

3. **Open a PR against the upstream repo's `main`** following their contribution guidelines

4. **Once merged upstream**, the change will flow back into our `main` on the next sync. Drop the corresponding commit from `ht` during the next rebase (it's now redundant).

5. **Annotate in the README**: mark the change as upstreamed in the HT Fork Changes section:
   ```markdown
   - ~~Change description~~ *(now in upstream)*
   ```

## Setting Up a New Fork

1. Fork the upstream repo on GitHub
2. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/<upstream-org>/<repo>.git
   ```
3. Create and push the `ht` branch:
   ```bash
   git checkout -b ht main
   git push -u origin ht
   ```
4. Set `ht` as default branch:
   ```bash
   gh repo edit --default-branch ht
   ```
5. Add the PR template checkbox for HT fork changes
6. Add the HT Fork Changes section to the README
7. (Optional) Brand the logo with the HT avatar overlay
