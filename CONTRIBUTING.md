## HT Fork Management

> This section applies to the **Heiervang Technologies fork** only. For upstream contribution guidelines, see below.

This fork follows the [HT Fork Management Guide](https://github.com/orgs/heiervang-technologies/discussions/3). Key points:

### Branch conventions

- **`main`** — Clean fast-forward mirror of upstream `main`. Never commit directly.
- **`ht`** — Default branch. All HT-specific changes live here, rebased on top of `main`.
- **Feature branches** — Create from `ht`, squash-merge back into `ht` via PR.

### Sync workflow

1. Fast-forward `main` to match upstream
2. Rebase `ht` onto the updated `main`
3. Force-push `ht` (notify teammates so they can recover local branches)

### Commit standards

- Use conventional commits: `feat(scope):`, `fix(scope):`, `docs:`, etc.
- One logical change per commit — no merge commits from sync (rebase instead)
- Linear history on `ht`

### Questions & discussion

For all inquiries about this fork — questions, ideas, bug reports, or feature requests — use the [HT Discussions page](https://github.com/orgs/heiervang-technologies/discussions).

---

# Contributing to vLLM-Omni

You may find information about contributing to vLLM-Omni on [Contributing](https://vllm-omni.readthedocs.io/en/latest/contributing/)
