# Contributing to Cannondale Business Intelligence Agent

Welcome! This document outlines the workflow and conventions for contributing to this project.

---

## 🌿 Branch Structure

| Branch | Purpose | Who Pushes |
|--------|---------|------------|
| `main` | Stable, release-ready code only | **No direct pushes** — only via PR from `dev` |
| `dev` | Active integration branch | Maintainers merge feature branches here |
| `feature/*`, `fix/*`, `chore/*` | Individual work items | Contributors |

### Branch Protection

- **`main` is protected**: Direct pushes are disabled.
- All changes to `main` must come via a **pull request from `dev`**.
- PRs to `main` require **at least one approval** before merging.
- **Force pushes to `main` are disabled** — history rewrites are not allowed.

---

## 🔄 Pull Request Workflow

```
1. Create feature branch from dev
       ↓
2. Implement + test locally
       ↓
3. Open PR targeting dev
       ↓
4. Review + iterate
       ↓
5. Merge to dev
       ↓
6. (Periodically) PR dev → main for release
       ↓
7. Merge to main (stable release)
```

### For Contributors

1. **Start from `dev`:**
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit locally.

3. **Push and open a PR:**
   ```bash
   git push -u origin feature/your-feature-name
   ```
   Then open a pull request on GitHub targeting the **`dev`** branch.

4. **Wait for review** — address feedback if requested.

5. **Merge** — once approved, the PR will be merged into `dev`.

### For Maintainers

- Regular feature work flows through `dev`.
- When ready for a stable release, open a PR from `dev` → `main`.
- Ensure all tests pass and all active maintainers approve before merging to `main`.

---

## 📝 Branch Naming Conventions

Use descriptive prefixes:

| Prefix | Use For |
|--------|---------|
| `feature/` | New features or enhancements |
| `fix/` | Bug fixes |
| `chore/` | Maintenance, tooling, docs |
| `refactor/` | Code restructuring (no behavior change) |
| `test/` | Adding or updating tests |

**Examples:**
- `feature/5-critical-thinking-mode`
- `fix/16-export-chat-buttons-greyed-out`
- `chore/add-contributing-guide`

---

## 🧪 Testing

Before opening a PR:

- ✅ Test your changes locally with the Streamlit app
- ✅ Verify no console errors or warnings
- ✅ If adding a feature, test edge cases
- ✅ If fixing a bug, confirm the fix resolves the issue

---

## 📬 Questions?

- Open an issue for bugs, feature requests, or workflow questions.
- Tag existing maintainers for faster responses.

---

Thanks for contributing! 🚴‍♂️
