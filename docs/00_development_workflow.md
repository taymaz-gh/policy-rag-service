# Development Workflow

## Git Branching Strategy

All new functionality is developed on feature branches created from `main`.

```text
main
 |
 A ---- B ---------------- S ---- main
         \                /
          C ---- D ---- E       feature branch

S = squash-merged feature commit
```

## Workflow

```text
Create feature branch
        |
        v
Implement functionality
        |
        v
Run pytest + Ruff
        |
        v
Update documentation
        |
        v
Push branch
        |
        v
Open GitHub pull request
        |
        v
Squash and merge into main
        |
        v
Sync main + delete feature branch
```

## Useful Commands

Create a feature branch:

```powershell
git switch main
git pull --ff-only origin main
git switch -c feature/example
```

Verify changes:

```powershell
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

Push the feature:

```powershell
git push -u origin feature/example
```

After squash merging on GitHub:

```powershell
git switch main
git pull --ff-only origin main
git branch -D feature/example
git fetch --prune origin
```

## Commit Conventions

Use concise, descriptive commit messages:

- `feat:` New functionality
- `fix:` Bug fixes
- `test:` New or improved tests
- `docs:` Documentation
- `refactor:` Internal code restructuring
- `chore:` Maintenance and configuration

## Pull Request Requirements

Before merging:

- Tests pass.
- Ruff reports no errors.
- Documentation reflects implemented functionality.
- No API keys or sensitive data are committed.
- The PR summary explains what changed and how it was verified.