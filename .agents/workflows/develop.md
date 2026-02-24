---
description: Git workflow and development conventions for TradeSense HFP
---

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Production-ready releases |
| `develop` | Integration branch |
| `overhaul/tradesense-best` | Current active development branch |
| `feature/<name>` | Feature branches off develop |
| `fix/<name>` | Bug fix branches |

## Commit Conventions

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(scope): description     # New feature
fix(scope): description      # Bug fix
chore(scope): description    # Maintenance
refactor(scope): description # Code restructure
docs(scope): description     # Documentation only
```

Scopes: `ui`, `engine`, `data`, `trader`, `infra`, `config`

## Development Workflow

1. Ensure you're on the right branch:
```powershell
git checkout overhaul/tradesense-best
git pull origin overhaul/tradesense-best
```

2. Make changes, then stage and commit:
```powershell
git add -A
git commit -m "feat(scope): description"
```

3. Push changes:
```powershell
git push origin overhaul/tradesense-best
```

## Pre-Push Checklist
- [ ] App launches without errors (`streamlit run src/ui/app.py`)
- [ ] Import check passes: `python -c "from src.data import DataLake; from src.models import AIEngine, Gauntlet"`
- [ ] No secrets in code (check `.env` is gitignored)
