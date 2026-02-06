# Git Hooks for UFC Fight Predictor

## Setup Instructions

### 1. Enable Custom Git Hooks Directory

Run this command from the project root:

```bash
git config core.hooksPath .githooks
```

This tells git to use hooks from [.githooks/](.githooks/) instead of the default [.git/hooks/](.git/hooks/).

### 2. Verify Hook is Executable

```bash
chmod +x .githooks/post-commit
```

### 3. Test the Hook

Make a test commit:

```bash
echo "# Test" >> test_file.txt
git add test_file.txt
git commit -m "[TEST] Testing auto-changelog hook"
```

Check [CHANGELOG.md](../CHANGELOG.md) — you should see a new entry with today's date.

### 4. Cleanup Test

```bash
git reset --soft HEAD~1  # Undo commit
rm test_file.txt         # Remove test file
```

---

## Available Hooks

### post-commit

**Purpose:** Automatically update [CHANGELOG.md](../CHANGELOG.md) after every commit.

**What it does:**
1. **Captures commit info:** Hash, date, message, changed files
2. **Auto-categorizes:** Based on file patterns or explicit `[CATEGORY]` in commit message
3. **Appends to CHANGELOG.md:** Creates new date section or adds to existing
4. **Flags architecture changes:** Prints reminder if critical files modified
5. **Flags feature engineering changes:** Prints reminder to re-run pipeline

**Categories:**
- `[DATA]` — Data scraping, collection, quality (e.g., `src/etl/scraper.py`)
- `[FEATURE]` — Feature engineering (e.g., `src/features/engineer.py`)
- `[MODEL]` — Model training, evaluation (e.g., `src/models/train.py`)
- `[FIX]` — Bug fixes (commit message contains "fix", "bug", "error")
- `[INFRA]` — Config, dependencies, tooling (e.g., `config.yaml`, `requirements.txt`)
- `[DOCS]` — Documentation (e.g., `*.md`, `*.html`)

**Auto-categorization rules:**
```bash
src/etl/scraper.py, scrape_*, data/           → [DATA]
src/features/, run_feature_engineering.py     → [FEATURE]
src/models/train.py, src/models/predict.py    → [MODEL]
Commit message contains "fix|bug|error"       → [FIX]
*.md, *.txt, *.html                          → [DOCS]
config.*, requirements.*, run.py, .git*      → [INFRA]
```

**Manual categorization (recommended):**
```bash
# Prefix commit message with category
git commit -m "[DATA] Scrape upcoming UFC events"
git commit -m "[FEATURE] Add interaction features for striker vs wrestler"
git commit -m "[FIX] Fix null pointer in rolling window calculation"
```

**Output example:**
```
✅ Auto-changelog: Added entry to CHANGELOG.md
   Category: [DATA]
   Commit: a1b2c3d

⚠️  ARCHITECTURE-LEVEL CHANGE DETECTED!
   The following critical files were modified:
     - src/etl/scraper.py
     - requirements.txt

   📋 ACTION REQUIRED: Review and update ARCHITECTURE.md if needed
      - Data flow changes?
      - New dependencies?
      - Module responsibilities changed?
```

---

## Disabling Hooks (Temporarily)

### For a single commit:
```bash
git commit --no-verify -m "Message"
```

### Permanently disable:
```bash
git config core.hooksPath ""
```

---

## Troubleshooting

### Hook not running?

**Check hook is enabled:**
```bash
git config core.hooksPath
# Should output: .githooks
```

**Check hook is executable:**
```bash
ls -l .githooks/post-commit
# Should show: -rwxr-xr-x (x = executable)
```

**Re-enable if disabled:**
```bash
git config core.hooksPath .githooks
chmod +x .githooks/post-commit
```

### CHANGELOG.md not updating?

**Check file exists:**
```bash
ls -l CHANGELOG.md
# If missing, hook will print: "⚠️  CHANGELOG.md not found. Skipping auto-changelog."
```

**Check git status:**
```bash
git status
# CHANGELOG.md should show as "modified" after commit
```

**Manual test:**
```bash
# Run hook manually (simulates post-commit)
./.githooks/post-commit
```

### Permission denied error?

```bash
chmod +x .githooks/post-commit
```

---

## Customizing the Hook

Edit [.githooks/post-commit](.githooks/post-commit) to:
- Add new categories
- Change auto-categorization rules
- Modify changelog format
- Add additional checks (linting, tests, etc.)

**Example: Add [TEST] category:**
```bash
# In post-commit script, add to categorization logic:
elif echo "$CHANGED_FILES" | grep -qE '^tests/'; then
    CATEGORY="[TEST]"
```

**Example: Run tests before commit:**
```bash
# Add before changelog update:
if [ -d "tests" ]; then
    pytest tests/ || {
        echo "❌ Tests failed! Commit aborted."
        exit 1
    }
fi
```

---

## Why Git Hooks?

**Benefits:**
- ✅ **Automatic changelog:** No manual changelog updates (easy to forget)
- ✅ **Consistent format:** All entries follow same structure
- ✅ **Architecture reminders:** Flags when critical files change
- ✅ **Pipeline reminders:** Flags when feature engineering needs re-run
- ✅ **Zero overhead:** Runs automatically, no extra commands

**Alternatives considered:**
- Manual changelog updates → Error-prone, often forgotten
- CI/CD changelog generation → Only works on push, not local commits
- Commit message parser → Post-hoc, doesn't help during development

---

## Git Hooks Best Practices

1. **Keep hooks fast** — Slow hooks delay commits (current hook is <0.1s)
2. **Make hooks optional** — Use `git commit --no-verify` for emergencies
3. **Version control hooks** — Store in repo (`.githooks/`) not `.git/hooks/`
4. **Test hooks thoroughly** — Bad hook = broken git workflow
5. **Document hooks** — This README helps teammates enable hooks

---

## Additional Hooks (Future)

### pre-commit (Not Yet Implemented)
- Run linters (black, flake8)
- Check for secrets (.env leaks)
- Validate no data files staged (prevent large CSV commits)

### pre-push (Not Yet Implemented)
- Run full test suite
- Check branch protection (prevent force push to main)
- Validate changelog is up to date

---

**Maintained By:** JSKI + Claude Sonnet 4.5
**Last Updated:** 2026-02-05
