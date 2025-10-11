# GitHub Quality Scraper

Production-grade scraper for mining high-quality Python bug fixes from GitHub to augment GNN training data.

## Features

✅ **Multi-Stage Quality Filtering**
- Commit message patterns (reject merges, formatting, typos)
- File type validation (Python source only)
- Size constraints (3-300 lines changed)
- Syntactic validation (both versions must parse)
- Semantic quality scoring (0-100 score, threshold 60+)

✅ **Intelligent Quality Assessment**
- Complexity metrics (cyclomatic complexity delta)
- Security improvements (removes eval/exec, adds validation)
- Code quality (error handling, type hints, docstrings)
- Structural improvements (modularization, refactoring)

✅ **Automatic Test Synthesis**
- Security tests for vulnerability fixes
- Regression tests for bug fixes
- Equivalence tests for refactorings
- Smoke tests as fallback

✅ **Production Ready**
- GitHub API rate limiting (60/hour unauth, 5000/hour auth)
- Automatic retry and backoff
- Progress reporting with statistics
- SQLite storage matching your curriculum schema

## Installation

```bash
# No extra dependencies needed - uses existing Nerion environment
cd /Users/ed/Nerion-V2
```

## Quick Start

### Test Mode (Recommended First)

```bash
# Test with 100 commits, no GitHub token required
python -m nerion_digital_physicist.data_mining.run_scraper \
  --target 100 \
  --db test_github_lessons.db \
  --test

# This will:
# - Fetch ~500 commits
# - Accept ~50-100 quality examples
# - Save to test_github_lessons.db
# - Take ~10-15 minutes
```

### Production Run (Authenticated)

```bash
# Get GitHub token at: https://github.com/settings/tokens
# No special permissions needed, just public repo read access

export GITHUB_TOKEN=ghp_your_token_here

# Scrape 10,000 quality commits
python -m nerion_digital_physicist.data_mining.run_scraper \
  --target 10000 \
  --db github_lessons.db

# Expected:
# - Process ~100,000 commits
# - Accept ~10,000 quality examples
# - Duration: 6-10 hours
# - Rate: ~300-500 commits/hour
```

## Command Line Options

```bash
python -m nerion_digital_physicist.data_mining.run_scraper --help

Options:
  --target INT          Target number of quality commits (default: 1000)
  --db PATH             Database path (default: github_lessons.db)
  --test                Run in test mode (limited results)
  --min-quality INT     Min quality score 0-100 (default: 60)
  --github-token STR    GitHub token (or use GITHUB_TOKEN env var)
```

## Quality Filtering Pipeline

### Stage 1: Message Filter
Rejects:
- Merge commits
- Version bumps
- Documentation only
- Formatting/typo fixes
- WIP commits

Accepts:
- Bug fixes
- Security fixes
- Refactorings
- Optimizations

### Stage 2: File Type Filter
- Python source files only (`.py`)
- Excludes test files, configs, documentation
- Max 5 files changed per commit

### Stage 3: Size Filter
- 3-300 lines changed
- Must have both additions AND deletions
- Balanced changes (not just adds or deletes)
- Min 30% code ratio (exclude whitespace)

### Stage 4: Syntax Validation
- Both before/after must parse as valid Python
- Must have meaningful AST structure (≥5 nodes)

### Stage 5: Quality Assessment (0-100 score)

**Complexity (40 points max):**
- Significant complexity reduction: +20
- Some complexity reduction: +10

**Security (30 points max):**
- Removes eval(): +15
- Removes exec(): +15
- Adds input validation: +10

**Code Quality (30 points max):**
- Adds error handling: +15
- Adds type hints: +10
- Adds docstrings: +5

**Threshold:** Score ≥ 60 to accept

### Stage 6: Category Inference

Maps to CEFR levels based on:
- Security fixes → C1/C2
- Complex refactorings → C1
- Standard improvements → B2/B1
- Simple fixes → A2

### Stage 7: Test Synthesis

Generates appropriate test code:
- **Security fixes:** Tests for removed vulnerabilities
- **Bug fixes:** Regression tests
- **Refactorings:** Equivalence tests
- **Default:** Basic smoke tests

## Expected Results

### Quality Funnel

| Stage | Input | Output | Rate |
|-------|-------|--------|------|
| Fetch | 100,000 | 100,000 | 100% |
| Message filter | 100,000 | 40,000 | 40% |
| File type filter | 40,000 | 30,000 | 75% |
| Size filter | 30,000 | 20,000 | 67% |
| Syntax filter | 20,000 | 15,000 | 75% |
| Quality filter | 15,000 | **10,000** | **67%** |

**Overall acceptance:** ~10% of fetched commits

### Quality Distribution

Expected category breakdown for 10,000 accepted commits:
- **C2 (Expert):** ~500 (5%) - Security, complex algorithms
- **C1 (Advanced):** ~2,000 (20%) - Significant refactorings
- **B2 (Upper-intermediate):** ~3,500 (35%) - Standard improvements
- **B1 (Intermediate):** ~3,000 (30%) - Bug fixes
- **A2 (Elementary):** ~1,000 (10%) - Simple fixes

## Database Schema

Saves to SQLite with same schema as your curriculum lessons:

```sql
CREATE TABLE lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,           -- "github_{sha}_{category}"
    description TEXT,                     -- Commit message
    before_code TEXT NOT NULL,
    after_code TEXT NOT NULL,
    test_code TEXT,
    category TEXT,                        -- CEFR level (a1-c2)
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT                         -- JSON with GitHub details
);
```

**Metadata JSON includes:**
- `source`: "github"
- `repo`: Full repository name
- `commit_sha`: Full commit SHA
- `commit_message`: Original commit message
- `commit_url`: GitHub commit URL
- `author`: Commit author
- `timestamp`: Commit timestamp
- `quality_score`: Computed quality score (0-100)
- `metrics`: Detailed quality metrics

## Merging with Existing Lessons

```bash
# After scraping, merge with your existing curriculum
python -c "
import sqlite3

# Copy GitHub lessons to main database
src = sqlite3.connect('github_lessons.db')
dst = sqlite3.connect('nerion_knowledge.db')

# Copy lessons
src_cursor = src.cursor()
dst_cursor = dst.cursor()

src_cursor.execute('SELECT * FROM lessons')
for row in src_cursor.fetchall():
    try:
        dst_cursor.execute('''
            INSERT INTO lessons VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', row)
    except sqlite3.IntegrityError:
        pass  # Skip duplicates

dst.commit()
src.close()
dst.close()

print('✅ Lessons merged')
"

# Verify counts
sqlite3 nerion_knowledge.db "SELECT COUNT(*) as total FROM lessons"
```

## Rate Limiting

### Unauthenticated (No Token)
- **Rate:** 60 requests/hour
- **Speed:** ~50-60 commits/hour
- **Recommendation:** Good for testing only

### Authenticated (With Token)
- **Rate:** 5,000 requests/hour
- **Speed:** ~300-500 commits/hour
- **Recommendation:** Required for production

### Get a GitHub Token
1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. No special permissions needed (leave all unchecked)
4. Copy token and export: `export GITHUB_TOKEN=ghp_xxx`

## Monitoring Progress

The scraper shows real-time statistics:

```
╔══════════════════════════════════════╗
║     GitHub Scraping Progress        ║
╠══════════════════════════════════════╣
║ Fetched:               50000     ║
║ ├─ Message filter:     30000 ❌  ║
║ ├─ File type filter:    8000 ❌  ║
║ ├─ Size filter:         4000 ❌  ║
║ ├─ Syntax filter:       2000 ❌  ║
║ ├─ Quality filter:      3000 ❌  ║
║ └─ ACCEPTED:            3000 ✅  ║
║                                      ║
║ Acceptance rate:         6.0%      ║
║ Errors:    12                   ║
╚══════════════════════════════════════╝
```

## Troubleshooting

### "Rate limit exceeded"
- Wait for rate limit to reset (shown in output)
- Or add GitHub token for 5000/hour limit

### "No commits found"
- Check internet connection
- Verify GitHub API is accessible
- Try with `--test` flag first

### "Too many syntax errors"
- Normal! GitHub has lots of invalid Python
- Scraper filters these out automatically
- Acceptance rate should be 5-15%

### "Quality score too low"
- Adjust with `--min-quality 50` (lower threshold)
- Trade-off: more volume vs higher quality
- Recommended: keep at 60+ for GNN training

## Next Steps

After scraping:

1. **Validate samples:** Check 10-20 random examples
   ```bash
   sqlite3 github_lessons.db "SELECT name, category, quality_score FROM lessons ORDER BY RANDOM() LIMIT 10"
   ```

2. **Merge with curriculum:** Add to main database
   ```bash
   # See "Merging with Existing Lessons" above
   ```

3. **Export for GNN training:**
   ```bash
   python -m nerion_digital_physicist.training.dataset_builder \
     --db nerion_knowledge.db \
     --output-dir experiments/datasets/gnn/augmented \
     --mode supervised
   ```

4. **Retrain GNN:**
   ```bash
   python -m nerion_digital_physicist.training.run_training \
     --dataset experiments/datasets/gnn/augmented/dataset.pt \
     --architecture gat \
     --epochs 50 \
     --pretrained digital_physicist_brain.pt
   ```

## Architecture

```
run_scraper.py (CLI)
    │
    ├── github_api_connector.py
    │   ├── Search GitHub commits
    │   ├── Fetch commit details
    │   ├── Extract diffs
    │   └── Handle rate limiting
    │
    └── github_quality_scraper.py
        ├── Stage 1: Message filter
        ├── Stage 2: File type filter
        ├── Stage 3: Size filter
        ├── Stage 4: Syntax validation
        ├── Stage 5: Quality assessment
        ├── Stage 6: Category inference
        ├── Stage 7: Test synthesis
        └── Stage 8: Save to SQLite
```

## Contributing

To add new quality filters or metrics:

1. **Add filter:** Edit `github_quality_scraper.py`
2. **Add metric:** Edit `assess_quality()` method
3. **Test:** Run with `--test` flag
4. **Validate:** Check acceptance rate (should be 5-15%)

## License

Part of the Nerion Digital Physicist project.
