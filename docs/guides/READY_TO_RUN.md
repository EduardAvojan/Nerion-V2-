# Ready to Run: Quality Scraper Pilot

## ✅ System Status: ARMED & TESTED

All components are implemented, tested, and committed. The quality scraper is ready for a pilot run.

---

## What's Built

### 1. Evidence-Based Quality Assessment ✅
- **5 evidence types**: complexity, security, guard_or_boundary_or_raii, verification, structure
- **Two-tier gates**: GOLD (pristine) and SILVER (useful)
- **Score tracking**: gross_score, penalty_score, net_score

### 2. Negative Evidence (7 Penalties) ✅
| Penalty | Weight | What It Catches |
|---------|--------|----------------|
| `complexity_increase_no_verification` | -4 pts | Complexity ↑2 without tests |
| `sql_string_concat` | -6 pts | SQL with string concat/f-strings |
| `removed_validation` | -4 pts | Lost isinstance/hasattr checks |
| `swallowed_exception` | -5 pts | except: pass / except: return None |
| `linter_disabled` | -3 pts | # noqa, # type: ignore, etc. |
| `wildcard_import` | -3 pts | from x import * |
| `unsafe_path_concat` | -3 pts | String concat instead of os.path.join |

### 3. Hardened Verification ✅
- Test files with body changes (not renames)
- New asserts (not moved code)
- Filtered CI keywords ("failing test", "fix ci")
- Bug issue linkage (fixes #123 + bug keywords)

### 4. Quarantine Filter ✅
Blocks poison data:
- Vendor/generated/lockfiles (vendor/, node_modules/, *.min.js, package-lock.json)
- Mass whitespace diffs (>80% whitespace)
- Pure formatting (>95% line overlap)

### 5. Comprehensive Metrics ✅
`EnhancedStats` tracks:
- Tier distribution (GOLD/SILVER/REJECT %)
- Evidence distribution & co-occurrence
- Score histogram
- Penalty incidence
- Per-repo normalization

### 6. Unit Tests ✅
**26 tests passing**:
- 14 penalty detector tests
- 8 hardened verification tests
- 4 integration tests

---

## Tier Gates (Final)

```python
GOLD:
  net_score ≥ 8 AND
  2+ evidence types AND
  "verification" in evidence AND
  penalty_score == 0  # Zero tolerance

SILVER:
  net_score ≥ 2 AND
  penalty_score ≤ 2  # Some tolerance

REJECT: else
```

Where `net_score = gross_score - penalty_score`

---

## How to Run Pilot

### 1. Prepare Test Dataset
Choose 100-1000 commits from 3-5 diverse repos:
- Mix of bug fixes, refactors, security patches
- Different repo sizes/maturity levels
- Avoid mono-repos initially (they skew stats)

### 2. Run Scraper
```python
from pathlib import Path
from nerion_digital_physicist.data_mining.github_quality_scraper import GitHubQualityScraper

scraper = GitHubQualityScraper(
    db_path=Path("pilot_lessons.db"),
    github_token="your_token_here"
)

# When API connector is ready:
# scraper.scrape(target_count=100, max_attempts=1000)

# After scraping:
scraper.print_final_report()
```

### 3. Validate Metrics

**Check these targets**:
| Metric | Target | What It Means |
|--------|--------|---------------|
| GOLD rate | 1-5% | Pristine commits (zero penalties) |
| SILVER rate | 10-25% | Useful commits (≤2 penalties) |
| GOLD share | ≥25% | Of accepted commits |
| Verification in accepted | ≥40% | Most changes have tests/verification |
| Verification in GOLD | 100% | By design |

**Top evidence combos** (should see):
- `guard_or_boundary_or_raii` ∧ `verification`
- `security` ∧ `verification`
- `complexity` ∧ `verification`

**Common penalties** (expected):
- `complexity_increase_no_verification`: Changes without tests
- `swallowed_exception`: Error hiding

### 4. Spot-Check Quality
**Manual review** (critical validation):
- **10 GOLD commits**: Should be pristine, zero penalties, 2+ evidence types
- **10 SILVER commits**: May have 1-2 penalties, at least one positive evidence
- **10 REJECT commits**: Low score or high penalties

**Questions to ask**:
- Do GOLD commits look boringly correct? (guards+tests, security+tests, complexity↓+tests)
- Is SILVER useful but not trash?
- Did any penalty slip into GOLD? (If yes, bug in gate logic)

### 5. Adjust If Needed

**If GOLD rate is too low (<0.5%)**:
- Check if repos are test-light (add repo-local percentile gates)
- Verify penalties aren't too aggressive (check penalty distribution)

**If GOLD rate is too high (>10%)**:
- Tighten GOLD threshold (try net_score ≥ 10)
- Add more penalties (if trash is slipping through)

**If SILVER is noisy**:
- Raise SILVER threshold (try net_score ≥ 4)
- Lower penalty tolerance (try penalties ≤ 1)

**If verification rate is low (<30%)**:
- Check if verification hardening is too strict
- May need to relax test proximity detection

---

## What to Watch

### Red Flags 🚩
- GOLD with penalties > 0 (bug in gate logic)
- GOLD without verification (bug in gate logic)
- GOLD rate >10% (too lenient, trash leaking)
- Verification in accepted <20% (hardening too strict)
- One penalty type dominates >60% (detector too sensitive)

### Green Signals ✅
- GOLD share ≥25% of accepted
- Top evidence combos involve verification
- Per-repo stats within 2-3× of each other
- Spot-checks confirm GOLD is pristine
- No duplicate patterns in SILVER (dedup working)

---

## Files Reference

### Core Components
- `github_quality_scraper.py`: Main scraper with all filters
- `quality_hardening.py`: EnhancedStats, NegativeEvidenceDetector, VerificationHardening
- `quality_hardening_test.py`: 26 unit tests

### Documentation
- `QUALITY_HARDENING_SUMMARY.md`: What was locked down
- `HARDENING_INTEGRATION.md`: Integration guide
- `READY_TO_RUN.md`: This file

### Database
- `pilot_lessons.db`: SQLite database (will be created on first run)
- Schema: lessons table with before_code, after_code, test_code, category, metadata

---

## Next Steps After Pilot

### If Metrics Look Good:
1. **Full scrape**: 10k-100k commits across 20-50 repos
2. **Feed GOLD to GNN**: Start training on pristine examples
3. **Queue SILVER**: Review and graduate weekly
4. **Monitor drift**: Track GOLD rate by repo over time

### If Adjustments Needed:
1. **Tune thresholds**: Based on spot-check results
2. **Add/remove penalties**: Based on false positives
3. **Adjust verification**: If too strict/loose
4. **Add repo-local percentile gates**: If per-repo variance is high

### Future Enhancements:
1. **Deduplication**: MinHash for SILVER tier
2. **Enhanced payloads**: Typed edit ops, CPG deltas
3. **Drift monitoring**: Auto-alert on metric shifts
4. **Calibration loop**: Logistic model for percentile gates

---

## Quick Start Commands

```bash
# Run unit tests
cd /Users/ed/Nerion-V2/nerion_digital_physicist/data_mining
pytest quality_hardening_test.py -v

# Verify syntax
python3 -c "import ast; ast.parse(open('github_quality_scraper.py').read())"

# Start pilot (when API connector ready)
# python3 run_pilot_scrape.py --repos "user/repo1,user/repo2" --count 100

# View results
# python3 -c "from github_quality_scraper import *; scraper.print_final_report()"
```

---

## Status Summary

✅ Evidence-based scoring (5 types)
✅ Negative evidence (7 penalties)
✅ Hardened verification (noise-filtered)
✅ Quarantine filter (vendor/generated/whitespace)
✅ Comprehensive metrics (EnhancedStats)
✅ Unit tests (26 passing)
✅ Tier gates (GOLD/SILVER/REJECT)
⏳ API connector (not yet implemented)
⏳ Pilot scrape (waiting for API)

**System is armed, tested, and ready to prove itself under stress.**

The only missing piece is the GitHub API connector to actually fetch commits. Once that's implemented, run a pilot on 100-1000 commits and validate the metrics.

**Expected outcome**: GOLD 1-5%, SILVER 10-25%, GOLD share ≥25%, no trash leaking through.
