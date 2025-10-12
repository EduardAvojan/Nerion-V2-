# Quality Hardening Integration Guide

This guide shows how to integrate the hardening enhancements into the main scraper.

## Phase 1: Enhanced Stats (DONE)

The `quality_hardening.py` module provides:
- `EnhancedStats`: Comprehensive metrics tracking
- `NegativeEvidenceDetector`: Penalty detection for risky patterns
- `VerificationHardening`: Noise-filtered verification signals

## Phase 2: Integration Steps

### Step 1: Import the hardening module

Add to imports in `github_quality_scraper.py`:

```python
from .quality_hardening import EnhancedStats, NegativeEvidenceDetector, VerificationHardening
```

### Step 2: Initialize hardening components in __init__

```python
def __init__(self, ...):
    # ... existing code ...
    self.stats = ScraperStats()
    self.enhanced_stats = EnhancedStats()  # NEW
    self.penalty_detector = NegativeEvidenceDetector()  # NEW
    self.verification_hardener = VerificationHardening()  # NEW
```

### Step 3: Update assess_quality() to use penalties

Replace the current assess_quality() method with the enhanced version below.

### Step 4: Update save_lesson() to record stats

```python
def save_lesson(self, commit: CommitData, test_code: str):
    # ... existing save code ...

    # NEW: Record to enhanced stats
    self.enhanced_stats.record_commit(commit)
```

### Step 5: Add comprehensive reporting

```python
def print_final_report(self):
    """Print both basic and enhanced stats."""
    self.stats.print_progress()
    print("\n")
    self.enhanced_stats.print_comprehensive_report()
```

## Enhanced assess_quality() Implementation

Replace the current method (lines 291-439) with this enhanced version:

```python
def assess_quality(self, commit: CommitData) -> bool:
    """Stage 5: Evidence-based semantic quality assessment with negative evidence.

    Two-tier system with penalty enforcement:
    - GOLD: score ≥8 AND 2+ evidence types AND verification AND zero penalties
    - SILVER: score ≥2 AND penalties ≤2
    - REJECT: else

    Returns True if quality score >= threshold.
    Updates commit.quality_score, commit.metrics, and commit.tier.
    """
    before_code = commit.before_code
    after_code = commit.after_code

    if not before_code or not after_code:
        return False

    metrics = {}
    score = 0
    evidence = set()  # Track independent evidence types
    penalties = []  # Track penalties

    try:
        # Parse ASTs once
        before_ast = ast.parse(before_code)
        after_ast = ast.parse(after_code)

        # ===== NEGATIVE EVIDENCE (PENALTIES) =====
        penalties = self.penalty_detector.detect_penalties(
            commit, before_ast, after_ast, before_code, after_code
        )
        penalty_score = self.penalty_detector.calculate_penalty_score(penalties)
        metrics['penalties'] = penalties
        metrics['penalty_score'] = penalty_score

        # ===== 1. COMPLEXITY (Evidence: "complexity") =====
        before_complexity = self._calculate_complexity(before_ast)
        after_complexity = self._calculate_complexity(after_ast)
        metrics["complexity_delta"] = after_complexity - before_complexity

        complexity_points = 0
        if metrics["complexity_delta"] < -2:
            complexity_points = 20
        elif metrics["complexity_delta"] < 0:
            complexity_points = 10

        score += complexity_points
        if complexity_points > 0:
            evidence.add("complexity")

        # ===== 2. SECURITY (Evidence: "security") =====
        security_points = 0

        # Removes eval/exec
        metrics["removes_eval"] = "eval(" in before_code and "eval(" not in after_code
        metrics["removes_exec"] = "exec(" in before_code and "exec(" not in after_code

        if metrics["removes_eval"]:
            security_points += 15
        if metrics["removes_exec"]:
            security_points += 15

        # Adds validation
        metrics["adds_validation"] = self._count_validations(after_ast) > self._count_validations(before_ast)
        if metrics["adds_validation"]:
            security_points += 10

        # Sanitization patterns
        metrics["adds_sanitization"] = self._detect_sanitization(before_code, after_code)
        if metrics["adds_sanitization"]:
            security_points += 5

        score += security_points
        if security_points > 0:
            evidence.add("security")

        # ===== 3. GUARDS / BOUNDARY / RAII / DEFAULTS (Evidence: "guard_or_boundary_or_raii") =====
        micro_fix_points = 0

        # Guard checks (null/None checks, length checks, type checks)
        guard_score = self._detect_guard_additions(before_ast, after_ast, before_code, after_code)
        metrics["adds_guards"] = guard_score > 0
        micro_fix_points += guard_score

        # RAII patterns (with statements, context managers)
        raii_score = self._detect_raii_patterns(before_ast, after_ast)
        metrics["adds_raii"] = raii_score > 0
        micro_fix_points += raii_score

        # Deterministic defaults (return None/[]/{}/ 0 instead of undefined)
        default_score = self._detect_deterministic_defaults(before_ast, after_ast)
        metrics["adds_defaults"] = default_score > 0
        micro_fix_points += default_score

        score += micro_fix_points
        if micro_fix_points > 0:
            evidence.add("guard_or_boundary_or_raii")

        # ===== 4. VERIFICATION (Evidence: "verification") - HARDENED =====
        has_verification, verification_points = self.verification_hardener.detect_hardened_verification(
            commit, before_ast, after_ast
        )
        metrics["verification_points"] = verification_points
        metrics["has_verification"] = has_verification

        score += verification_points
        if has_verification:
            evidence.add("verification")

        # ===== 5. STRUCTURE (Evidence: "structure") =====
        structure_points = 0

        # Type hints
        metrics["adds_type_hints"] = self._count_type_hints(after_ast) > self._count_type_hints(before_ast)
        if metrics["adds_type_hints"]:
            structure_points += 10

        # Docstrings
        metrics["adds_docstrings"] = self._count_docstrings(after_ast) > self._count_docstrings(before_ast)
        if metrics["adds_docstrings"]:
            structure_points += 5

        # Function extraction
        before_functions = self._count_functions(before_ast)
        after_functions = self._count_functions(after_ast)
        metrics["function_count_delta"] = after_functions - before_functions

        if metrics["function_count_delta"] > 0 and "refactor" in commit.message.lower():
            structure_points += 10

        score += structure_points
        if structure_points > 0:
            evidence.add("structure")

        # ===== APPLY PENALTIES =====
        net_score = score - penalty_score

        # ===== FINALIZE =====
        commit.quality_score = max(0, min(100, net_score))  # Clamp to [0, 100]
        commit.metrics = metrics
        metrics["evidence_types"] = list(evidence)
        metrics["evidence_count"] = len(evidence)
        metrics["gross_score"] = score
        metrics["net_score"] = net_score

        # ===== TWO-TIER GATE WITH PENALTY ENFORCEMENT =====
        GOLD_THRESH = 8
        SILVER_THRESH = 2
        MAX_PENALTY_FOR_SILVER = 2

        # Gold requirements:
        # - Net score >= 8 AND
        # - At least 2 different evidence types AND
        # - Has verification evidence AND
        # - Zero penalties (pristine)
        is_gold = (
            net_score >= GOLD_THRESH and
            len(evidence) >= 2 and
            "verification" in evidence and
            penalty_score == 0  # NEW: Zero tolerance for penalties in GOLD
        )

        # Silver requirements:
        # - Net score >= 2 AND
        # - Penalties <= 2 (some tolerance)
        is_silver = net_score >= SILVER_THRESH and penalty_score <= MAX_PENALTY_FOR_SILVER

        if is_gold:
            commit.tier = "GOLD"
            return True  # Always accept Gold
        elif is_silver:
            commit.tier = "SILVER"
            # Accept Silver only if threshold allows (for two-tier collection)
            return self.thresholds.min_quality_score <= SILVER_THRESH
        else:
            commit.tier = "REJECT"
            return False

    except Exception as e:
        print(f"  - Quality assessment error: {e}")
        return False
```

## Expected Results After Integration

### Target Metrics (first run validation):
- **GOLD**: 1-5% overall
- **SILVER**: 10-25% overall
- **GOLD share** among accepted: ≥25%
- **Verification** in GOLD: 100% (by design)
- **Verification** in accepted: ≥40%

### Common Penalty Triggers:
- `complexity_increase_no_verification`: Complexity went up without tests
- `sql_string_concat`: SQL queries using string concatenation
- `swallowed_exception`: Added except: pass or except: return None
- `wildcard_import`: Added from x import *
- `linter_disabled`: Added # noqa or similar

### Evidence Co-occurrence (healthy patterns):
- `guard_or_boundary_or_raii` ∧ `verification`
- `security` ∧ `verification`
- `complexity` ∧ `verification`

## Testing Checklist

1. ✅ Import quality_hardening module
2. ✅ Initialize EnhancedStats, NegativeEvidenceDetector, VerificationHardening
3. ✅ Replace assess_quality() with enhanced version
4. ✅ Update save_lesson() to call enhanced_stats.record_commit()
5. ✅ Add print_final_report() method
6. ✅ Run scraper on test dataset
7. ✅ Validate metrics against targets
8. ✅ Spot-check GOLD commits (should be pristine)
9. ✅ Spot-check SILVER commits (should have ≤2 penalty points)
10. ✅ Verify no trash slipped through

## Next Steps (Post-Integration)

1. **Create unit tests** for penalty detectors (quality_hardening_test.py)
2. **Add repo-local percentile calibration** (Phase 4)
3. **Implement deduplication** for SILVER tier (MinHash)
4. **Enhance lesson payloads** with graph delta information
5. **Shadow run** to compare old vs new acceptance rates

## Files Modified
- `github_quality_scraper.py`: Import hardening, update assess_quality(), update save_lesson()
- `quality_hardening.py`: NEW (created)
- `HARDENING_INTEGRATION.md`: NEW (this file)
