# Lesson Quality Standards - MANDATORY FOR ALL AGENTS

**Date:** October 28, 2025
**Status:** Production-ready standards after achieving Grade A (9/10)
**Purpose:** Prevent duplicates and quality issues when scaling to 20,000+ lessons

---

## 🎯 MISSION CRITICAL RULES

**ALL lesson generator agents MUST follow these standards WITHOUT EXCEPTION.**

These standards were established after comprehensive quality auditing that eliminated 37 bad lessons and achieved 94.9% clean lesson rate.

---

## 1. DUPLICATE PREVENTION (MANDATORY - STEP 0)

### ⚠️ YOU MUST CHECK FOR DUPLICATES BEFORE GENERATING ANY LESSON

**Run this code FIRST, before generating any lessons:**

```python
import sqlite3
from pathlib import Path

# Load existing lesson names from workspace database
conn = sqlite3.connect("agent_generated_curriculum.sqlite")
cursor = conn.cursor()
existing_names = {row[0] for row in cursor.execute("SELECT name FROM lessons").fetchall()}
existing_count = len(existing_names)
conn.close()

print(f"✅ LOADED {existing_count} existing lessons from database")
print(f"Sample names: {list(existing_names)[:10]}")

# Create duplicate checker function
def is_duplicate(lesson_name):
    if lesson_name in existing_names:
        print(f"⚠️  SKIPPING {lesson_name} - already exists!")
        return True
    return False

# Add new lesson to tracking set after saving
def mark_lesson_created(lesson_name):
    existing_names.add(lesson_name)
```

**Before generating EACH lesson:**
```python
lesson_name = "a1_python_type_error_001"
if is_duplicate(lesson_name):
    # SKIP THIS LESSON - pick different name or topic
    continue

# Only generate if NOT duplicate
# ... generate lesson ...

# After successfully saving:
mark_lesson_created(lesson_name)
```

**⚠️ CRITICAL:** SafeCurriculumDB prevents duplicates at database level, but checking FIRST saves API tokens and avoids wasted work!

---

## 2. TEST CODE QUALITY REQUIREMENTS (MANDATORY)

### ❌ NEVER CREATE PLACEHOLDER TESTS

**FORBIDDEN test patterns:**
- `assert true;` (Java/SQL)
- `assert(true);` (JavaScript)
- `assert True` (Python)
- `pass` (Python - alone without any logic)
- Just comments with no executable code

### ✅ REQUIRED test structure:

**Minimum Requirements:**
1. **Length:** ≥ 100 characters (meaningful tests can't be shorter)
2. **Executable:** Must actually run the code
3. **Validates fix:** Must fail on before_code, pass on after_code
4. **Has assertions:** Must check actual behavior, not just "true"

**Python Example (GOOD):**
```python
test_code = """
import subprocess
import sys

# Test before_code (should fail)
before = '''
def add(a, b):
    return a + b
result = add("5", 3)  # Type error
'''
result = subprocess.run([sys.executable, '-c', before], capture_output=True)
assert result.returncode != 0, "before_code should fail with TypeError"

# Test after_code (should pass)
after = '''
def add(a, b):
    return a + b
result = add(int("5"), 3)  # Fixed
assert result == 8
'''
result = subprocess.run([sys.executable, '-c', after], capture_output=True)
assert result.returncode == 0, "after_code should pass"
"""
```

**JavaScript Example (GOOD):**
```javascript
test_code = """
const assert = require('assert');

// Test before_code (should throw)
const before = () => {
    const arr = [1, 2, 3];
    return arr[5];  // Undefined access
};
assert.strictEqual(before(), undefined, 'before returns undefined');

// Test after_code (should work)
const after = () => {
    const arr = [1, 2, 3];
    return arr[2] || 'default';  // Safe access
};
assert.strictEqual(after(), 3, 'after returns correct value');

console.log('✓ All tests passed');
"""
```

### Test Code Checklist:

Before saving ANY lesson, verify:
- [ ] Test code > 100 characters
- [ ] Has actual executable code (not just comments)
- [ ] Tests before_code behavior (shows the bug)
- [ ] Tests after_code behavior (shows the fix)
- [ ] Has meaningful assertions (not just `assert true`)
- [ ] Includes error handling where appropriate

---

## 3. CERF COMPLEXITY STANDARDS (MANDATORY)

### Complexity Thresholds by Level:

**A1 (Beginner):** Complexity 1-5
- Simple inline scripts (< 10 lines)
- Single concept per lesson
- Basic syntax, types, loops
- NO nested structures
- NO advanced concepts

**A2 (Elementary):** Complexity 1-8
- Simple functions (1-2 functions)
- Basic error handling
- Simple data structures
- Minimal nesting (1 level max)

**B1 (Intermediate):** Complexity 4-10
- Multiple functions
- Moderate nesting (2-3 levels)
- Basic patterns (iterators, callbacks)
- Simple class structures

**B2 (Upper Intermediate):** Complexity 6-12
- Complex functions
- Multiple classes/modules
- Advanced patterns
- Concurrency basics

**C1 (Advanced):** Complexity 8+
- Advanced algorithms
- Complex patterns
- System-level concepts
- Performance optimization

**C2 (Mastery/PhD):** Complexity 10+
- Compiler internals
- Runtime optimization
- Distributed systems
- Language design

### Complexity Calculation:

```python
def estimate_complexity(code):
    """Estimate code complexity (McCabe-style)."""
    complexity = 1
    complexity += code.count('if ')
    complexity += code.count('elif ')
    complexity += code.count('else:')
    complexity += code.count('for ')
    complexity += code.count('while ')
    complexity += code.count('try:')
    complexity += code.count('except ')
    complexity += code.count('def ')
    complexity += code.count('class ')
    return complexity
```

**Verify before saving:**
- A1: Reject if complexity > 5
- A2: Reject if complexity > 8
- C1/C2: Reject if complexity < 3

---

## 4. CODE STRUCTURE REQUIREMENTS

### Required Patterns by Level:

**A1/A2 Exceptions Allowed:**
- Inline scripts for beginner lessons (teaching script → function refactoring)
- Config/module code (< 200 chars or has imports)
- Very short snippets (< 10 lines)

**B1+ Requirements:**
- Must have function or class definitions
- Cannot be just inline script (unless teaching refactoring pattern)

### Language Field (MANDATORY):

**ALWAYS specify language when saving:**
```python
db.add_lesson(
    name="...",
    description="...",
    focus_area="...",
    language="python",  # ← CRITICAL: ALWAYS set this!
    before_code="...",
    after_code="...",
    test_code="..."
)
```

**Supported languages:** python, javascript, typescript, sql, java, go, rust, cpp, csharp, ruby, php

---

## 5. CONTENT QUALITY REQUIREMENTS

### before_code and after_code:

**Must have:**
- Clear bug demonstration (before_code)
- Clear fix demonstration (after_code)
- Meaningful difference (>10% code change) **or** documented as a verified minimal diff (see below)
- Realistic bugs (things developers actually encounter)

**Avoid:**
- Trivial changes (< 2% different) unless intentional (e.g., security one-liners)
- Artificial bugs that nobody would write
- Over-complicated examples
- Missing context

### Description Field:

**Required format:**
```
Fix [bug type] - [specific issue] in [context]
```

**Examples:**
- "Fix SQL injection vulnerability in user login query"
- "Fix race condition in concurrent HashMap access"
- "Fix memory leak in event listener cleanup"

**Include:**
- What the bug is
- Why it's a problem
- How the fix works

> **Update (2025-10-30):** Real-world fixes are often tight one-liners. Minimal diffs are allowed when you:
> - Capture fail→pass evidence (tests must fail on `before_code`, pass on `after_code`)
> - Store the reason in `metadata["minimal_diff_reason"]`
> - Ensure the change touches more than pure formatting (at least a few tokens/characters)

---

## 6. SELF-VETTING PROCESS (MANDATORY)

**Before saving ANY lesson, you MUST:**

### Step 1: Generate Test
```python
# Write test_code that validates the lesson
test_code = """..."""
```

### Step 2: Test Before Code
```bash
# Save before_code to temp file and test it
# Verify it FAILS or shows the bug
```

### Step 3: Test After Code
```bash
# Save after_code to temp file and test it
# Verify it PASSES or fixes the bug
```

### Step 4: Run Quality Checks
```python
# Check complexity
complexity = estimate_complexity(before_code)
assert complexity <= cerf_threshold, f"Complexity {complexity} too high for {focus_area}"

# Check test length
assert len(test_code) >= 100, f"Test code too short: {len(test_code)} chars"

# Check no placeholders
forbidden = ['assert true', 'assert(true)', 'assert True', 'pass  # TODO']
assert not any(f in test_code for f in forbidden), "Test has placeholder code"

# Check meaningful difference
diff_ratio = calculate_diff(before_code, after_code)
assert diff_ratio > 0.10, f"Diff too small: {diff_ratio*100:.1f}%"
```

### Step 5: Save Only If All Pass
```python
if all_checks_passed:
    with SafeCurriculumDB(Path("agent_generated_curriculum.sqlite")) as db:
        db.add_lesson(...)
else:
    print("❌ SKIPPING lesson - failed quality checks")
```

---

## 7. MULTI-LANGUAGE DISTRIBUTION (MANDATORY)

### Production-Realistic Distribution:

**TIER 1 (20% each):**
- Python: 20% (AI/ML, backend, scripting)
- Java: 20% (Enterprise, Android)
- SQL: 20% (Database - bugs affect everything)

**TIER 2:**
- JavaScript/TypeScript: 15% (Web, Node.js)
- C++: 8% (Systems, performance)
- C#: 5% (Enterprise .NET, Unity)
- Go: 4% (Cloud, microservices)
- PHP: 3% (Web, WordPress)
- Rust: 3% (Systems, safety)
- Ruby: 2% (Rails, API)

**For N lessons, create distribution plan FIRST:**
```python
def create_language_plan(total_lessons):
    plan = {
        'python': int(total_lessons * 0.20),
        'java': int(total_lessons * 0.20),
        'sql': int(total_lessons * 0.20),
        'javascript': int(total_lessons * 0.08),
        'typescript': int(total_lessons * 0.07),
        'cpp': int(total_lessons * 0.08),
        'csharp': int(total_lessons * 0.05),
        'go': int(total_lessons * 0.04),
        'php': int(total_lessons * 0.03),
        'rust': int(total_lessons * 0.03),
        'ruby': int(total_lessons * 0.02),
    }
    return plan
```

**Follow plan EXACTLY - no deviations!**

---

## 8. DATABASE SAFETY (MANDATORY)

### ALWAYS Use SafeCurriculumDB:

```python
from pathlib import Path
from nerion_digital_physicist.db.safe_curriculum import SafeCurriculumDB

# ✅ CORRECT - Uses protected wrapper
with SafeCurriculumDB(Path("agent_generated_curriculum.sqlite")) as db:
    db.add_lesson(...)

# ❌ FORBIDDEN - Direct access
import sqlite3
conn = sqlite3.connect("...")  # NEVER DO THIS!
```

### Database Protections:

SafeCurriculumDB automatically:
1. Prevents duplicate names (UNIQUE constraint)
2. Prevents duplicate content (SHA256 hash)
3. Creates backups before writes
4. Validates all fields
5. Prevents DROP/TRUNCATE/DELETE operations
6. Logs all changes
7. Verifies data integrity

---

## 9. CLEANUP REQUIREMENTS (MANDATORY)

**After generating lessons, DELETE temporary files:**

```bash
# Delete Python scripts
rm -f create_*_lessons.py
rm -f test_*.py
rm -f temp_*.py

# Delete markdown summaries
rm -f out/learning/*_summary.md
rm -f *_report.md

# Delete temporary data
rm -f temp_*.json
rm -f /tmp/lesson_*.txt
```

**Only leave:**
- `agent_generated_curriculum.sqlite` (with lessons)
- No temporary files
- No scripts

---

## 10. QUALITY METRICS TO ACHIEVE

### Target Grade: A (9/10)

**Your lessons must achieve:**
- ✅ 0% duplicates (100% unique)
- ✅ 0% placeholder tests
- ✅ 100% CERF-appropriate complexity
- ✅ 100% executable tests
- ✅ > 90% clean lessons (no warnings)

### Quality Checkpoints:

After generating 10 lessons, run self-check:
```bash
python3 comprehensive_quality_check.py
```

If any issues found:
1. Fix the lesson generation logic
2. Delete bad lessons from workspace
3. Regenerate correctly
4. Re-verify

**DO NOT PROCEED to 100+ lessons if 10-lesson batch has issues!**

---

## 11. REFERENCE EXAMPLES

### ✅ PERFECT A1 Lesson:

```python
name = "a1_python_string_concat_type_error_001"
description = "Fix TypeError in string concatenation - convert int to str before concatenating"
focus_area = "A1"
language = "python"

before_code = """
# Bug: Trying to concatenate string + int
age = 25
message = "You are " + age + " years old"
print(message)
"""

after_code = """
# Fix: Convert int to string first
age = 25
message = "You are " + str(age) + " years old"
print(message)
"""

test_code = """
import subprocess
import sys

# Test before_code (should fail with TypeError)
before = '''
age = 25
message = "You are " + age + " years old"
'''
result = subprocess.run([sys.executable, '-c', before], capture_output=True, text=True)
assert result.returncode != 0, "before_code should fail"
assert 'TypeError' in result.stderr, "Should have TypeError"

# Test after_code (should pass)
after = '''
age = 25
message = "You are " + str(age) + " years old"
assert message == "You are 25 years old"
'''
result = subprocess.run([sys.executable, '-c', after], capture_output=True, text=True)
assert result.returncode == 0, "after_code should pass"

print("✓ All tests passed")
"""
```

**Why this is perfect:**
- Clear, beginner-appropriate bug
- Complexity = 1 (appropriate for A1)
- Test code > 100 chars
- Test validates bug and fix
- Has proper assertions
- Language specified
- Clean, realistic code

### ❌ BAD Examples (DO NOT CREATE):

**Bad Example 1: Placeholder Test**
```python
test_code = "assert true;"  # ❌ NO! Meaningless test
```

**Bad Example 2: Too Complex for A1**
```python
before_code = """
class DatabaseConnection:
    def __init__(self):
        self.pool = ConnectionPool()
    # ... 50 more lines
"""
# ❌ NO! Complexity 15+ is not A1!
```

**Bad Example 3: Missing Language**
```python
db.add_lesson(
    name="lesson_001",
    focus_area="A1",
    # language=???  # ❌ NO! Always specify language
)
```

---

## 12. EMERGENCY PROCEDURES

### If You Generate Bad Lessons:

**DO NOT MERGE TO MAIN DATABASE!**

1. Stop generation immediately
2. Run quality check on workspace
3. Identify all bad lessons
4. Delete from workspace database:
   ```sql
   DELETE FROM lessons WHERE id IN (bad_ids);
   ```
5. Fix generation logic
6. Regenerate correctly
7. Re-verify before merging

### If Duplicates Found:

SafeCurriculumDB will reject them automatically, but:
1. Check your duplicate checking code
2. Verify you're loading existing_names correctly
3. Ensure you're checking BEFORE generation

### If Tests Are Bad:

1. Review test code requirements (Section 2)
2. Test your test code manually
3. Ensure complexity is appropriate
4. Re-generate with fixed logic

---

## 🎯 FINAL CHECKLIST

Before saving ANY lesson, verify:

- [ ] Checked for duplicate name
- [ ] Complexity appropriate for CERF level
- [ ] Test code > 100 characters
- [ ] Test code has real assertions
- [ ] Test validates bug and fix
- [ ] Language field specified
- [ ] before_code has clear bug
- [ ] after_code has clear fix
- [ ] Description explains bug/fix
- [ ] Self-tested with Bash tool
- [ ] No placeholder code
- [ ] No temporary files left behind

---

**These standards are MANDATORY. No exceptions. No shortcuts.**

**Quality > Quantity. Always.**

**See:** `out/learning/QUALITY_REPORT.md` for current database status.
