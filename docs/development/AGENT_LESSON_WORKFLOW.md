# Agent Lesson Generation Workflow

**Quality-First Approach:** All agent-generated lessons are reviewed before merging into production.

---

## 📊 Database Architecture (Updated)

### Production Database (Protected)
- **Path:** `out/learning/curriculum.sqlite`
- **Contents:** 973 verified, high-quality lessons (IDs may have gaps)
- **Access:** SafeCurriculumDB with automatic backups
- **Used by:** Nerion's Digital Physicist for training
- **Status:** ✅ **DO NOT WRITE DIRECTLY - Protected from agent writes**

### Agent-Generated Database (Workspace)
- **Path:** `out/learning/agent_generated_curriculum.sqlite`
- **Contents:** 973 production lessons (IDs 1-973, sequential) + NEW agent-generated lessons (IDs 974+)
- **Access:** SafeCurriculumDB with automatic backups
- **Used by:** All 6 CERF lesson generator agents
- **Status:** 🟡 **Agents write here, review NEW lessons only (id > 973)**

### Key Design Decision:

**Agent database contains BOTH production + new lessons:**
- ✅ **IDs 1-973:** Production lessons (copied from production, renumbered sequentially)
- ✅ **IDs 974+:** NEW agent-generated lessons (pending review)
- ✅ **Built-in duplicate prevention:** Agents can query to avoid generating existing lessons
- ✅ **SafeCurriculumDB protection:** SHA256 hash prevents duplicate content
- ✅ **Review script:** Only reviews NEW lessons (id > 973)
- ✅ **Merge script:** Only merges NEW approved lessons (id > 973) back to production

---

## 🔄 Complete Workflow

### **Step 1: Generate Lessons with Agents**

Agents now have **built-in duplicate prevention** - they query the database first to avoid generating lessons that already exist.

```bash
# Example: Generate 10 A1 (beginner) lessons
# Use Task tool with subagent_type="cerf-a1-programming-lesson-generator"
# Prompt: "Generate 10 high-quality A1 lessons covering variable scope, type errors, and basic loops.
#         Check the database first to avoid duplicates."

# Example: Generate 20 B2 (upper-intermediate) lessons
# Use Task tool with subagent_type="cerf-b2-programming-lesson-generator"
# Prompt: "Generate 20 B2 lessons covering async/await, metaclasses, and threading patterns.
#         Query existing lessons first to find gaps in coverage."
```

**Important:**
- ✅ Agents automatically write to `agent_generated_curriculum.sqlite`, **not** the production database
- ✅ Agents check database first to avoid duplicating existing lessons (IDs 1-973)
- ✅ New lessons get IDs starting from 974
- ✅ SafeCurriculumDB automatically prevents duplicate content via SHA256 hash

---

### **Step 2: Review Lesson Quality**

Run the **BULLETPROOF** quality review script to validate only NEW lessons (id > 973):

```bash
python scripts/review_and_merge_lessons.py --review
```

**What this does (BULLETPROOF VERSION):**

**Technical Validation:**
1. ✅ Validates syntax (all code must compile)
2. ✅ Checks test framework (must use unittest)
3. ✅ Verifies test count (minimum 2 tests)
4. ✅ Tests before_code (must FAIL - demonstrates bug)
5. ✅ Tests after_code (must PASS - proves fix)

**Subjective Quality Validation (10/10 Standard):**
6. ✅ Code similarity check (before/after should be similar, not completely different)
7. ✅ Single bug check (warns if multiple bug markers found)
8. ✅ Realistic code check (warns if code is too short/trivial)
9. ✅ Test quality check (verifies assertions exist, not just code execution)
10. ✅ Code complexity check (verifies imports, functions, or classes exist)
11. ✅ CERF-level appropriateness (checks for expected patterns by level)

**Output:**
- ✅ Only reviews NEW lessons (id > 973, skips production lessons 1-973)
- ✅ Generates quality report (`out/learning/lesson_review_log.json`)

**Output example:**
```
=== REVIEWING 25 LESSONS ===

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lesson: a1_variable_scope_001
Description: Fix variable scope error - using variable before definition
Focus Area: a1_variable_scope
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ PASSED - Quality 10/10

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lesson: a1_type_errors_003
Description: Fix type error - string + int
Focus Area: a1_type_errors
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ FAILED - Issues found:
   ❌ Tests PASSED on before_code (should FAIL to demonstrate bug)

=== REVIEW SUMMARY ===
Total: 25
Approved: 23
Rejected: 2

Review log saved to: out/learning/lesson_review_log.json

To merge approved lessons into production:
   python scripts/review_and_merge_lessons.py --merge
```

---

### **Step 3: Merge Approved Lessons**

After reviewing the quality report, merge approved lessons into production:

```bash
python scripts/review_and_merge_lessons.py --merge
```

**What this does:**
1. ✅ Reads review log (`lesson_review_log.json`)
2. ✅ Identifies approved lessons (passed all quality checks)
3. ✅ Uses SafeCurriculumDB to merge into production (with automatic backup)
4. ✅ Handles duplicates gracefully (SHA256 hash check)
5. ✅ Reports success/failure for each lesson

**Output example:**
```
=== MERGING 23 APPROVED LESSONS ===

✅ Merged: a1_variable_scope_001
✅ Merged: a1_type_errors_001
✅ Merged: a1_loops_001
...
✅ Merged: b2_async_await_005

Successfully merged 23/23 lessons

To clear agent-generated database:
   rm out/learning/agent_generated_curriculum.sqlite
```

---

### **Step 4: Clean Up (Optional)**

After successful merge, you can clear the agent-generated database:

```bash
rm out/learning/agent_generated_curriculum.sqlite
```

**Note:** This deletes ALL lessons (approved and rejected). Make sure you've merged everything you want to keep!

---

## 📈 Check Statistics

View statistics for both databases:

```bash
python scripts/review_and_merge_lessons.py --stats
```

**Output example:**
```
=== DATABASE STATISTICS ===

📚 Production DB: 996 lessons
   A1: 127
   A2: 207
   B1: 420
   B2: 188
   C1: 62
   C2: 6

🤖 Agent-Generated DB: 0 lessons (pending review)
```

---

## 🔍 Quality Standards (10/10)

Every lesson must meet these requirements:

1. ✅ **100% test pass rate** - ALL tests pass for after_code
2. ✅ **Bug demonstrable** - At least ONE test fails for before_code
3. ✅ **Syntactic validity** - All code compiles without errors
4. ✅ **Test framework** - Uses unittest (not raw assertions)
5. ✅ **Sufficient coverage** - Minimum 2 tests
6. ✅ **Single clear bug** - One primary issue (CERF-level appropriate)
7. ✅ **Real-world relevance** - Production code patterns

**Reference:** Example 3 (Thread-Safe Cache) - c2_concurrency_advanced_001

---

## 🚨 Troubleshooting

### Agent writes to wrong database
**Problem:** Agent wrote to production database instead of agent-generated database.

**Solution:** All agents are configured to use `agent_generated_curriculum.sqlite`. Check agent prompt:
```python
AGENT_DB = Path("out/learning/agent_generated_curriculum.sqlite")
with SafeCurriculumDB(db_path=AGENT_DB) as db:
```

### Review script shows "No lessons to review"
**Problem:** Agent-generated database is empty.

**Solution:** Generate lessons first using CERF agents (Step 1).

### Merge fails with duplicate error
**Problem:** Lesson already exists in production database.

**Solution:** SafeCurriculumDB prevents duplicates by:
- Name (UNIQUE constraint)
- Content (SHA256 hash)

This is **expected behavior** - duplicate lessons are automatically skipped.

### All tests fail during review
**Problem:** Test execution environment issue.

**Solution:** Ensure Python environment has all required dependencies:
```bash
pip install -r requirements.txt
```

---

## 📝 Files Generated

| File | Purpose | When Created |
|------|---------|--------------|
| `out/learning/agent_generated_curriculum.sqlite` | Agent-generated lessons | First agent run |
| `out/learning/lesson_review_log.json` | Quality review results | `--review` |
| `backups/curriculum/hourly/*` | Hourly backups | Every merge |
| `backups/curriculum/daily/*` | Daily backups | Every merge |

---

## 🎯 Best Practices

1. **Review before merging** - Always run `--review` before `--merge`
2. **Read the review log** - Check `lesson_review_log.json` for detailed issues
3. **Fix rejected lessons** - Manually edit or regenerate failed lessons
4. **Clean up regularly** - Delete agent DB after merging to avoid confusion
5. **Check statistics** - Use `--stats` to track lesson distribution
6. **Batch generation** - Generate 10-50 lessons at a time for manageable review

---

## 🔐 Safety Features

### SafeCurriculumDB Protection
- ✅ Automatic backup before EVERY write
- ✅ Duplicate prevention (name + content hash)
- ✅ WAL mode for concurrency
- ✅ Integrity checks before writes
- ✅ Automatic restore on error

### Database Separation
- ✅ Production DB isolated from agent writes
- ✅ Agent DB for quality review only
- ✅ Explicit merge step prevents accidents
- ✅ Clear naming (`agent_generated_*`)

---

## 📚 See Also

- [DATABASE_PROTECTION.md](./DATABASE_PROTECTION.md) - 7-layer protection system
- [nerion_lesson_specification.md](../nerion_lesson_specification.md) - Lesson quality standards
- [.claude/agents/](../.claude/agents/) - CERF lesson generator agents (A1-C2)

---

**Summary:** Generate → Review → Merge. Always review before merging into production!
