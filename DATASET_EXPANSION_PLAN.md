# Dataset Expansion Plan: 2,357 → 20,000+ Lessons

## Current State
- **Current:** 2,357 lessons (deduplicated)
- **Target:** 20,000+ lessons
- **Gap:** 17,643 lessons needed

## Available High-Quality Datasets

### 1. **Bugs.jar** (Java)
- **Size:** 1,158 bugs
- **Source:** https://github.com/bugs-dot-jar/bugs-dot-jar
- **Quality:** Real bugs from 8 large Java projects
- **Projects:** accumulo, camel, commons-math, flink, jackrabbit-oak, maven, etc.
- **Tests:** Each bug has test cases and patches
- **Status:** Ready to download

### 2. **BugSwarm** (Python + Java)
- **Size:** 3,091 fail-pass pairs
- **Source:** https://github.com/BugSwarm/bugswarm
- **Quality:** Reproducible bugs from GitHub Actions/Travis CI
- **Languages:** Java and Python
- **Tests:** Full CI build environment reproduced in Docker
- **Status:** Requires BugSwarm CLI setup

###3. **QuixBugs** (Python + Java)
- **Size:** 40 programs × 2 languages = 80 bugs
- **Source:** https://github.com/jkoppel/QuixBugs
- **Quality:** Classic algorithm bugs
- **Tests:** JUnit (Java) + pytest (Python)
- **Status:** Ready to download

### 4. **More BugsInPy Projects**
- **Current:** Used 10 projects (327 bugs)
- **Available:** 17 total projects in BugsInPy
- **Potential:** ~166 more bugs from unused projects
- **Projects:** luigi, cookiecutter, PySnooper, etc.

### 5. **More Defects4J Versions**
- **Current:** Used 854 bugs
- **Available:** Defects4J has been continuously updated
- **Potential:** Check for new bugs added since import

## Projected Totals

| Dataset | Bugs | Status |
|---------|------|--------|
| Current Database | 2,357 | ✅ Done |
| Bugs.jar | +1,158 | 🔄 Next |
| BugSwarm | +3,091 | 🔄 Planned |
| QuixBugs | +80 | 🔄 Planned |
| More BugsInPy | +166 | 🔄 Optional |
| **TOTAL** | **6,852** | |

## To Reach 20,000: Additional Sources Needed

**Gap after above:** 20,000 - 6,852 = 13,148 lessons

### Additional Dataset Options:

1. **ManyBugs/IntroClass** (C programs)
   - Classic APR benchmarks
   - ~185 bugs total

2. **Bears** (Java)
   - 251 reproducible build failures
   - From Apache, Spring, etc.

3. **GitHub Mining via PyDriller**
   - Mine commits with "fix bug" from popular repos
   - Extract before/after from git diffs
   - Validate with CI test results

4. **LeetCode Solutions with Common Mistakes**
   - Mine wrong submissions that were later fixed
   - Excellent for beginner-intermediate levels

5. **Stack Overflow Bugs**
   - Extract MCVE (Minimal Complete Verifiable Example) bug fixes
   - Filter for accepted answers with code changes

6. **Synthetic Generation via Agents**
   - Use your existing 6 CERF agents
   - Generate 5,000+ lessons across all languages
   - 100% unique, tested, quality-controlled

## Quality Rating System

### Auto-Rating Criteria (10-point scale):

**Code Quality (4 points):**
- before_code and after_code both >100 chars (1 pt)
- Clear single bug fix, not complete rewrite (1 pt)
- Imports, functions, classes present (1 pt)
- Realistic, production-like code (1 pt)

**Test Quality (4 points):**
- Real test framework (unittest, pytest, JUnit) (1 pt)
- Test length >200 chars (1 pt)
- Has assertions/expects (1 pt)
- Tests fail on before, pass on after (1 pt)

**Metadata Quality (2 points):**
- Clear description (1 pt)
- Proper CERF classification (1 pt)

### Quality Flags:
- `stub_test` - Test doesn't actually validate
- `incomplete_code` - Code fragments, not complete
- `trivial_fix` - Too simple (single char change)
- `multiple_bugs` - More than one bug mixed
- `no_test_validation` - Can't verify test actually works

### Quality Thresholds:
- **10/10:** Perfect, production-ready
- **9/10:** Excellent, minor issues
- **<9/10:** FLAG FOR MANUAL REVIEW
- **<7/10:** Auto-reject

## Import Strategy

### Phase 1: Bugs.jar (This Week)
1. Clone repository
2. Parse bug metadata
3. Extract before/after code
4. Generate test templates
5. Auto-rate quality
6. Import high-quality only (9-10/10)
7. Target: +800-1,000 lessons

### Phase 2: BugSwarm (Next Week)
1. Setup BugSwarm CLI
2. Download reproducible pairs
3. Extract code changes
4. Use Docker test validation
5. Auto-rate quality
6. Import 9-10/10 only
7. Target: +2,000-2,500 lessons

### Phase 3: Fill Remaining Gap
- Use agent generation for remaining ~14,000
- Multi-language distribution
- All CERF levels
- 100% quality controlled

## Implementation Timeline

**Week 1:**
- ✅ Add quality_rating column
- ✅ Create auto-rating system
- ✅ Rate existing 2,357 lessons
- 🔄 Import Bugs.jar
- Target: 3,200 lessons

**Week 2:**
- Import BugSwarm (high-quality subset)
- Import QuixBugs
- Target: 5,500 lessons

**Weeks 3-4:**
- Agent generation at scale
- Quality control and deduplication
- Target: 10,000 lessons

**Weeks 5-8:**
- Continue agent generation
- Import additional sources as needed
- Target: 20,000+ lessons

## Success Criteria

- ✅ Zero duplicates (name + content hash)
- ✅ All lessons rated (quality_rating column filled)
- ✅ <5% lessons below 9/10 (manual review queue)
- ✅ Multi-language distribution maintained
- ✅ CERF distribution balanced
- ✅ All lessons have real, executable tests
- ✅ Ready for GNN training at 90% target accuracy