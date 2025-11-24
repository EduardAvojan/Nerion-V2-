# Nerion Project - Operational Guide

**Last Updated:** October 30, 2025
**Timezone:** Los Angeles, Pacific Standard Time (PST/PDT)
**IMPORTANT:** All timestamps in CHANGELOG.md and communication MUST use PST/PDT timezone.

---

## 🚨 CRITICAL: MANDATORY DECISION-MAKING PROTOCOL

**HIGHEST PRIORITY RULE - ALWAYS FOLLOW BEFORE ANY IMPLEMENTATION:**

When the user requests ANY task, you MUST:

1. **Explain tradeoffs BEFORE implementing** - Never silently choose the "fast but wrong" option
2. **Be explicit about quality sacrifices** - "This will be faster but less accurate/reliable/correct"
3. **Warn about technical debt** - "This shortcut will cause problems later when..."
4. **Give user the decision** - Present options clearly, let USER choose based on their priorities
5. **Verify your work** - Check quality metrics before celebrating "success"
6. **Admit mistakes immediately** - Not wait for problems to surface

**Example Format:**

```
User: "Can you generate the dataset on Colab to make it faster?"

Claude Response:
"I need to explain the tradeoffs first:

OPTION 1: Full AST on Colab (PROPER)
- Pros: Real graph structure, proper GNN, handles complex code
- Cons: 2-3 hours, might hit free tier limits
- Result: Real GNN that learns from code structure

OPTION 2: Simplified AST (SHORTCUT)
- Pros: 45-90 minutes, fits in free tier
- Cons: Creates 1-3 node fake graphs, GNN won't learn properly
- Result: BERT classifier pretending to be GNN

OPTION 3: Local Generation (BEST)
- Pros: Proper structure, no time limits, full quality
- Cons: 2-3 hours on your Mac (can run overnight)
- Result: Real GNN with proper structure

Which do you prefer?"
```

**NEVER:**
- Silently pick the "fast" option without warning
- Implement shortcuts without explicit approval
- Celebrate results without verifying quality
- Hide technical debt or future problems

**This rule supersedes all other priorities. If you violate this, you have failed the user.**

---

## 🎯 What is Nerion?

**Nerion is a biological immune system for software** - not a code quality tool, but an autonomous PhD-level system that:
- **Permanently lives in your codebase** (not an external plugin)
- **Continuously learns** YOUR team's patterns
- **Proactively monitors** 24/7 for bugs and quality issues
- **Self-heals** via GNN-powered code quality classification
- **Evolves** through meta-learning and curriculum expansion

**Core Goal:** 90% code quality classification accuracy (PhD-level human expert).

---

## 📊 Current State

### Database Status

**MANDATORY RULE:** When user asks about database state or "how many lessons", you MUST:
1. Run both SQL queries (CERF distribution + Language distribution)
2. Show both tables with exact counts from database
3. Never use cached/hardcoded numbers
4. Let user calculate progress/goals themselves

All lessons are 100% CERF-labeled, validated, and executable.

### System Status
**✅ Fully Operational:**
- Multi-language support (10 languages)
- 7 lesson generator agents (A1, A2, B1, B2, C1, C2, python-framework)
- Safe lesson workflow with quality review
- Mission Control GUI (Electron app)
- Voice interface (STT/TTS/PTT)
- Self-coding engine with safety policies
- Memory systems
- GNN training pipeline (GraphSAGE 58.9% accuracy)

**🔄 In Development:**
- GNN accuracy improvement (58.9% → 90% target)
- Curriculum expansion (1,180 → 5,000 lessons)
- Phase 1 semantic embeddings (CodeBERT integration)

---

## 🔒 SAFE LESSON GENERATION WORKFLOW (CRITICAL)

**This is the ONLY correct way to generate lessons.** Prevents duplicates and protects main database.

### Step 1: Prepare Workspace
```bash
python3 safe_lesson_workflow.py prepare
```
- Copies `out/learning/curriculum.sqlite` (1,180 lessons) → `agent_generated_curriculum.sqlite`
- Agents can now check for duplicates before generating
- Main DB is READ-ONLY (never modified directly)

### Step 2: Activate Agent
Use Task tool to activate desired agent:
- `cerf-a1-programming-lesson-generator`
- `cerf-a2-programming-lesson-generator`
- `cerf-b1-programming-lesson-generator`
- `cerf-b2-programming-lesson-generator`
- `cerf-c1-programming-lesson-generator`
- `cerf-c2-programming-lesson-generator`
- `python-framework-lesson-generator` (NumPy, Pandas, Flask, FastAPI, SQLAlchemy)

Agent writes NEW lessons to workspace (existing + new).

### Step 3: Review Quality
```bash
python3 safe_lesson_workflow.py review
```
**Automated quality checks:**
- No placeholder variables (CODE, TEST_TYPE, BEFORE_CODE, AFTER_CODE)
- Test code has imports/functions
- Code not trivially short (<20 chars)
- before_code ≠ after_code
- All required fields present

**If review FAILS:** Fix agent prompt and regenerate. Do NOT merge.

### Step 4: Merge New Lessons
```bash
python3 safe_lesson_workflow.py merge
```
- Reads ALL lessons from workspace
- SafeCurriculumDB rejects duplicates automatically (name + SHA256 content hash)
- Only NEW lessons added to main DB
- Automatic backup created before merge

### Step 5: Cleanup
```bash
python3 safe_lesson_workflow.py cleanup
```
- Deletes workspace database
- Main DB verified intact

### Safety Guarantees
- Main DB never at risk (read-only during prepare)
- Automatic duplicate detection (name + content hash)
- No copies left behind (workspace deleted after cleanup)
- Quality gate prevents broken lessons from entering production
- Automatic backups before every merge

---

## 🗂️ File Structure

```
/Users/ed/Nerion-V2/
├── app/                           # Application layer
│   ├── chat/                      # Chat engine, voice, intents
│   ├── parent/                    # Task planning system
│   ├── ui/holo-app/               # Mission Control Electron app
│   ├── api/                       # REST API, terminal server
│   └── learning/                  # Upgrade agent
│
├── selfcoder/                     # Self-modification engine
│   ├── planner/                   # Plan generation
│   ├── actions/                   # Code modification primitives
│   ├── ast_editor/                # AST-based editing
│   ├── policy/                    # Safety policies
│   ├── security/                  # Malicious code detection
│   └── learning/                  # Behavioral coach
│
├── nerion_digital_physicist/      # GNN deep learning
│   ├── agent/
│   │   ├── brain.py               # GNN architectures
│   │   ├── data.py                # AST → Graph
│   │   └── semantics.py           # CodeBERT embeddings
│   ├── training/
│   │   ├── run_training.py        # Training loop
│   │   └── dataset_builder.py    # Curriculum → PyTorch
│   └── db/
│       └── safe_curriculum.py     # Database wrapper
│
├── voice/                         # Voice interface (STT/TTS/PTT)
├── daemon/                        # Immune system daemon (24/7)
├── config/                        # Configuration (tools.yaml, intents.yaml)
│
├── out/                           # Runtime data
│   ├── learning/
│   │   └── curriculum.sqlite      # 1,180 validated lessons
│   └── training_runs/             # GNN training history
│
├── .claude/agents/                # 7 lesson generator agents
├── safe_lesson_workflow.py        # Lesson generation workflow
├── digital_physicist_brain.pt     # Current GNN weights
├── CLAUDE.md                      # This file (operational rules)
└── CHANGELOG.md                   # History (7-day rolling window)
```

---

## 🎮 Quick Commands

### Lesson Generation
```bash
# Generate 30 A1 lessons
python3 safe_lesson_workflow.py prepare
# Activate agent via Task tool: cerf-a1-programming-lesson-generator
python3 safe_lesson_workflow.py review
python3 safe_lesson_workflow.py merge
python3 safe_lesson_workflow.py cleanup
```

### Database Queries

**MANDATORY RULE:** When user asks "how many lessons" or about database state, you MUST:
1. Query database directly (never guess or use cached numbers)
2. Show BOTH tables: CERF distribution + Language distribution
3. Show exact numbers from database, not approximations

```bash
# Table 1: CERF distribution
sqlite3 out/learning/curriculum.sqlite "SELECT focus_area, COUNT(*) as count FROM lessons WHERE focus_area IN ('A1', 'A2', 'B1', 'B2', 'C1', 'C2') GROUP BY focus_area ORDER BY focus_area;"

# Table 2: Language distribution
sqlite3 out/learning/curriculum.sqlite "SELECT language, COUNT(*) as count FROM lessons GROUP BY language ORDER BY count DESC;"

# Total count
sqlite3 out/learning/curriculum.sqlite "SELECT COUNT(*) FROM lessons;"
```

### GNN Training
```bash
# Train GraphSAGE with current dataset
python3 -m nerion_digital_physicist.training.run_training \
  --dataset experiments/datasets/gnn/final_complete/supervised/*/dataset.pt \
  --architecture sage \
  --epochs 50 \
  --batch-size 32
```

### Mission Control GUI
```bash
cd app/ui/holo-app
npm run dev  # Development mode
# OR
npm run build && open ../../Nerion.app  # Production
```

### Voice Interface
```bash
python3 -m app.chat.engine --voice --profile whisper_fast
# Press Ctrl+Space for push-to-talk
```

---

## 📝 Maintenance Guidelines

### CLAUDE.md (This File)
**Purpose:** Timeless operational rules for Claude to operate Nerion effectively.

**Update when:**
- Core capabilities change
- Database state significantly changes
- Critical workflows change
- File structure changes

**Do NOT include:**
- Historical events (use CHANGELOG.md)
- In-progress work (use CHANGELOG.md)
- Detailed technical specs (doesn't change often)
- Use cases or examples (educational, not operational)

### CHANGELOG.md
**Purpose:** Rolling 7-day history of confirmed changes.

**Rules:**
- ✅ **ONLY add AFTER change is tested and working**
- ✅ Include timestamp (YYYY-MM-DD HH:MM PDT/PST) - **MUST use Pacific Time**
- ✅ Use types: ADD, UPDATE, REMOVE, FIX, REFACTOR
- ✅ **Delete entries older than 7 days** (keep it lean)
- ❌ Do NOT add experimental/in-progress work
- ❌ Do NOT add failed attempts

**When to update:**
- After completing and testing any feature
- After fixing and verifying any bug
- After making and confirming any refactor
- When current work status changes

### Quality Control
- **CHANGELOG.md = Factual recent history** (7-day window)
- **CLAUDE.md = Operational rules** (timeless)
- **If you create temporary scripts:** Delete them when done
- **If you create documentation:** Check if existing file can be updated instead

---

## 🔧 Environment Variables

```bash
# .env file
NERION_V2_GEMINI_KEY=<your_key>        # For embeddings/chat
CLAUDE_API_KEY=<your_key>              # For chat
DEEPSEEK_API_KEY=<your_key>            # For chat
NERION_SEMANTIC_PROVIDER=codebert      # or "gemini"
```

---

*CLAUDE.md = Operational rules for Claude*
*CHANGELOG.md = Recent history (7-day rolling window)*
*See CHANGELOG.md for current work and recent changes*
