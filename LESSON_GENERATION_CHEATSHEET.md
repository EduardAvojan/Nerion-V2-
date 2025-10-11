# Lesson Generation Cheatsheet

Quick reference for generating high-quality curriculum lessons with Claude Sonnet 4.5.

---

## 🚀 Quick Start

```bash
# 1. Set API key (one time)
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# 2. Test the integration
python test_claude_integration.py

# 3. Generate lessons
./generate_batch_lessons.sh 100 a1    # 100 beginner lessons
```

---

## 📊 Check Current Status

### View curriculum coverage
```bash
./check_curriculum_coverage.sh
```

### Count total lessons
```bash
sqlite3 out/learning/curriculum.sqlite "SELECT COUNT(*) FROM lessons;"
```

### View recent lessons
```bash
sqlite3 out/learning/curriculum.sqlite "SELECT name, focus_area, timestamp FROM lessons ORDER BY timestamp DESC LIMIT 10;"
```

### Breakdown by CEFR level
```bash
sqlite3 out/learning/curriculum.sqlite "
SELECT
  CASE
    WHEN name LIKE 'a1_%' THEN 'A1-Beginner'
    WHEN name LIKE 'a2_%' THEN 'A2-Elementary'
    WHEN name LIKE 'b1_%' THEN 'B1-Intermediate'
    WHEN name LIKE 'b2_%' THEN 'B2-UpperIntermediate'
    WHEN name LIKE 'c1_%' THEN 'C1-Advanced'
    WHEN name LIKE 'c2_%' THEN 'C2-Expert'
    ELSE 'Other'
  END as level,
  COUNT(*)
FROM lessons
GROUP BY level;"
```

---

## 🎯 Generate Lessons

### Basic Usage
```bash
./generate_batch_lessons.sh [LESSONS] [CATEGORY] [WORKERS]
```

### Examples

**Random mix (all levels):**
```bash
./generate_batch_lessons.sh 500           # 500 lessons, 5 workers
./generate_batch_lessons.sh 100           # 100 lessons, 5 workers
```

**Specific CEFR levels:**
```bash
./generate_batch_lessons.sh 100 a1        # 100 A1 beginner lessons
./generate_batch_lessons.sh 100 a2        # 100 A2 elementary lessons
./generate_batch_lessons.sh 100 b1        # 100 B1 intermediate lessons
./generate_batch_lessons.sh 100 b2        # 100 B2 security/concurrency lessons
./generate_batch_lessons.sh 100 c1        # 100 C1 advanced lessons
./generate_batch_lessons.sh 100 c2        # 100 C2 expert lessons
```

**Specific topics:**
```bash
./generate_batch_lessons.sh 50 refactoring           # Refactoring lessons
./generate_batch_lessons.sh 50 security_hardening    # Security lessons
./generate_batch_lessons.sh 50 performance_optimization
./generate_batch_lessons.sh 50 bug_fixing
```

**More workers (faster, but watch rate limits):**
```bash
./generate_batch_lessons.sh 200 b2 10    # 200 B2 lessons, 10 workers
./generate_batch_lessons.sh 500 "" 8     # 500 random, 8 workers
```

---

## 📈 Monitor Progress

### Watch worker logs (real-time)
```bash
tail -f logs/worker_1.log
```

### Check all worker logs
```bash
tail -f logs/worker_*.log
```

### Live lesson count
```bash
watch -n 5 'sqlite3 out/learning/curriculum.sqlite "SELECT COUNT(*) FROM lessons;"'
```

### Check success rate
```bash
# View recent structural metrics
tail -20 out/learning/structural_metrics.jsonl | grep -E "passed|failed"
```

---

## 🧪 Testing

### Test Claude API integration
```bash
python test_claude_integration.py
```

### Test single lesson generation
```bash
python -m nerion_digital_physicist.learning_orchestrator \
  --provider anthropic:claude-sonnet-4-5-20250929
```

### Test specific category
```bash
python -m nerion_digital_physicist.learning_orchestrator \
  --provider anthropic:claude-sonnet-4-5-20250929 \
  --category a1
```

---

## 💰 Cost Tracking

### Current pricing (Claude Sonnet 4.5)
- **Input:** $3 per 1M tokens
- **Output:** $15 per 1M tokens
- **Average per lesson:** ~$0.20 (including retries)

### Budget calculator
```bash
# With $50 budget
echo "scale=0; 50 / 0.20" | bc    # ≈ 250 lessons

# With $100 budget
echo "scale=0; 100 / 0.20" | bc   # ≈ 500 lessons
```

### Check your Anthropic dashboard
https://console.anthropic.com/settings/usage

---

## 🎓 CEFR Categories Reference

### A1 - Beginner (16 types)
Basic syntax, variables, loops, simple conditionals
```bash
a1_variable_scope_errors
a1_type_conversion_bugs
a1_list_index_errors
a1_basic_loop_bugs
# ... 12 more
```

### A2 - Elementary (16 types)
OOP basics, file handling, decorators
```bash
a2_list_mutation_bugs
a2_basic_oop_errors
a2_file_handling_errors
a2_json_parsing_issues
# ... 12 more
```

### B1 - Intermediate (16 types)
Design patterns, database, testing
```bash
b1_design_pattern_issues
b1_error_handling_strategies
b1_database_transactions
b1_testing_strategies
# ... 12 more
```

### B2 - Upper Intermediate (16 types)
Security, concurrency, performance
```bash
b2_race_conditions
b2_sql_injection_prevention
b2_xss_vulnerabilities
b2_circuit_breaker_pattern
# ... 12 more
```

### C1 - Advanced (22 types)
Specialized domains
```bash
refactoring
bug_fixing
feature_implementation
performance_optimization
security_hardening
concurrency_patterns
distributed_systems
# ... 15 more
```

### C2 - Expert (40+ types)
Advanced algorithms, distributed systems, cryptography
```bash
c2_distributed_locking
c2_consensus_algorithms
c2_eventual_consistency
c2_advanced_concurrency
c2_cryptographic_implementations
# ... 35+ more
```

---

## 🔧 Troubleshooting

### API key not found
```bash
# Check if set
echo $ANTHROPIC_API_KEY

# Set it
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# Or add to .env file
echo 'ANTHROPIC_API_KEY=sk-ant-your-key-here' >> .env
```

### Rate limits
If you hit rate limits, reduce workers:
```bash
./generate_batch_lessons.sh 100 a1 3    # Only 3 workers instead of 5
```

### Database locked
If you get "database is locked" errors:
```bash
# Stop all workers
pkill -f learning_orchestrator

# Wait a moment, then restart
./generate_batch_lessons.sh 100 a1
```

### Check for running processes
```bash
ps aux | grep learning_orchestrator
```

### Kill stuck processes
```bash
pkill -f learning_orchestrator
```

---

## 📁 Important Files

### Configuration
- `.env` - API keys and config
- `config/model_catalog.yaml` - Model definitions

### Scripts
- `generate_batch_lessons.sh` - Main generation script
- `check_curriculum_coverage.sh` - Coverage report
- `test_claude_integration.py` - Integration test

### Database
- `out/learning/curriculum.sqlite` - All lessons
- `out/learning/structural_metrics.jsonl` - Vetting metrics

### Logs
- `logs/worker_*.log` - Worker logs
- `out/learning/prompts.jsonl` - Prompt metrics

---

## 📋 Common Workflows

### Generate balanced curriculum (250 lessons, $50)
```bash
./generate_batch_lessons.sh 40 a1     # Beginners
./generate_batch_lessons.sh 40 a2     # Elementary
./generate_batch_lessons.sh 40 b1     # Intermediate
./generate_batch_lessons.sh 40 b2     # Security
./generate_batch_lessons.sh 90        # Random mix
```

### Focus on security (250 lessons, $50)
```bash
./generate_batch_lessons.sh 100 b2              # Security fundamentals
./generate_batch_lessons.sh 100 security_hardening
./generate_batch_lessons.sh 50 c2               # Advanced security
```

### Fill missing levels
```bash
# Check what you have
./check_curriculum_coverage.sh

# Generate what's missing
./generate_batch_lessons.sh 50 a1    # If you have 0 A1 lessons
./generate_batch_lessons.sh 50 a2    # If you have 0 A2 lessons
```

---

## 🔍 Advanced Database Queries

### Find lessons by topic
```bash
sqlite3 out/learning/curriculum.sqlite \
  "SELECT name, description FROM lessons WHERE description LIKE '%SQL injection%';"
```

### Count lessons by focus area
```bash
sqlite3 out/learning/curriculum.sqlite \
  "SELECT focus_area, COUNT(*) FROM lessons GROUP BY focus_area ORDER BY COUNT(*) DESC;"
```

### Export lessons to JSON
```bash
sqlite3 out/learning/curriculum.sqlite \
  "SELECT json_object('name', name, 'description', description) FROM lessons LIMIT 10;"
```

### Recently failed lessons
```bash
tail -100 out/learning/structural_metrics.jsonl | \
  grep '"structural_status": "failed"' | \
  jq -r '.lesson'
```

---

## ⚡ Performance Tips

1. **Parallel workers:** More workers = faster, but watch rate limits
   - Safe: 5 workers
   - Aggressive: 8-10 workers

2. **Category filtering:** Specific categories generate faster
   ```bash
   ./generate_batch_lessons.sh 100 a1    # Faster (simpler lessons)
   ./generate_batch_lessons.sh 100 c2    # Slower (complex lessons)
   ```

3. **Monitor and adjust:** Watch logs to see if workers are waiting
   ```bash
   tail -f logs/worker_1.log
   ```

---

## 📞 Support

### Check Claude API status
https://status.anthropic.com/

### View API usage
https://console.anthropic.com/settings/usage

### Get more API credits
https://console.anthropic.com/settings/billing

---

## 🎯 Quick Commands Summary

```bash
# Check what you have
./check_curriculum_coverage.sh

# Generate 100 beginner lessons
./generate_batch_lessons.sh 100 a1

# Monitor progress
tail -f logs/worker_1.log

# Count lessons
sqlite3 out/learning/curriculum.sqlite "SELECT COUNT(*) FROM lessons;"

# Test API
python test_claude_integration.py
```

---

**Last Updated:** October 2025
**Model:** Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`)
**Cost:** ~$0.20 per lesson
