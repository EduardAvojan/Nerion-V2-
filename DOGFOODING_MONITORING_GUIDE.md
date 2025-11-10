# 🔍 How to Monitor Nerion Dogfooding

When you run the daemon in dogfooding mode, here are ALL the ways to monitor progress:

---

## 1. **Real-Time Console Output** (Immediate Feedback)

When you start the daemon, you'll see live output:

```bash
python3 -m daemon.nerion_daemon --target /Users/ed/Nerion-V2 --mode dogfood
```

**What you'll see:**
```
[2025-11-06 12:00:00] [NERION-DAEMON] INFO: Nerion Immune Daemon initialized
[2025-11-06 12:00:00] [NERION-DAEMON] INFO: Monitoring codebase: /Users/ed/Nerion-V2
[2025-11-06 12:00:01] [NERION-DAEMON] INFO: Starting immune system daemon
[2025-11-06 12:00:01] [NERION-DAEMON] INFO: Socket server started at ~/.nerion/daemon.sock
[2025-11-06 12:00:05] [NERION-DAEMON] INFO: Initializing GNN (SAGE, 85.2% accuracy)
[2025-11-06 12:00:10] [NERION-DAEMON] INFO: Scanning nerion_digital_physicist/training/run_training.py
[2025-11-06 12:00:12] [NERION-DAEMON] WARNING: Potential issue detected in run_training.py:215
[2025-11-06 12:00:13] [NERION-DAEMON] INFO: Multi-agent verification: Security agent reviewing...
[2025-11-06 12:00:15] [NERION-DAEMON] INFO: Issue confirmed! Memory leak in dataset loading
[2025-11-06 12:00:16] [NERION-DAEMON] INFO: Auto-fix applied: Added context manager
[2025-11-06 12:00:17] [NERION-DAEMON] INFO: Tests passed ✓
[2025-11-06 12:00:18] [NERION-DAEMON] INFO: New lesson added to curriculum (bug #47)
```

---

## 2. **Mission Control GUI** (Visual Dashboard)

Launch the Electron app for a beautiful visual interface:

```bash
cd app/ui/holo-app
npm run dev
```

**What you'll see:**
- 🎯 **Live Metrics**: Bugs found, fixes applied, accuracy trend
- 📊 **Graphs**: Training loss over time, curriculum growth
- 🧬 **System Health**: GNN status, multi-agent activity, learning cycles
- 📁 **File Monitor**: Which files are being scanned
- 🐛 **Bug Feed**: Real-time stream of detected issues
- 🔬 **Curiosity Reports**: Novel patterns discovered
- 📈 **Meta-Learning Stats**: MAML adaptation speed, EWC forgetting rate

**GUI connects to daemon via Unix socket** (`~/.nerion/daemon.sock`)

---

## 3. **Daemon Log File** (Complete History)

Everything is logged to: `~/.nerion/daemon.log`

```bash
# Watch live
tail -f ~/.nerion/daemon.log

# Search for specific events
grep "Issue confirmed" ~/.nerion/daemon.log
grep "New lesson" ~/.nerion/daemon.log
grep "Learning cycle triggered" ~/.nerion/daemon.log
```

**Log format:**
```
[2025-11-06 12:00:00] [NERION-DAEMON] INFO: Event details
[2025-11-06 12:00:01] [NERION-DAEMON] WARNING: Potential issue
[2025-11-06 12:00:02] [NERION-DAEMON] ERROR: Critical problem
```

---

## 4. **Curriculum Database** (Learning Progress)

Watch lessons being added in real-time:

```bash
# Count total lessons (should grow over time)
sqlite3 out/learning/curriculum.sqlite "SELECT COUNT(*) FROM lessons;"

# Show latest 10 lessons added
sqlite3 out/learning/curriculum.sqlite "
  SELECT name, focus_area, created_at
  FROM lessons
  ORDER BY created_at DESC
  LIMIT 10;
"

# Watch for new lessons (run in separate terminal)
watch -n 5 'sqlite3 out/learning/curriculum.sqlite "SELECT COUNT(*) FROM lessons;"'
```

**Expected growth:**
- Start: 1,635 lessons
- After 1 week: ~1,650 lessons (+15 from dogfooding)
- After 1 month: ~1,700 lessons (+65 from dogfooding)

---

## 5. **Training Runs** (Model Improvement)

New training runs appear when learning cycles trigger:

```bash
# List recent training runs
ls -lt out/training_runs/ | head -10

# Watch for new runs
watch -n 10 'ls -lt out/training_runs/ | head -5'

# Check latest accuracy
python3 << 'EOF'
import torch
from pathlib import Path

runs = sorted(Path("out/training_runs").glob("*/*/digital_physicist_brain.pt"),
              key=lambda x: x.stat().st_mtime, reverse=True)
if runs:
    latest = runs[0]
    checkpoint = torch.load(latest, weights_only=False)
    print(f"Latest run: {latest.parent.name}")
    print(f"Accuracy: {checkpoint.get('val_accuracy', 'N/A')}")
EOF
```

---

## 6. **Bug Reports** (What's Being Found)

Daemon creates structured bug reports:

```bash
# View bugs found (daemon writes to JSON)
cat ~/.nerion/bugs_found.json | jq '.[] | {file, issue, confidence}'

# Count bugs by severity
cat ~/.nerion/bugs_found.json | jq 'group_by(.severity) | map({severity: .[0].severity, count: length})'
```

**Report format:**
```json
{
  "file": "nerion_digital_physicist/training/run_training.py",
  "line": 215,
  "issue": "Memory leak in dataset loading",
  "severity": "medium",
  "confidence": 0.87,
  "fix_applied": true,
  "verified_by": ["security_agent", "performance_agent"],
  "timestamp": "2025-11-06T12:00:15Z"
}
```

---

## 7. **Socket Client** (Custom Monitoring)

Connect to daemon socket for real-time metrics:

```python
# monitor_daemon.py
import asyncio
import json

async def monitor():
    reader, writer = await asyncio.open_unix_connection('~/.nerion/daemon.sock')

    # Request metrics
    writer.write(b'{"command": "get_metrics"}\n')
    await writer.drain()

    # Receive response
    data = await reader.readline()
    metrics = json.loads(data)

    print(f"Threats detected: {metrics['threats_detected']}")
    print(f"Auto-fixes applied: {metrics['auto_fixes_applied']}")
    print(f"Patterns discovered: {metrics['patterns_discovered']}")
    print(f"Current accuracy: {metrics['gnn_accuracy']}")

asyncio.run(monitor())
```

---

## 8. **Metrics Summary Command**

Quick status check:

```bash
# Create simple status script
cat > check_dogfood_status.sh << 'EOF'
#!/bin/bash
echo "🧬 NERION DOGFOODING STATUS"
echo "=========================="
echo ""
echo "📚 Curriculum:"
echo "  Lessons: $(sqlite3 out/learning/curriculum.sqlite 'SELECT COUNT(*) FROM lessons;')"
echo ""
echo "🏋️ Training Runs:"
echo "  Total runs: $(ls -d out/training_runs/*/* 2>/dev/null | wc -l)"
echo "  Latest: $(ls -td out/training_runs/*/* 2>/dev/null | head -1 | xargs basename)"
echo ""
echo "📝 Daemon Log (last 5 events):"
tail -5 ~/.nerion/daemon.log 2>/dev/null || echo "  No log yet"
echo ""
echo "🎯 Bugs Found:"
cat ~/.nerion/bugs_found.json 2>/dev/null | jq '. | length' || echo "  0"
EOF

chmod +x check_dogfood_status.sh
./check_dogfood_status.sh
```

---

## 9. **Continuous Learning Events**

Watch for learning cycle triggers:

```bash
# Learning cycles logged to daemon.log
grep "Learning cycle triggered" ~/.nerion/daemon.log

# Shows:
# [2025-11-06 14:30:00] Learning cycle triggered: 50 bugs collected
# [2025-11-06 14:30:05] Generating curriculum from 50 production bugs
# [2025-11-06 14:30:10] Auto-generated 12 new lessons
# [2025-11-06 14:30:15] Starting retraining with EWC
# [2025-11-06 15:00:00] Retraining complete: 85.2% → 86.1% (+0.9%)
```

---

## 10. **What To Watch For**

### ✅ **Good Signs:**
- Curriculum growing steadily (1-2 lessons/day)
- Bugs found → verified → fixed → tests pass
- Accuracy slowly improving (85% → 86% → 87%)
- Learning cycles completing successfully
- Low false positive rate (<10%)

### 🚨 **Warning Signs:**
- Accuracy dropping (EWC failure)
- High false positive rate (>20%)
- Fixes breaking tests
- Daemon crashes/restarts
- No bugs found (GNN too conservative)

---

## Quick Reference Card

```bash
# Start dogfooding
python3 -m daemon.nerion_daemon --target /Users/ed/Nerion-V2 --mode dogfood

# Open GUI
cd app/ui/holo-app && npm run dev

# Watch logs
tail -f ~/.nerion/daemon.log

# Check progress
./check_dogfood_status.sh

# Count lessons
sqlite3 out/learning/curriculum.sqlite "SELECT COUNT(*) FROM lessons;"

# Latest accuracy
ls -lt out/training_runs/*/digital_physicist_brain.pt | head -1
```

---

## Expected Timeline

| Time | Curriculum | Accuracy | Bugs Found | Learning Cycles |
|------|-----------|----------|------------|-----------------|
| **Day 1** | 1,635 | 85.0% | 5-10 | 0 |
| **Week 1** | 1,650 | 85.5% | 30-40 | 1 |
| **Week 2** | 1,665 | 86.2% | 50-60 | 2 |
| **Month 1** | 1,700 | 87.5% | 80-100 | 4-5 |
| **Month 3** | 1,800 | 90.0% | 150-200 | 12-15 |

---

**You have FULL visibility into every aspect of the dogfooding process!**
