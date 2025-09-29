# Nerion Learning Cycle – Copy/Paste Commands

Use these four steps to keep the Digital Physicist self-improving. Every command block can be pasted straight into the terminal.

---

## Step 1 – Generate Lessons
Populate `out/learning/curriculum.sqlite` with fresh lessons.

### Option 1: Manual prompt (single lesson)
```bash
python -m nerion_digital_physicist.generation.curriculum_generator --description "Fix the off-by-one bug in range iteration" --name fix_off_by_one
```

### Option 2: Template sampler (batch scripted scenarios)
```bash
python -m nerion_digital_physicist.generation.service 10 --curriculum --output generation/generated_tasks
```

**Bias toward advanced curriculum**
```bash
python -m nerion_digital_physicist.generation.service 10 --curriculum --output generation/generated_tasks --templates '{"advanced_curriculum":4, "bug_off_by_one":1}'
```

### Option 3: Queue + workers (continuous automation)
```bash
python -m nerion_digital_physicist.generation.orchestrator --count 20 --curriculum
```
(Open additional terminals and run `python -m nerion_digital_physicist.generation.worker` in each.)

### Option 4: Full learning orchestrator loop
```bash
python -m nerion_digital_physicist.learning_orchestrator --max-lessons 20 --auto-refresh
```
(The orchestrator samples templates, vets lessons, and can trigger training refreshes automatically.)

All successful lessons are recorded in `out/learning/curriculum.sqlite`, with telemetry in `out/learning/structural_metrics.jsonl` and artefact snapshots under `out/learning/artifacts/`.

---

## Step 2 – Refresh the Structural Brain
Run masked-node pretraining and warm-start supervised training.

### One command
```bash
scripts/run_pretraining_cycle.sh out/learning/curriculum.sqlite experiments/datasets/gnn/latest experiments/runs/gnn_pretrain experiments/runs/gnn 5e-4 25 0.2
```

### Individual steps (if you need to tune something)
```bash
python -m nerion_digital_physicist.training.dataset_builder --db out/learning/curriculum.sqlite --output-dir experiments/datasets/gnn/latest --mode pretrain
```
```bash
python -m nerion_digital_physicist.training.pretrain --dataset experiments/datasets/gnn/latest/pretrain/<timestamp>/dataset.pt --output-dir experiments/runs/gnn_pretrain
```
```bash
python -m nerion_digital_physicist.training.run_training --dataset experiments/datasets/gnn/latest/<timestamp>/dataset.pt --output-dir experiments/runs/gnn --architecture gat --hidden-channels 256 --pooling sum --residual --dropout 0.3 --attention-heads 4 --epochs 25 --lr 5e-4 --val-ratio 0.2 --pretrained digital_physicist_pretrain.pt
```

---

## Step 3 – Monitor Health
Watch lesson pass rates and training metrics.

**Structural pass rate / Δ summary**
```bash
scripts/structural_metrics_report.py --hours 24 --limit 200
```

**Training leaderboard**
```bash
python -m nerion_digital_physicist.training.metrics_report --runs-dir experiments/runs/gnn --top 5
```

**Optional warm-start sweep**
```bash
python -m nerion_digital_physicist.training.sweep --dataset experiments/datasets/gnn/latest/<timestamp>/dataset.pt --output-dir experiments/runs/gnn_sweeps_pretrain --architecture gat --hidden 256,384 --lr 1e-3,5e-4 --dropout 0.2,0.3 --epochs 20 --pooling sum --residual true --pretrained digital_physicist_pretrain.pt
```
Promote a sweep result only if the improvement repeats and telemetry stays healthy.

---

## Step 4 – Repeat
1. Generate lessons (Step 1).
2. Refresh the brain (Step 2).
3. Review telemetry (Step 3).

If pass rate falls or structural deltas shrink, inspect failing lessons, adjust template weights or prompts, and loop back to Step 1.
