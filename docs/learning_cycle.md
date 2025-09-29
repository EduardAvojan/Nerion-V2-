# Nerion Learning Cycle – Copy/Paste Commands

Follow the four steps below. Every code block contains a command you can paste straight into the terminal.

---

## Step 1 – Generate Lessons
Populate `out/learning/curriculum.sqlite` with new lessons.

**Single custom lesson**
```bash
python -m nerion_digital_physicist.generation.curriculum_generator --description "Fix the off-by-one bug in range iteration" --name fix_off_by_one
```

**Batch from templates (mixed topics)**
```bash
python -m nerion_digital_physicist.generation.service 10 --curriculum --output generation/generated_tasks
```

**Batch biased toward advanced curriculum**
```bash
python -m nerion_digital_physicist.generation.service 10 --curriculum --output generation/generated_tasks --templates '{"advanced_curriculum":4, "bug_off_by_one":1}'
```

**Continuous queue + workers**
```bash
python -m nerion_digital_physicist.generation.orchestrator --count 20 --curriculum
```
(Open one or more extra terminals and run `python -m nerion_digital_physicist.generation.worker` in each.)

---

## Step 2 – Refresh the Structural Brain
Re-export the curriculum, run masked-node pretraining, and warm-start supervised training.

**One command**
```bash
scripts/run_pretraining_cycle.sh out/learning/curriculum.sqlite experiments/datasets/gnn/latest experiments/runs/gnn_pretrain experiments/runs/gnn 5e-4 25 0.2
```

**Individual steps (if you want to tweak anything)**
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
Check structural pass rates and training results.

**Structural telemetry summary (last 24h)**
```bash
scripts/structural_metrics_report.py --hours 24 --limit 200
```

**Training leaderboard**
```bash
python -m nerion_digital_physicist.training.metrics_report --runs-dir experiments/runs/gnn --top 5
```

**Optional sweep (warm-started)**
```bash
python -m nerion_digital_physicist.training.sweep --dataset experiments/datasets/gnn/latest/<timestamp>/dataset.pt --output-dir experiments/runs/gnn_sweeps_pretrain --architecture gat --hidden 256,384 --lr 1e-3,5e-4 --dropout 0.2,0.3 --epochs 20 --pooling sum --residual true --pretrained digital_physicist_pretrain.pt
```

---

## Step 4 – Repeat
1. Generate more lessons (Step 1).
2. Refresh the brain (Step 2).
3. Review telemetry (Step 3).

If pass rates fall or deltas shrink, inspect the failing lessons, adjust template weights or prompts, then loop back to Step 1.
