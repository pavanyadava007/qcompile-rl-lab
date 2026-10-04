#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
for s in 0 1 2; do
  .venv/bin/python scripts/train_vqe_rl.py --iters 250 --problems 8 --group 8 --seed $s > results/vqe_rl/train_gnn_s$s.txt 2>&1
  .venv/bin/python scripts/train_vqe_rl.py --iters 250 --problems 8 --group 8 --seed $s --no-mp > results/vqe_rl/train_mlp_s$s.txt 2>&1
done
echo DONE > results/vqe_rl/TRAINING_DONE
