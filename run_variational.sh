#!/usr/bin/env bash
# run_seeds.sh
set -euo pipefail

PYTHON=python
SCRIPT=main_variational.py

# Usage:
#   ./run_seeds.sh                 # uses default seeds below
#   ./run_seeds.sh 1 2 3 4 5       # custom seeds
#
# NOTE: If your script expects a positional seed (not --seed),
# change the line inside the loop to:  "$PYTHON" "$SCRIPT" "$s"

# If seeds are passed on the command line, use them; otherwise use defaults.
if [ "$#" -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(seq 1 25)
fi

for s in {0..1}; do
  echo "=== Running seed ${s} ==="
  "$PYTHON" "$SCRIPT" --seed "$s"
done

echo "All runs completed."