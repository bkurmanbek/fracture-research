#!/usr/bin/env bash
# Run all distribution visualization scripts in order.
# Usage: bash run_all_viz.sh
# Outputs go to: plots/

PYTHON=python
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Running all visualization scripts ==="
echo "Output directory: $DIR/plots"
echo ""

for script in "$DIR"/viz_*.py; do
    name=$(basename "$script")
    echo ">>> $name"
    "$PYTHON" "$script" && echo "    OK" || echo "    FAILED: $name"
    echo ""
done

echo "=== Done. Check plots/ folder ==="
