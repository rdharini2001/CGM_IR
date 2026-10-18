#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-}" in
  --recipe)
    python src/weight_recipe.py --mode all
    python src/trace_forward.py --write-example >/dev/null
    python src/plot_weight_recipe.py
    python scripts/build_site.py
    python -m pytest -q
    ;;
  "" | --verify)
    python src/verify_forward.py
    python src/audit_targets.py
    python src/plot_forward.py
    python scripts/build_site.py
    python -m pytest -q
    ;;
  --train | --train-all)
    python src/train_realization.py
    python src/weight_recipe.py --mode all
    python src/trace_forward.py --write-example >/dev/null
    python src/plot_weight_recipe.py
    python src/verify_forward.py
    python src/audit_targets.py
    python src/heldout_control.py
    python src/hba1c_inverse.py
    python src/clarke_population_scenario.py
    if [[ "$1" == "--train-all" ]]; then
      python src/optimizer_slopes.py
    fi
    python src/verify_supplementary.py
    python src/plot_forward.py
    python src/make_figures.py
    python scripts/build_site.py
    python -m pytest -q
    ;;
  *)
    echo "Usage: bash run_all.sh [--verify | --recipe | --train | --train-all]" >&2
    exit 2
    ;;
esac
