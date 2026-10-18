#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-}" in
  "" ) exec bash run_all.sh --train ;;
  --all ) exec bash run_all.sh --train-all ;;
  * ) echo "Usage: bash train.sh [--all]" >&2; exit 2 ;;
esac
