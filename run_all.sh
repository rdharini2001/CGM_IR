#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m src.study
python -m src.sensitivity
python -m analysis.calibration_coverage
python scripts/build_site.py
python scripts/build_supplement.py
python -m unittest discover -s tests -v
