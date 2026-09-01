#!/usr/bin/env bash
# MoE -> field-engine pipeline in one run.
# Anything missing (pip packages, corpus, checkpoint) is downloaded/fetched
# automatically.
# Examples: ./run_pipeline.sh            # full run (bootstrap+transform+verify)
#          ./run_pipeline.sh --smoke    # quick wiring run
set -e
cd "$(dirname "$0")"
exec python3 pipeline.py all "$@"
