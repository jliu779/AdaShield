#!/usr/bin/env bash
# Deprecated: use runners/run_ecso_all_models.sh
echo "NOTE: run_baseline_all_models.sh now delegates to run_ecso_all_models.sh" >&2
exec bash "$(dirname "$0")/run_ecso_all_models.sh" "$@"
