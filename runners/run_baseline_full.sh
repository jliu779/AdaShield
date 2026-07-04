#!/usr/bin/env bash
# Deprecated: use runners/run_ecso_full.sh
echo "NOTE: run_baseline_full.sh now delegates to run_ecso_full.sh (ECSO-only pipeline)" >&2
exec bash "$(dirname "$0")/run_ecso_full.sh" "$@"
