#!/bin/bash
# Run SupportBench eval on all 9 datasets.
# Usage: ./scripts/run_all_evals.sh [system]   (default: supportbot)
#        ./scripts/run_all_evals.sh baseline
set -e
source .env 2>/dev/null

SYSTEM="${1:-supportbot}"
SPLIT="${SPLIT:-100}"
HISTORY="${HISTORY:-900}"

# All 9 datasets: 3 Ukrainian, 3 English, 2 Spanish, 1 mixed
DATASETS="ua_ardupilot ua_selfhosted mikrotik_ua tasmota 3dprinting grapheneos lineageos domotica_es naseros"

FAILED=""
for ds in $DATASETS; do
    echo "========================================"
    echo "Dataset: $ds — $SYSTEM (split=$SPLIT history=$HISTORY)"
    echo "========================================"
    OUTPUT="results/eval_${ds}_${SYSTEM}.json"
    if python3 scripts/eval_supportbench.py \
        --dataset "$ds" --split "$SPLIT" --history "$HISTORY" \
        --system "$SYSTEM" \
        --output "$OUTPUT" 2>&1; then
        echo "OK: $ds → $OUTPUT"
    else
        echo "FAILED: $ds $SYSTEM"
        FAILED="$FAILED $ds"
    fi
    echo ""
done

echo "========================================"
if [ -n "$FAILED" ]; then
    echo "FAILED datasets:$FAILED"
    exit 1
else
    echo "All done! Results in results/"
fi
