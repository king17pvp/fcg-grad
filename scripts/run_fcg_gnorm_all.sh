#!/usr/bin/env bash
#
# run_pg_gnorm_all.sh
#
# Batch-run ablations/run_eval_fcg_sentiment_gnorm.py across every dataset (excluding imdb),
# every model, and every gnorm mode.  Each run writes a log under logs/pg_gnorm/
# and the final metrics are aggregated into a single CSV.
#
# Usage:
#   bash run_pg_gnorm_all.sh                      # all combos, default args
#   bash run_pg_gnorm_all.sh --steps 50 --topk 10 # override extra args
#

set -uo pipefail

#  Configuration 

MODELS=(   distilbert bert    roberta )
DATASETS=( sst2       rotten  )            # imdb excluded as requested
GNORMS=(   sign_norm  sign_magl2  sign_magl1  safe_norm  square_norm )

SCRIPT="ablations/run_eval_fcg_sentiment_gnorm.py"
LOG_DIR="logs/pg_gnorm"
CSV_FILE="${LOG_DIR}/results.csv"

# Optional: pass extra args to every run, e.g. --steps 50 --num-samples 500
EXTRA_ARGS="${@}"

#  Helper: extract a metric value from a logfile 

extract_metric() {
    # $1 = logfile path, $2 = metric label (e.g. "Log-odds", "Soft-Comp")
    # Prints the numeric value (strips trailing "s" for time, leading "+")
    local file="$1" label="$2"
    grep "^  ${label}  *:" "$file" 2>/dev/null \
        | head -1 \
        | sed 's/.*:\s*//; s/s$//; s/^+//'
}

extract_count() {
    local file="$1"
    grep "Final results" "$file" 2>/dev/null \
        | head -1 \
        | sed -n 's/.*\[\([0-9]\+\) samples\].*/\1/p'
}

#  Setup 

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: $SCRIPT not found in the current directory." >&2
    echo "       Run this script from the FCGGrad project root." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} * ${#GNORMS[@]} ))
CURRENT=0

# CSV header
echo "model,dataset,gnorm,samples,log_odds,comprehensiveness,sufficiency,soft_log_odds,soft_comp,soft_suff,avg_time_s" \
    > "$CSV_FILE"

echo "=========================================================================="
echo "  PACE Gradient — Gnorm Batch Runner"
echo "=========================================================================="
echo "  Models    : ${MODELS[*]}"
echo "  Datasets  : ${DATASETS[*]}  (imdb excluded)"
echo "  Gnorm     : ${GNORMS[*]}"
echo "  Extra args: ${EXTRA_ARGS:-<none>}"
echo "  Total runs: $TOTAL"
echo "  Log dir   : $LOG_DIR"
echo "  CSV       : $CSV_FILE"
echo "=========================================================================="
echo

#  Main loop 

FAILED=0

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for gnorm in "${GNORMS[@]}"; do

            CURRENT=$((CURRENT + 1))
            LOGFILE="${LOG_DIR}/${model}_${dataset}_${gnorm}.log"

            printf "[%3d/%3d]  model=%-10s  dataset=%-6s  gnorm=%-12s  →  %s\n" \
                   "$CURRENT" "$TOTAL" "$model" "$dataset" "$gnorm" "$LOGFILE"

            #  Run 
            if python "$SCRIPT" \
                --model    "$model"    \
                --dataset  "$dataset"  \
                --gnorm    "$gnorm"    \
                $EXTRA_ARGS \
                > "$LOGFILE" 2>&1; then

                #  Parse metrics 
                SAMPLES=$(extract_count   "$LOGFILE")
                LO=$(     extract_metric   "$LOGFILE" "Log-odds")
                COMP=$(   extract_metric   "$LOGFILE" "Comprehensiveness")
                SUFF=$(   extract_metric   "$LOGFILE" "Sufficiency")
                SLO=$(    extract_metric   "$LOGFILE" "Soft-Log-odds")
                SCOMP=$(  extract_metric   "$LOGFILE" "Soft-Comp")
                SSUFF=$(  extract_metric   "$LOGFILE" "Soft-Suff")
                AVGT=$(   extract_metric   "$LOGFILE" "Avg time/sample")

                #  Append CSV row 
                echo "${model},${dataset},${gnorm},${SAMPLES:-NA},${LO:-NA},${COMP:-NA},${SUFF:-NA},${SLO:-NA},${SCOMP:-NA},${SSUFF:-NA},${AVGT:-NA}" \
                    >> "$CSV_FILE"

                echo "         ✓   samples=${SAMPLES:-?}  LO=${LO:-NA}  Comp=${COMP:-NA}  Suff=${SUFF:-NA}  avgT=${AVGT:-NA}s"
            else
                echo "         ✗   FAILED  (see $LOGFILE)"
                echo "${model},${dataset},${gnorm},FAILED,,,,,,," >> "$CSV_FILE"
                FAILED=$((FAILED + 1))
            fi

            echo

        done
    done
done

echo "=========================================================================="
echo "  Done — $((TOTAL - FAILED))/$TOTAL succeeded, $FAILED failed."
echo "  Logs : $LOG_DIR/"
echo "  CSV  : $CSV_FILE"
echo "=========================================================================="

#  Quick CSV preview 

if [[ -f "$CSV_FILE" ]]; then
    echo
    echo "CSV preview:"
    echo ""
    column -t -s, "$CSV_FILE" | head -20
    echo ""
    echo "  ($(wc -l < "$CSV_FILE") rows incl. header)"
fi
