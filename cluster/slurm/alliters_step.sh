#!/usr/bin/env bash
# One-shot top-up for the breast test-scoring array. Idempotent: submits as many
# unsubmitted windows as fit under the turbo submit cap, records each in a state
# file, then EXITS (no daemon — call it once per loop tick). Counts real task
# count via explicit -o "%j" (default squeue truncates the name column -> 0).
#
# Pre-seed: window 1-50 (772589) + pilot 221-239 (772579) are already submitted,
# so they are written into the state file as done and never resubmitted.
set -u
cd /cluster/maier/Agent4CT
M=/cluster/maier/Agent4CT/agentic_cfgs/breast_alliters/manifest.tsv
SB=cluster/slurm/breast_alliters_array.sbatch
STATE=docs/_debug/alliters_submitted.txt
LOG=docs/_debug/alliters_feeder.log
WINDOW=25; FLOOR=33
RANGES="1-50 221-239 51-75 76-100 101-125 126-150 151-175 176-200 201-220 \
240-264 265-289 290-314 315-339 340-364 365-389 390-414 415-439 \
440-464 465-489 490-514 515-515"

touch "$STATE"
# Seed the two already-submitted ranges once.
grep -qx "1-50"    "$STATE" || echo "1-50"    >>"$STATE"
grep -qx "221-239" "$STATE" || echo "221-239" >>"$STATE"

count() { squeue -u maier -h -r -o "%j" 2>/dev/null | grep -c breast-alli; }

n_sub=0
for R in $RANGES; do
  grep -qx "$R" "$STATE" && continue          # already submitted
  N=$(count)
  if [ "$N" -le "$FLOOR" ]; then
    OUT=$(sbatch --array=${R}%8 --time=00:30:00 --export=ALL,MANIFEST="$M" "$SB" 2>&1)
    if echo "$OUT" | grep -q "Submitted batch"; then
      echo "$R" >>"$STATE"
      echo "[step] $(date -u +%FT%TZ) submitted $R (N=$N) -> ${OUT##* }" >>"$LOG"
      n_sub=$((n_sub+1)); sleep 25
    else
      echo "[step] $(date -u +%FT%TZ) $R rejected (N=$N)" >>"$LOG"; break
    fi
  else
    break                                       # cap reached; try again next tick
  fi
done
REMAIN=$(comm -23 <(printf "%s\n" $RANGES | sort -u) <(sort -u "$STATE") | wc -l)
echo "STEP_DONE submitted_now=$n_sub remaining_ranges=$REMAIN cur_tasks=$(count)"
