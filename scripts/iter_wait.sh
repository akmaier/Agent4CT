#!/bin/bash
# iter_wait.sh <JOBID> <OBS_PATH>
# Robustly block until SLURM job <JOBID> is terminal (per sacct, the accounting
# DB — NOT squeue/NFS, which are eventually-consistent and cause phantom reads),
# then apply an NFS flush grace and read <OBS_PATH> ONCE. Prints exactly one line:
#   DONE hr=<val> state=<S>      (valid numeric-headroom observation written)
#   FAILED state=<S>            (job terminal but no numeric-headroom observation)
#   TIMEOUT_WAIT                (still not terminal after the bound)
# Designed to be launched as a single background command by the driving agent;
# it blocks on the CLUSTER, so no phantom-read gate logic is needed agent-side.
jid="$1"; obs="$2"
[ -z "$jid" ] || [ -z "$obs" ] && { echo "USAGE iter_wait.sh JOBID OBS_PATH"; exit 2; }
term_re='^(COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)$'
prev=""
for i in $(seq 1 80); do   # ~80 * 30s = 40 min bound
  st=$(sacct -j "$jid" -n -X -o State 2>/dev/null | head -1 | tr -d ' ' | sed 's/+$//')
  if [[ "$st" =~ $term_re ]] && [ "$st" = "$prev" ]; then
    # two consecutive identical terminal reads -> trust it; grace for NFS flush
    sleep 12
    if [ -f "$obs" ]; then
      hr=$(python3 - "$obs" <<'PY' 2>/dev/null
import json,sys
try:
    h=json.load(open(sys.argv[1])).get("headroom")
    print(h if isinstance(h,(int,float)) else "NONUM")
except Exception:
    print("NONUM")
PY
)
      if [ -n "$hr" ] && [ "$hr" != "NONUM" ]; then echo "DONE hr=$hr state=$st"; exit 0; fi
    fi
    echo "FAILED state=$st"; exit 0
  fi
  prev="$st"
  sleep 30
done
echo "TIMEOUT_WAIT"
