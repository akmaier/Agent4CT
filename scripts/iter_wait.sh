#!/bin/bash
# iter_wait.sh <JOBID> <ITERDIR>   (v3 — per-JOBID sentinel, stale-proof)
# The driver writes+fsyncs <ITERDIR>/DONE.<JOBID> as its LAST step. A file named
# after the CURRENT job cannot be a prior attempt's NFS-cached artifact, so this
# gate is immune to the stale-path phantoms that defeated v1/v2.
# Prints exactly one line: DONE hr=<val> | FAILED state=<S> | TIMEOUT_WAIT
jid="$1"; iterdir="$2"
[ -z "$jid" ] || [ -z "$iterdir" ] && { echo "USAGE iter_wait.sh JOBID ITERDIR"; exit 2; }
sent="$iterdir/DONE.$jid"
term_re='^(COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)$'
for i in $(seq 1 100); do   # ~100 * ~20s = ~33 min bound
  if [ -f "$sent" ]; then
    hr=$(cat "$sent" 2>/dev/null | tr -d ' \n')
    [ -n "$hr" ] && { echo "DONE hr=$hr"; exit 0; }
  fi
  inq=$(squeue -h -j "$jid" -o %t 2>/dev/null | tr -d ' ')
  if [ -z "$inq" ]; then
    st=$(sacct -j "$jid" -n -X -o State 2>/dev/null | head -1 | tr -d ' ' | sed 's/+$//')
    if [[ "$st" =~ $term_re ]]; then
      sleep 10                       # let NFS surface the sentinel
      if [ -f "$sent" ]; then hr=$(cat "$sent" 2>/dev/null | tr -d ' \n'); [ -n "$hr" ] && { echo "DONE hr=$hr"; exit 0; }; fi
      echo "FAILED state=$st"; exit 0
    fi
  fi
  sleep 20
done
echo "TIMEOUT_WAIT"
