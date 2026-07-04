#!/bin/bash
# iter_wait.sh <JOBID> <OBS_PATH>   (v2 — settled-read gate, phantom-proof)
# On this NFS/SSH cluster, single reads of squeue/sacct/files flap. The ONLY
# reliable signal is a SETTLED read: the same numeric headroom parsed on 3
# consecutive reads ~3s apart. This gate declares DONE only when (a) the job is
# absent from squeue AND (b) OBS_PATH settled-parses to one numeric headroom.
# Prints exactly one line: DONE hr=<val> | FAILED state=<S> | TIMEOUT_WAIT
jid="$1"; obs="$2"
[ -z "$jid" ] || [ -z "$obs" ] && { echo "USAGE iter_wait.sh JOBID OBS_PATH"; exit 2; }
rd() { python3 -c "import json,sys
try:
 h=json.load(open('$obs')).get('headroom'); print(h if isinstance(h,(int,float)) else '')
except Exception: print('')" 2>/dev/null; }
settled_hr() { local a b c; a=$(rd); sleep 3; b=$(rd); sleep 3; c=$(rd);
  if [ -n "$a" ] && [ "$a" = "$b" ] && [ "$b" = "$c" ]; then echo "$a"; fi; }
term_re='^(COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)$'
for i in $(seq 1 90); do   # ~90 * ~28s = ~42 min bound
  inq=$(squeue -h -j "$jid" -o %t 2>/dev/null | tr -d ' ')
  if [ -z "$inq" ]; then
    hr=$(settled_hr)                         # 3-read settled gate (the reliable one)
    if [ -n "$hr" ]; then echo "DONE hr=$hr"; exit 0; fi
    st=$(sacct -j "$jid" -n -X -o State 2>/dev/null | head -1 | tr -d ' ' | sed 's/+$//')
    if [[ "$st" =~ $term_re ]]; then
      sleep 8; hr=$(settled_hr)              # one more settle chance after flush
      if [ -n "$hr" ]; then echo "DONE hr=$hr"; exit 0; fi
      echo "FAILED state=$st"; exit 0        # terminal but no settled obs = real failure
    fi
  fi
  sleep 22
done
echo "TIMEOUT_WAIT"
