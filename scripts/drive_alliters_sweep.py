#!/usr/bin/env python3
"""drive_alliters_sweep.py — cluster-side driver for the Mayo all-iters TEST sweep.

The QOS caps submitted array tasks at 100/user and running at 10/user, so the full
497-task array can't be submitted at once. This driver keeps the queue topped up to
CAP submitted, refilling as tasks finish. Idempotent + self-healing:

  * a manifest line is "done" once its final.json has all 5 patients
    (score_mayo_alliters.py --pending), so a driver restart just resumes;
  * lines already in the queue (squeue %K array indices) are never resubmitted;
  * a line that keeps failing without producing final.json is retried at most
    MAX_RETRY times, then given up on (logged) so it can't hog slots forever.

Run detached on the login node:
  cd /cluster/maier/Agent4CT && nohup python scripts/drive_alliters_sweep.py \
      > results/alliters_driver.log 2>&1 &
"""
import json
import os
import subprocess
import time

REPO = "/cluster/maier/Agent4CT"
MANIFEST = f"{REPO}/agentic_cfgs/alliters/manifest.tsv"
SBATCH = f"{REPO}/cluster/slurm/mayo_alliters_array.sbatch"
STATE = f"{REPO}/results/alliters_driver_state.json"
CAP = 50              # target; a submit-vs-squeue race can overshoot ~10, so 50 keeps ACTUAL <=~60 (user-requested ceiling)
MAX_RETRY = 3         # give up on a line after this many submits w/o final.json
SLEEP = 300


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, cwd=REPO,
                          capture_output=True, text=True).stdout.strip()


def main() -> None:
    state = {}
    if os.path.exists(STATE):
        try:
            state = json.load(open(STATE))
        except Exception:
            state = {}
    print(f"[driver] start {time.strftime('%F %T')}  CAP={CAP} MAX_RETRY={MAX_RETRY}",
          flush=True)
    while True:
        pend = [x for x in sh("python scripts/score_mayo_alliters.py --pending").split(",") if x]
        if not pend:
            print(f"[driver] ALL DONE {time.strftime('%F %T')}", flush=True)
            break
        queued = {x for x in sh('squeue -h -u maier -r -o "%K"').split() if x.isdigit()}
        gave_up = [x for x in pend if state.get(x, 0) >= MAX_RETRY and x not in queued]
        todo = [x for x in pend if x not in queued and state.get(x, 0) < MAX_RETRY]
        take = max(0, CAP - len(queued))
        chunk = todo[:take]
        if chunk:
            csv = ",".join(chunk)
            out = sh(f"sbatch --parsable --array={csv}%10 "
                     f"--export=ALL,MANIFEST={MANIFEST} {SBATCH}")
            for x in chunk:
                state[x] = state.get(x, 0) + 1
            json.dump(state, open(STATE, "w"))
            print(f"[driver] {time.strftime('%T')} queued={len(queued)} pending={len(pend)} "
                  f"submitted={len(chunk)} gave_up={len(gave_up)} -> {out}", flush=True)
        else:
            print(f"[driver] {time.strftime('%T')} queued={len(queued)} pending={len(pend)} "
                  f"gave_up={len(gave_up)} (no submit)", flush=True)
        if gave_up:
            print(f"[driver] GAVE UP (>= {MAX_RETRY} tries, no final.json): {gave_up}", flush=True)
        time.sleep(SLEEP)


if __name__ == "__main__":
    main()
