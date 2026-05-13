# Cluster — Operator Guide for Humans and Agents (template)

This is a generic, hostname-free template for the cluster operating guide.
Copy it to `config/<your-site>_cluster_agent_guide.md`, fill in the
site-specific blanks, and keep that copy out of source control — the
`.gitignore` excludes anything matching `config/*_cluster_agent_guide.md`
except for this template.

Use this guide in two modes:

1. **As a human user.** First-time SSH-key setup and a mental model of how
   your Slurm cluster is laid out. Follow §§ 1 – 2.
2. **As an LLM agent driving the cluster on the user's behalf.** Safe
   probing, host-key handling, sandbox-aware authorization, Slurm
   operations. Follow §§ 3 – 6 after § 1 is done.

---

## 0. Cluster facts at a glance

Fill these in for your site. Treat anything dated more than ~6 months prior
as needing re-validation.

| | |
|---|---|
| DNS for head/login node | `<HEAD_NODE_HOSTNAME>` |
| Currently resolves to | `<HEAD_NODE_HOSTNAME_RESOLVED>` (do not hard-code) |
| Slurm version | `<SLURM_VERSION>` |
| Default partition | `<DEFAULT_PARTITION>` |
| Account convention | `<USERNAME_CONVENTION>` (e.g. institute IDM, SSO, ...) |
| User filesystem | `<USER_FS_ROOT>/<user>` (mounted on every compute node) |
| Avoid for heavy I/O | `<HOME_FS_ROOT>/<user>` — slows everyone down |
| Transient / fast scratch | `<SCRATCH_PATH>` (per-node, not shared) |
| Soft wall-time limit | `<WALL_TIME_LIMIT>` per job |
| GPU inventory (snapshot) | `<GPU_TYPES>` |

**You SSH into the head node.** The head node should have **no GPU**.
Anything GPU-related runs inside a Slurm job on a compute node.

---

## 1. SSH-key setup (the human user does this once)

Password-based SSH is discouraged. Set up a key once, use it forever. Do
this on the laptop you actually work from.

### 1.1 Generate a key (skip if you already have an Ed25519 key)

```bash
# Check what you already have
ls -la ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub 2>/dev/null

# If absent, generate one:
ssh-keygen -t ed25519 -C "$(whoami)@$(hostname) — cluster" -f ~/.ssh/id_ed25519
```

When prompted for a passphrase: **set one**. An empty passphrase means
that anyone who reads the key file owns your cluster account. macOS and
most Linux distributions integrate ssh-agent / Keychain so you type the
passphrase at most once per login.

Permissions matter; ssh refuses to use a too-open key:

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
```

### 1.2 Install the public key on the cluster

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub <user>@<HEAD_NODE_HOSTNAME>
```

`ssh-copy-id` will ask for your password once and append your public key
to `~/.ssh/authorized_keys` on the cluster. If `ssh-copy-id` is unavailable
(some macOS setups), the manual equivalent is:

```bash
cat ~/.ssh/id_ed25519.pub | ssh <user>@<HEAD_NODE_HOSTNAME> \
  'mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
```

### 1.3 Test that key auth works without a password prompt

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 <user>@<HEAD_NODE_HOSTNAME> \
  'echo ok host=$(hostname) user=$(whoami) cluster_dir=<USER_FS_ROOT>/$(whoami)'
```

`BatchMode=yes` disables every interactive prompt, so the command fails
fast if anything still wants a password.

If you get `Permission denied (publickey)`: re-run § 1.2 and confirm
`~/.ssh/authorized_keys` on the cluster has your public-key line.

If you get `Host key verification failed`: the cluster's host key changed
since you last connected. Confirm with your cluster admin that a rebuild
/ rekey actually happened, then run
`ssh-keygen -R <HEAD_NODE_HOSTNAME>` and try again. **Never silently
accept a changed host key.**

### 1.4 Recommended `~/.ssh/config` entry

```
Host cluster
    HostName <HEAD_NODE_HOSTNAME>
    User <your-username>
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 4
```

Then `ssh cluster` is enough; the alias is what you'll point your agent at.

`IdentitiesOnly yes` prevents ssh from offering every key in your agent
to the cluster, which avoids "too many authentication failures" on hosts
that don't recognise some other key.

---

## 2. Cluster mental model

```
your laptop  ──ssh──▶  head node (no GPU)
                       │
                       ├── <USER_FS_ROOT>/<user>/             ← put your repo here
                       │     (mounted on every node)
                       │
                       └── sbatch  ──▶  Slurm scheduler  ──▶  compute node
                                                              │
                                                              ├── 1–N GPUs
                                                              ├── <SCRATCH_PATH> (local SSD; transient)
                                                              └── runs your job
```

* Your repo lives at `<USER_FS_ROOT>/<user>/<project>` because that path is
  mounted on every compute node. Anything under `<HOME_FS_ROOT>/<user>` is
  also mounted but the home filesystem is shared and easily congested —
  put venvs and datasets on `<USER_FS_ROOT>`.
* `sbatch <script.sbatch>` enqueues a job; the scheduler picks a compute
  node that satisfies the requested resources.
* Standard out / standard error go to whatever paths the script's
  `#SBATCH --output=…` and `--error=…` directives name. **Those paths
  must exist before the job starts** (see § 4.3).
* Wall-time limits are enforced. Use checkpointing for long runs so the
  job can resume after pre-emption.

---

## 3. Agent operating manual

This section is written for an LLM agent (Claude, Codex, etc.) that an
authorised user has asked to operate the cluster on their behalf. The
human user has already finished § 1.

### 3.1 First contact: probe safely

Always start with a `BatchMode=yes` probe so missing auth fails fast
instead of hanging on a prompt:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 cluster \
  'hostname; whoami; ls -d <USER_FS_ROOT>/$(whoami) 2>/dev/null'
```

If this returns `Permission denied (publickey)`, stop and ask the user to
run § 1.2 / § 1.3. Do not try password auth. Do not try other keys.

### 3.2 Host-key changes — never bypass silently

If you ever see:

```
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
```

**Stop. Tell the user.** Possible causes are a legitimate rebuild / rekey,
or a man-in-the-middle. Only the user can disambiguate. The correct
response is:

1. Quote the warning to the user verbatim.
2. List the safe paths forward:
   * The user SSHs in interactively, accepts the new key after verifying
     with the cluster admin that a rekey happened, then asks the agent to
     retry.
   * Or the user explicitly authorises `ssh-keygen -R <host>` plus a
     re-add via `-o StrictHostKeyChecking=accept-new`.
3. Wait for the user's choice.

`-o StrictHostKeyChecking=no` is forbidden. It silently accepts any host
key and defeats the protection.

### 3.3 Authorization scope (sandbox boundaries)

The Anthropic agent sandbox treats remote shells onto shared infrastructure
as a sensitive scope. Read-only probes (a single `ssh … 'hostname'`) are
usually allowed; multi-line write/execute heredocs may require the user to
authorise specific commands. If the sandbox blocks an action, **do not
work around it**. Stop, list exactly which commands you intended to run,
and ask the user to authorise that specific set.

Always tell the user before:

* Cloning a repo onto `<USER_FS_ROOT>/<user>` (creates user-visible state).
* Building or modifying a venv on the cluster (writes lots of files).
* Submitting a job that costs more than a few GPU-minutes.
* Cancelling someone else's job (don't, ever; cancel only the user's own
  jobs and only with explicit authorisation).

### 3.4 Multi-line remote commands

Use a quoted heredoc so the local shell does not expand variables that
should be evaluated remotely:

```bash
ssh cluster bash -s <<'REMOTE'
set -euo pipefail
cd <USER_FS_ROOT>/<user>/<project>
git status -sb
git pull --ff-only
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
REMOTE
```

Note `<<'REMOTE'` (quoted) — without the quotes, `$(whoami)` would expand
on the laptop instead of on the cluster.

### 3.5 Reproducible probe checklist

Before kicking off a long run:

| Check | Command | Pass criterion |
|---|---|---|
| SSH works keyless | `ssh -o BatchMode=yes cluster 'true'` | exits 0 |
| Repo present | `ssh cluster 'ls -d <USER_FS_ROOT>/$(whoami)/<project>'` | path exists |
| Repo up to date | `ssh cluster 'cd … && git status -sb && git rev-parse HEAD'` | no diverged commits |
| Venv installs | `ssh cluster '… && source .venv/bin/activate && python -c "import torch"'` | no exception |
| Slurm reachable | `ssh cluster 'sinfo --version'` | prints version |
| Quota OK | `ssh cluster 'df -h <USER_FS_ROOT>/$(whoami)'` | not at 100 % |

CUDA cannot be verified on the head node — the head has no GPU. The
sbatch wrappers should `nvidia-smi` at job start so any CUDA problem
shows up in the first lines of `*.out`.

---

## 4. Slurm operations cheatsheet

### 4.1 Cluster overview

```bash
sinfo -h -o "%P %a %D %T %G"               # partition, state, GPU type per node
scontrol show node <node>                   # one node's full state
squeue -u $(whoami)                         # your queue
squeue --start -u $(whoami)                 # estimated start time
sacct -u $(whoami) -S "$(date -d '24 hours ago' '+%Y-%m-%dT%H:%M:%S')" \
      --format=JobID,JobName,State,ExitCode,Elapsed,NodeList -X    # 24-hour history
```

### 4.2 Submitting a job

```bash
sbatch path/to/job.sbatch
sbatch --export=ALL,MODEL=foo,CONFIG=configs/foo.yaml job.sbatch
sbatch --dependency=afterok:<jobid> job.sbatch    # chain after another job
```

Useful directives:

```bash
#SBATCH --job-name=ct_recon
#SBATCH --partition=<DEFAULT_PARTITION>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G                      # host RAM, *not* GPU memory
#SBATCH --gres=gpu:1                   # one GPU, any type
#SBATCH --gres=gpu:<TYPE>:1            # one specific GPU type
#SBATCH --time=04:00:00                # hard wall-time cap
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err
#SBATCH --exclude=<KNOWN_BAD_NODES>    # avoid known-bad nodes
```

### 4.3 The output-directory trap

Slurm redirects stdout / stderr **before** your script runs. If the
directories named in `--output=` and `--error=` do not exist on the
submit host at job-launch time, the job fails immediately with
`JobState=FAILED, Reason=NonZeroExitCode, RunTime=00:00:00`, and **no
`.out` or `.err` is written to disk**.

Two fixes, both safe to apply together:

1. Track the directory in your repo (e.g. via a `.gitkeep` file plus a
   `.gitignore` rule that allow-lists the directory but excludes `*.out`
   / `*.err`).
2. Inside the job script's `set -euo pipefail` body, also
   `mkdir -p $(dirname "$OUTPUT_DIR")`.

### 4.4 Wall-time limits and resume

The cluster enforces wall-time per job. For runs longer than the cap,
build a checkpoint-resume loop:

* The training script should checkpoint every N iterations and write a
  sentinel `.done` file when finished.
* The sbatch wrapper checks for the sentinel; if absent, it resubmits
  itself with `--dependency=afterok:$SLURM_JOB_ID` and a resubmit counter.
* `SIGTERM` and `SIGUSR1` from Slurm should trigger a clean checkpoint
  before exit.

### 4.5 Monitoring a running job

```bash
squeue -u $(whoami) --format="%.10i %.20j %.2t %.10M %.10L %.20R"
tail -f <USER_FS_ROOT>/$(whoami)/<project>/results/slurm/<jobname>-<jobid>.out
sstat -j <jobid> --format=JobID,AveCPU,AveRSS,MaxRSS,AveVMSize       # while running
```

Cancel one of your own jobs:

```bash
scancel <jobid>
scancel -u $(whoami) --name=<jobname>     # all your jobs with that name
```

---

## 5. GPU memory × workload table (illustrative)

Pick the GPU class that fits your model. If you request `--gres=gpu:1`
without a type pin, you'll get whatever is free — fine for small models,
surprising for a 90 GB FC matrix.

| GPU class | Memory | Typical use |
|---|---:|---|
| consumer 8 GB  | 8 GB | smoke tests only |
| consumer 11 GB | 11 GB | small models |
| consumer 12 GB | 12 GB | similar |
| professional 16 GB | 16 GB | mid-size 2D recon |
| professional 24 GB | 24 GB | larger 2D recon / small 3D |
| professional 48 GB | 48 GB | large 2D recon, moderate 3D |

When the single-GPU memory isn't enough, the right move is FSDP across
multiple smaller GPUs with CPU offload.

---

## 6. Common pitfalls

1. **CUDA index-URL mismatch.** `pip install torch` may pull a wheel built
   for a CUDA version newer than what the compute-node driver supports.
   Symptom: `torch.cuda.is_available()` returns `False` inside a Slurm
   job that *has* a GPU. Fix:

   ```bash
   pip uninstall -y torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

   Substitute `cu118`, `cu124`, etc. for older or newer drivers.

2. **Empty venv after wall-time pre-emption.** A `pip install` that gets
   `SIGTERM`'d mid-resolution leaves a partial venv that imports but
   breaks at first use. The fix is to delete `.venv/` and re-run setup.

3. **Heavy I/O on the home filesystem.** Putting datasets or venvs in
   `<HOME_FS_ROOT>/<user>` slows every cluster user down. Always put
   project working state on `<USER_FS_ROOT>/<user>`.

4. **Forgotten `--gres=gpu:N`.** A job without `--gres` runs on a compute
   node but with **zero** GPUs allocated. CUDA inside the job sees no
   devices. Always specify `--gres=gpu:1` (or more).

5. **`--mem` is host RAM, not GPU memory.** Don't conflate.

6. **`expandable_segments` for borderline OOM.** Set
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the sbatch to
   defragment the caching allocator. This often turns a borderline OOM
   into a successful run on a 24 GB card.

7. **Unowned files after sudo.** Don't `sudo` anything on the cluster.
   Your normal account has every permission you need under
   `<USER_FS_ROOT>/<user>`. Files owned by root in your tree are a mess
   to clean up.

8. **Two jobs on one node.** Slurm will pack two jobs onto a single node
   if each requests `--gres=gpu:1` and the node has ≥ 2 GPUs free. This
   is normal; the scheduler isolates the GPU visibility for each job via
   `CUDA_VISIBLE_DEVICES`.

9. **`squeue` empty after submission.** If `squeue -u <you>` returns
   nothing within a second of `sbatch`, the job either started and
   finished (check `sacct`) or failed at t=0 (see § 4.3). Run
   `scontrol show job <id>` to disambiguate.

10. **The head node is not a workhorse.** No GPU. Limited CPU. Don't run
    training, plotting, or large data conversions there. Use an
    interactive Slurm allocation: `srun --gres=gpu:1 --pty bash`.

---

## 7. Onboarding checklist for a new team member

- [ ] Generate or locate an Ed25519 key (§ 1.1).
- [ ] Install the public key on the cluster head node (§ 1.2).
- [ ] Test keyless SSH with `BatchMode=yes` (§ 1.3).
- [ ] Add the `Host cluster` block to `~/.ssh/config` (§ 1.4).
- [ ] `ssh cluster 'mkdir -p <USER_FS_ROOT>/$(whoami)'` to confirm the
      user filesystem is mounted.
- [ ] Read § 2 (mental model) and § 4.3 (the output-directory trap).
- [ ] Do a smoke test job: `srun --gres=gpu:1 --pty nvidia-smi`.

---

## Site-specific addendum

Append site-specific notes below this line in your copy:

- Hostnames, GPU inventory, partition names, wall-time policy.
- Quota / storage rules.
- Any in-house tooling.
- Account-rule deviations from § 1.

Keep that part **out of the public template**; it is the reason this file
is gitignored.
