"""RRZE autonomous DL-Sparse-View agent (family-agnostic).

Drives one autoresearch loop against the FAU/RRZE LLM gateway with a
chosen model family. The currently-wired families and their default
RRZE model IDs (overridable from llm_api.toml):

    kimi      moonshotai/Kimi-K2.6
    deepseek  deepseek-ai/DeepSeek-V284B4-Flash
    mistral   mistralai/Mistral-Medium-3.5-128B
    gptoss    openai/gpt-oss-120b

Each family writes its own slot:
    pentathlon/dl_sparse_view_<family>/solver.py
    cluster/slurm/dl_sparse_view_<family>_5min.sbatch

…and uses its own slug (``dl-sparse-view-<family>-YYYYMMDD-NN``). The
ISOLATION contract is identical regardless of family: every agent only
sees its own slug under ``docs/runs/``; the cross-run scratchpad
(``docs/runs/observations.jsonl``) and any other agent's slug
directories are refused at the tool layer. "Advice for others" therefore
only loops back to the agent that wrote it.

Credentials + endpoint live in the gitignored ``config/llm_api.toml``
(see ``config/llm_api.example.toml`` for the shape). The operator's
existing per-project layout uses ``[llm]``:

    [llm]
    provider = "nhr_fau"
    base_url = "https://hub.nhr.fau.de/api/llmgw/v1"
    api_key  = "..."

    # OPTIONAL — overrides the FAMILIES defaults below per family:
    [llm.models]
    kimi     = "moonshotai/Kimi-K2.6"
    deepseek = "deepseek-ai/DeepSeek-V284B4-Flash"
    mistral  = "mistralai/Mistral-Medium-3.5-128B"
    gptoss   = "openai/gpt-oss-120b"

If the ``[llm.models]`` sub-table is absent (the operator's current
file), each family falls back to the built-in default in ``FAMILIES``
below — same model ids as the comment table above.

Usage:

    .venv/bin/python cluster/rrze_agent.py \\
        --family kimi \\
        --slug   dl-sparse-view-kimi-20260514-01 \\
        --iters  5
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

try:
    from openai import OpenAI
    _OPENAI_OK = True
except ImportError:
    OpenAI = None  # type: ignore
    _OPENAI_OK = False

try:
    import tomllib  # py>=3.11
    _TOML_OK = True
except ImportError:
    _TOML_OK = False


# --- family registry ----------------------------------------------------

# Each family pins a *default* RRZE model. The [rrze.models] table in
# llm_api.toml (or `--model` on the CLI) wins when present.
FAMILIES: dict[str, dict[str, str]] = {
    # Model ids match what the FAU gateway lists in GET /models.
    "kimi":     {"model": "moonshotai/Kimi-K2.6"},
    "deepseek": {"model": "deepseek-ai/DeepSeek-V4-Flash"},
    "mistral":  {"model": "mistralai/Mistral-Medium-3.5-128B"},
    "gptoss":   {"model": "gpt-oss-120b"},
}

def family_solver_dir(family: str) -> str:
    return f"pentathlon/dl_sparse_view_{family}/"

def family_sbatch(family: str) -> str:
    return f"cluster/slurm/dl_sparse_view_{family}_5min.sbatch"


# --- credentials --------------------------------------------------------

def load_credentials(toml_path: Path, family: str) -> dict[str, str]:
    """Read api_key + base_url + family-specific model from llm_api.toml.

    Schemas supported (the operator's existing layout is `[llm]`):

        [llm]
        provider = "nhr_fau"
        base_url = "https://hub.nhr.fau.de/api/llmgw/v1"
        api_key  = "..."
        # optional, takes precedence over the FAMILIES built-in defaults:
        [llm.models]
        kimi     = "moonshotai/Kimi-K2.6"
        deepseek = "deepseek-ai/DeepSeek-V284B4-Flash"
        ...

    Or the older `[rrze]` table (still accepted for back-compat).

    The file is gitignored on purpose. Falls back to env-vars for any
    missing piece:
        LLMAPI_KEY   -> api_key
        LLM_BASE_URL -> base_url   (default: FAU gateway)
        LLM_MODEL    -> model      (default: per-family default above)
    """
    llm: dict[str, Any] = {}
    rrze: dict[str, Any] = {}
    if toml_path.exists():
        if not _TOML_OK:
            raise RuntimeError(
                "Python >= 3.11 required to parse llm_api.toml")
        with toml_path.open("rb") as f:
            data = tomllib.load(f)
        llm = dict(data.get("llm") or {})
        rrze = dict(data.get("rrze") or {})
    # api_key + base_url precedence: [llm] -> [rrze] -> env -> FAU default.
    api_key = ((llm.get("api_key") or rrze.get("api_key")
                or os.environ.get("LLMAPI_KEY", "")) or "").strip()
    base_url = (llm.get("base_url") or rrze.get("base_url")
                or os.environ.get("LLM_BASE_URL")
                or "https://hub.nhr.fau.de/api/llmgw/v1")
    # Per-family model precedence: env > [llm.models][fam] > [rrze.models][fam]
    # > FAMILIES default. We intentionally do NOT fall back to a flat
    # [llm].model / [rrze].model — that's a single-model convention that
    # would override the per-family routing for multi-agent runs.
    llm_models = dict(llm.get("models") or {})
    rrze_models = dict(rrze.get("models") or {})
    model = (os.environ.get("LLM_MODEL")
             or llm_models.get(family)
             or rrze_models.get(family)
             or FAMILIES[family]["model"])
    return {"api_key": api_key, "base_url": base_url, "model": model}


# --- isolation rules ----------------------------------------------------

def _allowed_read(path: Path, own_slug: str, family: str) -> bool:
    try:
        rel = path.resolve().relative_to(REPO)
    except ValueError:
        return False
    s = str(rel).replace(os.sep, "/")
    own_run_dir = f"docs/runs/{own_slug}"
    # Explicitly blocked: cross-agent scratchpad + every other slug dir.
    if s == "docs/runs/observations.jsonl":
        return False
    if s.startswith("docs/runs/") and not s.startswith(own_run_dir):
        return False
    for prefix in (
        family_solver_dir(family),
        "pentathlon/dl_sparse_view/program.md",
        "ddssl_ldct/",
        "literature/",
        family_sbatch(family),
        f"{own_run_dir}/",
        "scripts/agent4ct_record.py",
    ):
        if s == prefix.rstrip("/") or s.startswith(prefix):
            return True
    return False


def _allowed_write(path: Path, family: str) -> bool:
    try:
        rel = path.resolve().relative_to(REPO)
    except ValueError:
        return False
    s = str(rel).replace(os.sep, "/")
    return s == f"pentathlon/dl_sparse_view_{family}/solver.py"


# --- tool implementations ----------------------------------------------

def tool_read_file(args: dict, ctx: dict) -> str:
    p = REPO / args["path"]
    if not _allowed_read(p, ctx["slug"], ctx["family"]):
        return (f"REFUSED: {args['path']} is outside the agent's allow-list. "
                f"The cross-agent observations.jsonl and other agents' slug "
                f"directories are off-limits by design.")
    try:
        return p.read_text()[: args.get("max_bytes", 200_000)]
    except FileNotFoundError:
        return f"NOT FOUND: {args['path']}"


def tool_write_file(args: dict, ctx: dict) -> str:
    p = REPO / args["path"]
    if not _allowed_write(p, ctx["family"]):
        return (f"REFUSED: {args['path']} is read-only for this agent. "
                f"Only {family_solver_dir(ctx['family'])}solver.py is editable.")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(args["content"])
    return f"OK ({len(args['content'])} bytes -> {args['path']})"


# --- bounded pipeline tools (NO arbitrary shell) -----------------------

CLUSTER_HOST = "maier@cluster.i5.informatik.uni-erlangen.de"
CLUSTER_REPO = "/cluster/maier/Agent4CT"


# Static safety scan on the agent-written solver.py before we scp +
# sbatch it. A solver only needs torch / numpy / matplotlib + the
# ddssl_ldct package; anything that spawns processes, opens sockets,
# loads untrusted pickles, or runs strings as code is refused.
# Defense-in-depth on top of the bounded tools — an external-LLM agent
# can write almost any python it wants into solver.py, but the cluster
# executes that code, so we filter at the shipping boundary.
_DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"\bsubprocess\b",                       "subprocess module"),
    (r"\bos\.(system|popen|exec[lv]?p?e?|spawn[lv]?p?e?)\b",
                                              "os shell/process spawn"),
    (r"\bsocket\b",                           "socket module"),
    (r"\b(urllib|urllib2|urllib3|requests|httpx|http\.client)\b",
                                              "HTTP client"),
    # The bare builtins eval / exec / compile / __import__ — the
    # (?<!\.) lookbehind excludes attribute calls like pipe.eval(),
    # model.eval(), torch.compile(), tensor.exec_strategy() etc., which
    # are legitimate PyTorch / numpy patterns.
    (r"(?<!\.)\beval\s*\(",                   "eval() builtin"),
    (r"(?<!\.)\bexec\s*\(",                   "exec() builtin"),
    (r"(?<!\.)\bcompile\s*\(",                "compile() builtin"),
    (r"\b__import__\s*\(",                    "__import__()"),
    (r"\bpickle\.loads?\s*\(",                "pickle.load(s)"),
    (r"\bshutil\.rmtree\s*\(",                "shutil.rmtree"),
    (r"\bos\.remove\s*\(",                    "os.remove"),
    (r"\bos\.unlink\s*\(",                    "os.unlink"),
]


def _scan_solver_for_dangerous_patterns(solver_path: Path) -> list[str]:
    """Return a list of "line N: <label> (<match>)" strings for each hit.

    Naive regex over the whole file — won't strip comments / docstrings.
    False positives are fine; the agent can rewrite and resubmit. False
    negatives are what we want to avoid.
    """
    try:
        content = solver_path.read_text()
    except OSError as e:
        return [f"could not read solver.py: {e}"]
    findings: list[str] = []
    for pat, label in _DANGEROUS_PATTERNS:
        for m in re.finditer(pat, content):
            line_no = content[: m.start()].count("\n") + 1
            snippet = m.group(0)
            findings.append(f"line {line_no}: {label} ({snippet!r})")
    return findings


def _ssh(cmd_on_cluster: str, *, timeout_s: int = 60) -> subprocess.CompletedProcess:
    """One ssh round-trip with bounded timeout."""
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
         CLUSTER_HOST, cmd_on_cluster],
        capture_output=True, text=True, timeout=timeout_s,
    )


def _scp(src: str, dst: str, *, timeout_s: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["scp", "-q", "-o", "BatchMode=yes", src, dst],
        capture_output=True, text=True, timeout=timeout_s,
    )


JOB_STATE_SUBMITTED = "submitted"        # sbatch returned, waiting
JOB_STATE_MUST_RECORD = "must_record"    # finished (ok or fail), agent owes a record
JOB_STATE_RECORDED = "recorded"          # iter written to journal


def _outstanding_record(ctx: dict) -> Optional[str]:
    """Return the first job_id whose iteration must be recorded before the
    agent is allowed to submit anything new. None if the pipeline is clean.
    """
    for jid, state in ctx.get("pending", {}).items():
        if state == JOB_STATE_MUST_RECORD:
            return jid
    return None


def _has_in_flight(ctx: dict) -> Optional[str]:
    """Return the first job_id that's still submitted-but-not-polled-to-end."""
    for jid, state in ctx.get("pending", {}).items():
        if state == JOB_STATE_SUBMITTED:
            return jid
    return None


def tool_submit_iteration(args: dict, ctx: dict) -> str:
    """Push solver.py to the cluster, submit the family's sbatch, return job id.

    INVARIANT (record-or-die): before submitting a NEW iteration the
    agent must have called record_iteration for any previous submission
    whose poll has surfaced a terminal state (result.json present, OR
    Slurm job state COMPLETED/FAILED/TIMEOUT/CANCELLED without one).
    The agent can't just keep submitting until something works — every
    completed iteration, kept or discarded, gets a journal entry.
    """
    fam = ctx["family"]
    # --- record-or-die gate ------------------------------------------
    blocker = _outstanding_record(ctx)
    if blocker:
        return (f"REFUSED: cannot submit a new iteration while job "
                f"{blocker} is still un-recorded. The record_iteration "
                f"tool MUST be called first with kept=true/false, a "
                f"rationale, and the metrics from /tmp/result-{fam}-"
                f"{blocker}.json (if it exists) — or kept=false + a "
                f"rationale describing why the job failed (if it does "
                f"not). Once recorded, you can submit again.")
    in_flight = _has_in_flight(ctx)
    if in_flight:
        return (f"REFUSED: job {in_flight} is still running / queued. "
                f"Call poll_iteration(job_id=\"{in_flight}\") until it "
                f"reports terminal state, then record_iteration, then "
                f"submit a new one.")
    local_solver = REPO / f"pentathlon/dl_sparse_view_{fam}/solver.py"
    if not local_solver.exists():
        return f"ERROR: {local_solver.relative_to(REPO)} does not exist; write it first."
    findings = _scan_solver_for_dangerous_patterns(local_solver)
    if findings:
        return ("REFUSED: solver.py contains patterns the safety scan "
                "refuses to ship to the cluster. Rewrite solver.py using "
                "pure torch / numpy / matplotlib + the ddssl_ldct package "
                "(no subprocess, no os.system, no sockets / HTTP, no "
                "eval / exec / __import__, no pickle.load, no shutil.rmtree, "
                "no file deletion).\n  findings:\n"
                + "\n".join(f"    - {f}" for f in findings))
    remote_solver = f"{CLUSTER_REPO}/pentathlon/dl_sparse_view_{fam}/solver.py"
    p = _scp(str(local_solver), f"{CLUSTER_HOST}:{remote_solver}")
    if p.returncode != 0:
        return f"scp FAILED: {p.stderr[:600]}"
    sbatch_rel = f"cluster/slurm/dl_sparse_view_{fam}_5min.sbatch"
    p = _ssh(f"cd {CLUSTER_REPO} && sbatch {sbatch_rel}")
    out = (p.stdout + p.stderr).strip()
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        return f"sbatch FAILED: rc={p.returncode}\n{out[:600]}"
    jid = m.group(1)
    ctx.setdefault("pending", {})[jid] = JOB_STATE_SUBMITTED
    return (f"OK submitted job {jid}. Poll it with "
            f"poll_iteration(job_id=\"{jid}\") until it reports a "
            f"terminal state, then call record_iteration with "
            f"job_id=\"{jid}\". You cannot submit another job until "
            f"this one is recorded.\n{out}")


def tool_poll_iteration(args: dict, ctx: dict) -> str:
    """Check the Slurm job state and pull result.json when present.

    Three terminal outcomes the agent must distinguish:
      1. result.json present  → STATUS=OK     (model + metrics ready)
      2. job in {FAILED, TIMEOUT, CANCELLED, OUT_OF_MEMORY, NODE_FAIL,
         COMPLETED} *without* result.json    → STATUS=FAILED
      3. job still RUNNING / PENDING         → STATUS=WAITING

    On (1) or (2) the agent MUST call record_iteration before submitting
    another job — the tool layer will refuse the next submit until then.
    """
    fam = ctx["family"]
    jid = str(args["job_id"])
    if not jid.isdigit():
        return "ERROR: job_id must be numeric"
    remote_out = f"{CLUSTER_REPO}/runs/iter-{fam}-{jid}"
    # One round-trip pulls sacct State + result-existence + slurm-stdout tail
    # + slurm-stderr tail. Cheaper than multiple SSHs.
    probe = _ssh(
        f"echo '--SACCT--'; "
        f"sacct -j {jid} -X --noheader --format=State -P 2>/dev/null | head -1; "
        f"echo '--RESULT--'; "
        f"test -f {remote_out}/result.json && echo HAVE_RESULT || echo NO_RESULT; "
        f"echo '--STDOUT_TAIL--'; "
        f"tail -25 {CLUSTER_REPO}/results/slurm/ddssl-dlsv-{fam}-{jid}.out 2>/dev/null; "
        f"echo '--STDERR_TAIL--'; "
        f"tail -15 {CLUSTER_REPO}/results/slurm/ddssl-dlsv-{fam}-{jid}.err 2>/dev/null"
    )
    out = probe.stdout
    # Parse sacct State (between --SACCT-- and --RESULT--).
    state = "UNKNOWN"
    try:
        chunk = out.split("--SACCT--", 1)[1].split("--RESULT--", 1)[0].strip()
        if chunk:
            state = chunk.splitlines()[0].strip().split()[0]
    except Exception:
        pass
    have_result = "HAVE_RESULT" in out

    TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED",
                "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}

    pending = ctx.setdefault("pending", {})

    if have_result:
        # Success path: pull artifacts to /tmp/ and surface the body inline.
        local_result = f"/tmp/result-{fam}-{jid}.json"
        local_image  = f"/tmp/comparison-{fam}-{jid}.png"
        _scp(f"{CLUSTER_HOST}:{remote_out}/result.json", local_result)
        _scp(f"{CLUSTER_HOST}:{remote_out}/comparison.png", local_image)
        try:
            body = Path(local_result).read_text()
        except FileNotFoundError:
            return f"result.json reported present but scp failed:\n{out[-1500:]}"
        pending[jid] = JOB_STATE_MUST_RECORD
        return (f"STATUS=OK  job {jid} complete  (sacct State={state})\n"
                f"result.json at {local_result}\n"
                f"comparison.png at {local_image}\n"
                f"NEXT: call record_iteration(job_id=\"{jid}\", iter=N, "
                f"val_score=..., headroom=..., params_M=..., train_n=..., "
                f"change_class=..., rationale='...', advice='...', kept=...). "
                f"The tool layer will REFUSE submit_iteration until you "
                f"record this one.\n"
                f"--- result.json ---\n{body[:4000]}")
    if state in TERMINAL:
        pending[jid] = JOB_STATE_MUST_RECORD
        return (f"STATUS=FAILED  job {jid} terminated (sacct State={state}) "
                f"WITHOUT result.json. The solver crashed or timed out.\n"
                f"NEXT: call record_iteration(job_id=\"{jid}\", iter=N, "
                f"val_score=0, headroom=0, kept=false, "
                f"change_class='architecture' (or whichever knob), "
                f"rationale='describe the failure from the stdout/stderr "
                f"tail below', advice='generalisable lesson'). "
                f"The tool layer will REFUSE submit_iteration until you "
                f"record this failure.\n"
                f"--- stdout/stderr tail ---\n{out[-3000:]}")
    # still running / queued
    return (f"STATUS=WAITING  sacct State={state}. The job has not "
            f"reached a terminal state yet — re-poll in ~30 seconds.\n"
            f"--- progress tail ---\n{out[-2500:]}")


def tool_record_iteration(args: dict, ctx: dict) -> str:
    """Append the iteration to the agent's journal. Required after every
    terminal poll (success or failure) before the agent can submit the
    next iteration. The job_id arg connects the record to a previously-
    submitted job so the tool layer can clear the must-record flag.
    """
    fam = ctx["family"]
    slug = ctx["slug"]
    iter_n = int(args["iter"])
    val_score = float(args["val_score"])
    headroom = float(args["headroom"])
    params_M = float(args.get("params_M") or 0)
    train_n = int(args.get("train_n") or 0)
    change_class = str(args.get("change_class", "other"))
    rationale = str(args["rationale"])[:2000]
    advice = str(args.get("advice", ""))[:600]
    kept = bool(args["kept"])
    jid = str(args.get("job_id", "0"))
    comparison = f"/tmp/comparison-{fam}-{jid}.png"
    if not Path(comparison).exists():
        comparison = ""
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, cwd=str(REPO)
    ).stdout.strip()
    cmd = [
        str(REPO / ".venv" / "bin" / "python"),
        str(REPO / "scripts" / "agent4ct_record.py"), "record",
        "--slug", slug, "--iter", str(iter_n),
        "--val-score", f"{val_score:.6f}",
        "--headroom",  f"{headroom:.6f}",
        "--params-M",  f"{params_M:.6f}",
        "--train-n",   str(train_n),
        "--change-class", change_class,
        "--rationale", rationale,
        "--advice", advice,
        "--kept", "true" if kept else "false",
        "--commit", commit,
        "--solver", f"pentathlon/dl_sparse_view_{fam}/solver.py",
    ]
    if not kept and not val_score and not headroom:
        # crash/timeout — pass the status so it gets the right badge
        cmd += ["--status", "crash"]
    if comparison:
        cmd += ["--comparison", comparison]
    env = {**os.environ, "AGENT4CT_AGENT": fam, "AGENT4CT_MODEL": ctx["model"]}
    p = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=str(REPO), env=env, timeout=120)
    out = p.stdout + (("\nSTDERR:\n" + p.stderr) if p.stderr else "")
    if p.returncode == 0 and jid in ctx.get("pending", {}):
        ctx["pending"][jid] = JOB_STATE_RECORDED
    return (f"exit={p.returncode}\n{out[-3000:]}\n"
            f"(pending after this call: "
            f"{[jid for jid, st in ctx.get('pending', {}).items() if st != JOB_STATE_RECORDED]})")


TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file under the project root. Refuses paths outside the allow-list (cross-agent scratchpad, other slugs).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Repo-relative path."},
            "max_bytes": {"type": "integer", "default": 200000},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Overwrite the agent's editable solver file. Refuses anything else.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        }, "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "submit_iteration",
        "description": "Push pentathlon/dl_sparse_view_<family>/solver.py to the cluster and run cluster/slurm/dl_sparse_view_<family>_5min.sbatch. Returns the Slurm job id.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "poll_iteration",
        "description": "Check whether a Slurm job has produced result.json. When it has, scp result.json + comparison.png to /tmp/ and return the result.json body inline.",
        "parameters": {"type": "object", "properties": {
            "job_id": {"type": "string", "description": "Numeric Slurm job id returned by submit_iteration."},
        }, "required": ["job_id"]},
    }},
    {"type": "function", "function": {
        "name": "record_iteration",
        "description": "Append the iteration to the agent's journal (results.tsv + observation.json + scratchpad) and commit + push. Always records under the agent's own slug.",
        "parameters": {"type": "object", "properties": {
            "iter":          {"type": "integer"},
            "job_id":        {"type": "string"},
            "val_score":     {"type": "number"},
            "headroom":      {"type": "number"},
            "params_M":      {"type": "number"},
            "train_n":       {"type": "integer"},
            "change_class":  {"type": "string",
                              "enum": ["architecture","optimizer","loss","augmentation","other"]},
            "rationale":     {"type": "string"},
            "advice":        {"type": "string"},
            "kept":          {"type": "boolean"},
        }, "required": ["iter","val_score","headroom","change_class","rationale","kept"]},
    }},
]

TOOL_DISPATCH = {
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "submit_iteration": tool_submit_iteration,
    "poll_iteration":   tool_poll_iteration,
    "record_iteration": tool_record_iteration,
}


# --- agent loop ---------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """You are an autonomous coding agent on the
Agent4CT DL-Sparse-View challenge. Your model family is `{family}`
(model id `{model}`). Your run slug is `{slug}`.

GOAL: minimise the validation RMSE of a 2D sparse-view fan-beam CT
reconstruction by editing `{solver_dir}solver.py` and running short
Slurm iterations on the FAU/i5 cluster. Headroom is in [0,1]; higher
is better; `1.0` means RMSE=0 vs. ground truth.

ISOLATION RULES (hard contract, enforced at the tool level):
  * You may read: {solver_dir}, pentathlon/dl_sparse_view/program.md,
    ddssl_ldct/, literature/, {sbatch}, and ONLY your own slug's
    docs/runs/ directory (so you can re-read your own past observations
    — your previous "Advice for others" loops back to you only).
  * You MUST NOT read docs/runs/observations.jsonl or any other slug
    under docs/runs/. The read_file tool will refuse these paths.
  * You may only WRITE {solver_dir}solver.py.

AVAILABLE TOOLS (bounded — there is NO arbitrary shell):
  * read_file(path, max_bytes)      — read repo file from the allow-list.
  * write_file(path, content)       — overwrite ONLY your solver.py.
  * submit_iteration()              — scp your solver.py to the cluster
                                      and submit the family's sbatch.
                                      Returns Slurm job id.
  * poll_iteration(job_id)          — check Slurm state. Returns one of
                                      three STATUS lines:
                                        STATUS=WAITING — keep polling
                                        STATUS=OK      — result.json
                                                         fetched; record now
                                        STATUS=FAILED  — terminal crash/
                                                         timeout; record
                                                         now with kept=false
  * record_iteration(...)           — append the iteration to your
                                      journal + commit + push. REQUIRED
                                      after every terminal poll (OK or
                                      FAILED). The slug + agent + model
                                      labels are pinned to you.

RECORD-OR-DIE INVARIANT (enforced at the tool layer):
After a poll returns STATUS=OK or STATUS=FAILED, the *next*
submit_iteration() will be REFUSED until you call record_iteration with
the matching job_id. There is no way around this — your single output
to the world is the journal, so every job (kept, discarded, or crashed)
gets one record_iteration call before the next submit. For crashes,
just pass val_score=0 headroom=0 kept=false and a rationale that
describes what failed (from the stdout/stderr tail the poll returned).

WORKFLOW per iteration:
  1. read_file pentathlon/dl_sparse_view/program.md + (iter > 1) your
     own previous observation.json under docs/runs/{slug}/iterations/.
  2. Decide ONE change to solver.py (architecture | optimizer | loss |
     augmentation | other). DO NOT change the geometry (image_size=512,
     n_angles=128, n_det=736, etc.) or train_n/val_n (400/100).
  3. write_file {solver_dir}solver.py with the new content. The file
     must keep the build_geometry/build_dataset/main signatures used by
     the sbatch — the cluster runs `python {solver_dir}solver.py <out_dir>`.
  4. submit_iteration() → returns Slurm job id.
  5. poll_iteration(job_id) every ~30 s until STATUS=OK or STATUS=FAILED.
  6. record_iteration(job_id=..., iter=N, val_score=..., headroom=...,
                      params_M=..., train_n=..., change_class=...,
                      rationale='...', advice='...', kept=...).
     Required before submitting again (record-or-die).
  7. Loop steps 2-6 for the next iteration.

Iteration budget: {iters} iterations, ~60 tool calls each. After the
last successful record, output the text 'DONE'.
"""


def run_agent(slug: str, family: str, n_iters: int, *,
              model: str, base_url: str, api_key: str) -> int:
    if not _OPENAI_OK:
        print("ERROR: `pip install openai` first.", file=sys.stderr)
        return 2
    client = OpenAI(base_url=base_url, api_key=api_key)
    solver_dir = family_solver_dir(family)
    sbatch = family_sbatch(family)
    system = SYSTEM_PROMPT_TEMPLATE.format(
        family=family, model=model, slug=slug,
        solver_dir=solver_dir, sbatch=sbatch, iters=n_iters,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content":
         f"Begin iteration loop. Plan and run up to {n_iters} iterations. "
         f"After each iteration, summarise val_score + headroom + delta. "
         f"After the last one, write 'DONE'."},
    ]
    ctx = {"slug": slug, "family": family, "model": model, "pending": {}}
    # Bump per-iter budget. The prior bound (30) was eaten by polling
    # loops; with the record-or-die invariant gating the next submit,
    # the model can't burn a whole iter on polling without recording.
    max_steps = 60 * n_iters
    for step in range(max_steps):
        print(f"\n=== [{family}] step {step+1}/{max_steps} ===", flush=True)
        # Retry the chat-completion call on transient 5xx / connection
        # errors. The FAU gateway sits behind litellm + per-model
        # backends (vLLM hosts) that occasionally hiccup; one 500
        # shouldn't take down a 5-iteration agent. Up to 6 retries with
        # exponential backoff (2 -> 4 -> 8 -> 16 -> 32 -> 60 s, capped).
        # If a model's backend is genuinely dead, the loop still gives
        # up after ~2 min so the agent doesn't hang the whole night.
        resp = None
        last_err: Optional[Exception] = None
        for attempt in range(6):
            try:
                resp = client.chat.completions.create(
                    model=model, messages=messages, tools=TOOLS_SCHEMA,
                    tool_choice="auto", temperature=0.2,
                )
                break
            except Exception as e:  # InternalServerError, APIConnectionError, RateLimitError, ...
                last_err = e
                # Surface so the operator sees what's happening in the log.
                emsg = str(e)[:200]
                wait = min(60, 2 ** (attempt + 1))
                print(f"[{family}] chat.completions FAILED (attempt {attempt+1}/6): "
                      f"{type(e).__name__}: {emsg}\n[{family}] retrying in {wait}s",
                      flush=True)
                time.sleep(wait)
        if resp is None:
            print(f"[{family}] giving up after 6 retries; last error: {last_err}",
                  flush=True)
            # Try to record any outstanding work as a crash before exiting,
            # so the journal still tells the truth.
            for jid, st in list(ctx.get("pending", {}).items()):
                if st != JOB_STATE_RECORDED:
                    print(f"[{family}] orphaned job {jid} state={st} "
                          f"will not get a record (agent died).", flush=True)
            return 1
        msg = resp.choices[0].message
        content = msg.content or ""
        tool_calls = getattr(msg, "tool_calls", None) or []
        if content:
            print(f"[{family}] {content[:1000]}", flush=True)
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in tool_calls
            ] if tool_calls else None,
        })
        if not tool_calls:
            if "DONE" in content.upper():
                # The invariant: don't accept DONE with un-recorded work.
                unfinished = [j for j, st in ctx.get("pending", {}).items()
                              if st != JOB_STATE_RECORDED]
                if unfinished:
                    messages.append({"role": "user", "content":
                        (f"REFUSED 'DONE': you still have un-recorded "
                         f"submissions: {unfinished}. For each one, "
                         f"poll_iteration to terminal STATUS, then "
                         f"record_iteration. Then say DONE.")})
                    continue
                print(f"[{family}] agent signalled DONE.", flush=True)
                return 0
            messages.append({"role": "user",
                             "content": "Continue, or write DONE if finished."})
            continue
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            print(f"[{family}] tool {name}({json.dumps(args)[:300]})", flush=True)
            fn = TOOL_DISPATCH.get(name)
            if fn is None:
                result = f"UNKNOWN TOOL: {name}"
            else:
                try:
                    result = fn(args, ctx)
                except Exception as e:
                    result = f"ERROR: {type(e).__name__}: {e}"
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result[:8000],
            })
    print(f"[{family}] step budget exhausted.", flush=True)
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", required=True, choices=sorted(FAMILIES.keys()),
                   help="Which model family to drive.")
    p.add_argument("--slug", required=True,
                   help="The agent's own run slug (must already be created).")
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--model", default=None,
                   help="Override the model id from llm_api.toml.")
    p.add_argument("--base-url", default=None)
    p.add_argument("--credentials", default=str(REPO / "config" / "llm_api.toml"),
                   help="Path to llm_api.toml (gitignored). Default: <repo>/config/llm_api.toml.")
    args = p.parse_args()

    creds = load_credentials(Path(args.credentials), args.family)
    if args.model:    creds["model"] = args.model
    if args.base_url: creds["base_url"] = args.base_url
    if not creds["api_key"]:
        print(
            f"ERROR: no api_key found.\n"
            f"  Tried: {args.credentials} (table [rrze], key api_key)\n"
            f"  Then : env-var LLMAPI_KEY\n"
            f"  Copy llm_api.example.toml -> llm_api.toml and fill in your\n"
            f"  RRZE key from https://hpc.fau.de/request-llm-api-key/.",
            file=sys.stderr,
        )
        return 2
    tail = creds["api_key"][-4:] if len(creds["api_key"]) > 8 else "set"
    print(f"[{args.family}] model={creds['model']}  base_url={creds['base_url']}  "
          f"key=***{tail}", flush=True)
    return run_agent(args.slug, args.family, args.iters,
                     model=creds["model"],
                     base_url=creds["base_url"],
                     api_key=creds["api_key"])


if __name__ == "__main__":
    sys.exit(main())
