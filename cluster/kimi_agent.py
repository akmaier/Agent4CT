"""Kimi-K2.6 autonomous DL-Sparse-View agent.

Drives one full autoresearch loop against the FAU/RRZE LLM gateway:
    base_url = https://hub.nhr.fau.de/api/llmgw/v1
    model    = moonshotai/Kimi-K2.6
    auth     = Bearer  $LLMAPI_KEY

Per the operator's directive this agent is ISOLATED from what other agents
have already found: it must not read the cross-run scratchpad
(``docs/runs/observations.jsonl``) nor any other agent's slug directory
under ``docs/runs/``. Its own slug lives under ``docs/runs/<own_slug>/``
and that path *is* allowed.

Usage:

    export LLMAPI_KEY=...           # one-shot from the user's shell
    python cluster/kimi_agent.py    \\
        --slug   dl-sparse-view-kimi-20260514-01  \\
        --iters  5

The agent runs N iterations, each one:
  1. Reads the program contract + its OWN previous observations.
  2. Decides + writes a single change into
     pentathlon/dl_sparse_view_kimi/solver.py.
  3. scp's the solver + sbatch to the cluster, submits, polls for
     result.json + comparison.png.
  4. Calls scripts/agent4ct_record.py with --agent kimi --model
     moonshotai/Kimi-K2.6 to commit + publish the observation.

The agent's tool surface is tight:
    read_file(path)        — only the allowed allow-list below.
    write_file(path, txt)  — only the agent's own slug solver.
    run_bash(cmd)          — any shell command, captured for the LLM
                              to read (the run is opt-in; the LLM picks
                              what to scp / ssh / sbatch).
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Lazy-import openai so the file is at least importable without the
# package (the agent loop will refuse to run, but unit-test imports work).
try:
    from openai import OpenAI
    _OPENAI_OK = True
except ImportError:
    OpenAI = None  # type: ignore
    _OPENAI_OK = False

# --- isolation rules ----------------------------------------------------

# Paths the Kimi agent is allowed to *read*. Everything else is refused
# at the tool level so the LLM physically cannot pull cross-agent data.
def _allowed_read(path: Path, own_slug: str) -> bool:
    rel = path.resolve()
    try:
        rel = rel.relative_to(REPO)
    except ValueError:
        return False
    s = str(rel).replace(os.sep, "/")
    own_run_dir = f"docs/runs/{own_slug}"
    # Explicitly blocked: cross-agent scratchpad + other slugs.
    if s == "docs/runs/observations.jsonl":
        return False
    if s.startswith("docs/runs/") and not s.startswith(own_run_dir):
        return False
    # Explicitly allowed roots.
    for prefix in (
        "pentathlon/dl_sparse_view_kimi/",
        "pentathlon/dl_sparse_view/program.md",
        "ddssl_ldct/",
        "literature/",
        "cluster/slurm/dl_sparse_view_kimi_5min.sbatch",
        f"{own_run_dir}/",
        "scripts/agent4ct_record.py",
    ):
        if s == prefix.rstrip("/") or s.startswith(prefix):
            return True
    return False


# The agent may only edit its own slot.
def _allowed_write(path: Path) -> bool:
    try:
        rel = path.resolve().relative_to(REPO)
    except ValueError:
        return False
    s = str(rel).replace(os.sep, "/")
    return s == "pentathlon/dl_sparse_view_kimi/solver.py"


# --- tool implementations ----------------------------------------------

def tool_read_file(args: dict, own_slug: str) -> str:
    p = REPO / args["path"]
    if not _allowed_read(p, own_slug):
        return f"REFUSED: {args['path']} is outside the agent's allow-list. The cross-agent observations.jsonl and other agents' slug directories are off-limits by design."
    try:
        return p.read_text()[: args.get("max_bytes", 200_000)]
    except FileNotFoundError:
        return f"NOT FOUND: {args['path']}"


def tool_write_file(args: dict, _own_slug: str) -> str:
    p = REPO / args["path"]
    if not _allowed_write(p):
        return f"REFUSED: {args['path']} is read-only for this agent. Only pentathlon/dl_sparse_view_kimi/solver.py is editable."
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(args["content"])
    return f"OK ({len(args['content'])} bytes -> {args['path']})"


def tool_run_bash(args: dict, _own_slug: str) -> str:
    cmd = args["command"]
    timeout = int(args.get("timeout_s", 300))
    print(f"[bash] $ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=timeout, cwd=str(REPO))
    out = (p.stdout or "") + (("\nSTDERR:\n" + p.stderr) if p.stderr else "")
    return f"exit={p.returncode}\n{out[-6000:]}"


TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file under the project root. Refuses paths outside the agent's allow-list.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Repo-relative path."},
            "max_bytes": {"type": "integer", "default": 200000},
        }, "required": ["path"]}
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Overwrite the agent's editable file (pentathlon/dl_sparse_view_kimi/solver.py).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        }, "required": ["path", "content"]}
    }},
    {"type": "function", "function": {
        "name": "run_bash",
        "description": "Run a shell command (e.g. scp / ssh / sbatch / python). Use to submit and poll Slurm jobs.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "timeout_s": {"type": "integer", "default": 300},
        }, "required": ["command"]}
    }},
]

TOOL_DISPATCH = {
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "run_bash": tool_run_bash,
}


# --- agent loop ---------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """You are the autonomous Kimi-K2.6 agent on the
Agent4CT DL-Sparse-View challenge. Your slug is `{slug}`.

GOAL: minimise the validation RMSE of a 2D sparse-view fan-beam CT
reconstruction by editing `pentathlon/dl_sparse_view_kimi/solver.py` and
running short Slurm iterations on the FAU/i5 cluster. Headroom in
[0,1]; higher is better; `1.0` means RMSE=0 vs. ground truth.

ISOLATION RULES (hard contract, enforced at the tool level):
  * You may read: pentathlon/dl_sparse_view_kimi/, pentathlon/dl_sparse_view/program.md,
    ddssl_ldct/, literature/, cluster/slurm/dl_sparse_view_kimi_5min.sbatch,
    and ONLY your own slug's docs/runs/ directory.
  * You MUST NOT read docs/runs/observations.jsonl nor any other agent's
    docs/runs/<other-slug>/ directory. (The read_file tool will refuse.)
  * You may only WRITE pentathlon/dl_sparse_view_kimi/solver.py.

WORKFLOW per iteration:
  1. Read program.md + (if iter>1) your own last observation.json.
  2. Decide ONE change to solver.py (architecture | optimizer | loss |
     augmentation | other).
  3. write_file pentathlon/dl_sparse_view_kimi/solver.py with the new content.
  4. scp it to the cluster:
       scp pentathlon/dl_sparse_view_kimi/solver.py \\
         maier@cluster.i5.informatik.uni-erlangen.de:/cluster/maier/Agent4CT/pentathlon/dl_sparse_view_kimi/solver.py
  5. Submit:
       ssh maier@cluster.i5.informatik.uni-erlangen.de \\
         "cd /cluster/maier/Agent4CT && sbatch cluster/slurm/dl_sparse_view_kimi_5min.sbatch"
     Capture the job id.
  6. Poll until result.json appears, then scp it + comparison.png back to /tmp/.
  7. Read /tmp/kimi_result.json. Decide kept = true/false.
  8. Call scripts/agent4ct_record.py via run_bash, with
       --slug {slug} --iter <N> --agent kimi --model moonshotai/Kimi-K2.6
     (env-var AGENT4CT_AGENT/AGENT4CT_MODEL also work; flags win).

You have a budget of N iterations and ~30 tool calls per iteration.
Stop early if your headroom stalls for 3 iters.

Be concise. Cite which knob you changed and why in --rationale.
"""


def run_agent(slug: str, n_iters: int, *, model: str, base_url: str,
              api_key: str) -> int:
    if not _OPENAI_OK:
        print("ERROR: `pip install openai` first.", file=sys.stderr)
        return 2
    client = OpenAI(base_url=base_url, api_key=api_key)
    system = SYSTEM_PROMPT_TEMPLATE.format(slug=slug)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content":
         f"Begin iteration loop. Plan and run up to {n_iters} iterations. "
         f"After each iteration, summarise val_score + headroom + delta. "
         f"After the last one, write 'DONE'."},
    ]
    max_steps = 30 * n_iters
    for step in range(max_steps):
        print(f"\n=== step {step+1}/{max_steps} ===", flush=True)
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS_SCHEMA,
            tool_choice="auto", temperature=0.2,
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        tool_calls = getattr(msg, "tool_calls", None) or []
        if content:
            print(f"[kimi] {content[:1000]}", flush=True)
        # Append assistant turn including tool calls so the model sees its history.
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
            if "DONE" in content.upper() and "DONE" in content:
                print("[kimi] agent signalled DONE.", flush=True)
                return 0
            # No tools and no done signal: prompt once more, then exit.
            messages.append({"role": "user",
                             "content": "Continue, or write DONE if finished."})
            continue
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            print(f"[tool] {name}({json.dumps(args)[:300]})", flush=True)
            fn = TOOL_DISPATCH.get(name)
            if fn is None:
                result = f"UNKNOWN TOOL: {name}"
            else:
                try:
                    result = fn(args, slug)
                except Exception as e:
                    result = f"ERROR: {type(e).__name__}: {e}"
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result[:8000],
            })
    print("[kimi] step budget exhausted.", flush=True)
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slug", required=True,
                   help="The agent's own run slug (must already be created).")
    p.add_argument("--iters", type=int, default=3,
                   help="How many solver iterations to run.")
    p.add_argument("--model", default="moonshotai/Kimi-K2.6")
    p.add_argument("--base-url", default="https://hub.nhr.fau.de/api/llmgw/v1")
    args = p.parse_args()

    api_key = os.environ.get("LLMAPI_KEY", "").strip()
    if not api_key:
        print("ERROR: LLMAPI_KEY env-var is empty. Set it from your shell "
              "(see https://hpc.fau.de/request-llm-api-key/) before launching.",
              file=sys.stderr)
        return 2
    return run_agent(args.slug, args.iters, model=args.model,
                     base_url=args.base_url, api_key=api_key)


if __name__ == "__main__":
    sys.exit(main())
