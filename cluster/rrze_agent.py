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

Credentials + endpoint + per-family default model live in the gitignored
``llm_api.toml`` (see ``llm_api.example.toml`` for the shape):

    [rrze]
    api_key  = "..."                                    # required
    base_url = "https://hub.nhr.fau.de/api/llmgw/v1"    # optional

    [rrze.models]
    kimi     = "moonshotai/Kimi-K2.6"
    deepseek = "deepseek-ai/DeepSeek-V284B4-Flash"
    mistral  = "mistralai/Mistral-Medium-3.5-128B"
    gptoss   = "openai/gpt-oss-120b"

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
import subprocess
import sys
from pathlib import Path
from typing import Any

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
    "kimi":     {"model": "moonshotai/Kimi-K2.6"},
    "deepseek": {"model": "deepseek-ai/DeepSeek-V284B4-Flash"},
    "mistral":  {"model": "mistralai/Mistral-Medium-3.5-128B"},
    "gptoss":   {"model": "openai/gpt-oss-120b"},
}

def family_solver_dir(family: str) -> str:
    return f"pentathlon/dl_sparse_view_{family}/"

def family_sbatch(family: str) -> str:
    return f"cluster/slurm/dl_sparse_view_{family}_5min.sbatch"


# --- credentials --------------------------------------------------------

def load_credentials(toml_path: Path, family: str) -> dict[str, str]:
    """Read api_key + base_url + family-specific model from llm_api.toml.

    The file is gitignored on purpose. Falls back to env-vars for any
    missing piece:
        LLMAPI_KEY   -> api_key
        LLM_BASE_URL -> base_url   (default: FAU gateway)
        LLM_MODEL    -> model      (default: per-family default above)
    """
    rrze: dict[str, Any] = {}
    if toml_path.exists():
        if not _TOML_OK:
            raise RuntimeError(
                "Python >= 3.11 required to parse llm_api.toml")
        with toml_path.open("rb") as f:
            data = tomllib.load(f)
        rrze = dict(data.get("rrze") or {})
    models_table = dict(rrze.get("models") or {})
    api_key = (rrze.get("api_key")
               or os.environ.get("LLMAPI_KEY", "")).strip()
    base_url = (rrze.get("base_url")
                or os.environ.get("LLM_BASE_URL")
                or "https://hub.nhr.fau.de/api/llmgw/v1")
    # Precedence: env (LLM_MODEL) > [rrze.models][family] > legacy flat
    # rrze.model > built-in default.
    model = (os.environ.get("LLM_MODEL")
             or models_table.get(family)
             or rrze.get("model")
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


def tool_run_bash(args: dict, _ctx: dict) -> str:
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
        "name": "run_bash",
        "description": "Run a shell command from the repo root (scp / ssh / sbatch / python / git).",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "timeout_s": {"type": "integer", "default": 300},
        }, "required": ["command"]},
    }},
]

TOOL_DISPATCH = {
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "run_bash":  tool_run_bash,
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

WORKFLOW per iteration:
  1. Read program.md + (if iter > 1) your own last observation.json
     under docs/runs/{slug}/iterations/iter-<N-1>/.
  2. Decide ONE change to solver.py (architecture | optimizer | loss |
     augmentation | other).
  3. write_file {solver_dir}solver.py with the new content.
  4. scp it to the cluster:
       scp {solver_dir}solver.py \\
         maier@cluster.i5.informatik.uni-erlangen.de:/cluster/maier/Agent4CT/{solver_dir}solver.py
  5. Submit:
       ssh maier@cluster.i5.informatik.uni-erlangen.de \\
         "cd /cluster/maier/Agent4CT && sbatch {sbatch}"
     Capture the job id.
  6. Poll until result.json appears in
       /cluster/maier/Agent4CT/runs/iter-{family}-<jid>/
     then scp result.json + comparison.png back to /tmp/.
  7. Read /tmp/result.json. Decide kept = true / false.
  8. Record the iteration via the env-var path so the journal carries
     the right agent + model labels:
       AGENT4CT_AGENT={family} AGENT4CT_MODEL={model} \\
       .venv/bin/python scripts/agent4ct_record.py record \\
         --slug {slug} --iter <N> \\
         --val-score ... --headroom ... --params-M ... --train-n ... \\
         --change-class <architecture|optimizer|loss|augmentation|other> \\
         --rationale '...' --advice '...' \\
         --kept <true|false> --commit $(git rev-parse --short HEAD) \\
         --comparison /tmp/comparison.png \\
         --solver {solver_dir}solver.py

Be concise. Cite the knob you changed. Total budget: {iters} iterations,
~30 tool calls each. After the last iteration write 'DONE'.
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
    ctx = {"slug": slug, "family": family}
    max_steps = 30 * n_iters
    for step in range(max_steps):
        print(f"\n=== [{family}] step {step+1}/{max_steps} ===", flush=True)
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS_SCHEMA,
            tool_choice="auto", temperature=0.2,
        )
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
    p.add_argument("--credentials", default=str(REPO / "llm_api.toml"))
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
