"""
experiment_big_model.py - settle the two "implemented but off by default" decisions
on a machine big enough to run a real model.

Two features in this project are fully implemented, fully measured, and shipped OFF:

  * contextual retrieval (contextual.py, CONTEXTUAL_RETRIEVAL_ENABLED)
  * structure-aware chunking with parent expansion (chunking.py, CHUNKING_MODE)

Both were measured against dataset v2.0 on qwen2.5:0.5b, the only model that fits in
the 2 GB sandbox where this project was built. Contextual retrieval tied the gate at
16/16 and cost a correctness point; structure mode lost it 13/16. Neither verdict is
evidence about the features - it is evidence about the features *at 0.5b*, where a
situating blurb is written by a model that produced usable blurbs for 2 of 4 chunks
in one document and paraphrased the rest. config.py's own default answer model is
qwen2.5vl:3b. This script re-runs the same experiment at a model that can actually
be judged on its merits.

It is a measuring instrument, not a deployment tool. It never edits config.py, never
writes .env, and never promotes a run to evaluation/baseline.json. It backs the index
up, restores it byte for byte when it finishes, and prints a verdict rule it applies
mechanically rather than leaving the reader to interpret a table.

Methodology, because the numbers are worthless without it
----------------------------------------------------------
A model change and a feature change must never be mixed in one comparison. So:

  arm 1  reference   the new model, shipped configuration
                     -> compared against evaluation/baseline.json (qwen2.5:0.5b).
                        This answers "what does the bigger model alone do?", and it is
                        a MODEL decision, not a feature decision.
  arm 2  contextual  the new model + CONTEXTUAL_RETRIEVAL_ENABLED=true
  arm 3  structure   the new model + CHUNKING_MODE=structure
                     -> arms 2 and 3 are compared against arm 1's report, never
                        against baseline.json. Same model, same corpus, one flag: that
                        is the only comparison that can justify flipping a default.

Retrieval metrics (precision, recall, MRR) do not use the answer model at all, so a
model change should leave them untouched - except for image items, whose indexed
content is built from the vision model's description (images.py:158). A vision-capable
model therefore changes image ingestion, and the reference arm against the 0.5b
baseline is where that shows up. Answer metrics change with the model, the feature,
and their interaction, which is exactly why the feature arms are read against arm 1.

Each arm reindexes. Turning a flag on without reindexing does nothing to an existing
corpus (see the comment in rag_pipeline.py:287), so an arm that reused the shipped
index would measure the flag's absence and report it as the flag's effect.

Usage
-----
    # 1. See the plan, the resolved configuration and the cost estimate. Touches nothing.
    python scripts/experiment_big_model.py --plan --model qwen2.5:3b

    # 2. Run all three arms (budget 1-3 hours; see the estimate --plan prints).
    python scripts/experiment_big_model.py --model qwen2.5:3b --yes

    # 3. Just the contextual question, reusing a reference arm you already ran.
    python scripts/experiment_big_model.py --model qwen2.5:3b --arms contextual \
        --reference-report evaluation/reports/report_XXXXXXXX_XXXXXX.json --yes

    # 4. Add the LLM-judge arm. Advisory only, four extra completions per answer,
    #    but at 3B the judge is no longer the content-blind scorer it was at 0.5b.
    python scripts/experiment_big_model.py --model qwen2.5vl:3b --judge --yes

Outputs land in evaluation/experiments/: one JSON, one markdown summary, and a log per
arm per stage. Both are gitignored, like the reports they cite.

Exit codes: 0 every requested arm was measured and its report verified - read the
verdicts, they are the result, and "KEEP OFF" is a successful experiment. 1 an arm could
not be measured, or the index did not restore byte-identical, or baseline.json moved.
2 refused before touching anything (preflight). 130 interrupted; the index is still
restored and whatever was measured is still written out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENT_DIR = PROJECT_ROOT / "evaluation" / "experiments"
LOG_DIR = EXPERIMENT_DIR / "logs"
REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"
BASELINE_PATH = PROJECT_ROOT / "evaluation" / "baseline.json"

# Metrics the gate reads, and the direction that counts as better. Same lists
# evaluation.py gates on; duplicated here so this script's verdict can be read
# without opening the harness.
HIGHER_IS_BETTER = (
    "retrieval_precision", "retrieval_recall", "mrr", "citation_accuracy",
    "answer_correctness", "groundedness", "abstention_accuracy",
)
LOWER_IS_BETTER = ("hallucination_rate", "forbidden_leak_rate")
RETRIEVAL_ONLY = ("retrieval_precision", "retrieval_recall", "mrr")
PER_QUERY_WATCHED = ("answer_correctness", "citation_accuracy", "groundedness", "recall")

# ---------------------------------------------------------------------------
# Arms. A delta is a set of environment variables, because config.py resolves
# environment -> .env -> default, so an arm can change configuration without
# writing to any file that outlives the arm.
# ---------------------------------------------------------------------------
ARMS: dict[str, dict] = {
    "reference": {
        "label": "new model, shipped configuration",
        "delta": {},
        "expect": {"blurbs": "==0", "tables": ">=1", "parents": "==0"},
        "compare_against": "baseline",
        "compare_means": "MODEL change against the qwen2.5:0.5b baseline of record",
    },
    "contextual": {
        "label": "contextual retrieval ON (situating blurbs at ingest)",
        "delta": {"CONTEXTUAL_RETRIEVAL_ENABLED": "true"},
        "expect": {"blurbs": ">=1", "tables": ">=1", "parents": "==0"},
        "compare_against": "reference",
        "compare_means": "FEATURE change at the same model, same corpus, one flag",
    },
    "structure": {
        "label": "structure-aware chunking ON (sections + parent expansion)",
        "delta": {"CHUNKING_MODE": "structure"},
        "expect": {"blurbs": "==0", "tables": ">=1", "parents": ">=1"},
        "compare_against": "reference",
        "compare_means": "FEATURE change at the same model, same corpus, one flag",
    },
}

# Costs measured in the build sandbox on qwen2.5:0.5b, CPU only, 2 GB RAM, with
# OLLAMA_KEEP_ALIVE=0 so the model was reloaded for every call. They are quoted to
# give an order of magnitude, not a prediction.
SANDBOX_COSTS = {
    "reindex_s": 23.3,
    "reindex_contextual_s": 1089.1,
    "gate_s": 234.7,
    "retrieval_only_s": 33.0,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def say(text: str = "") -> None:
    print(text, flush=True)


def rule(char: str = "=") -> None:
    say(char * 78)


# ---------------------------------------------------------------------------
# Configuration resolution
# ---------------------------------------------------------------------------
def resolve_settings(env: dict[str, str]) -> dict[str, object]:
    """Import config.py with `env` applied and report what actually resolved.

    Run in a child process: settings is built once at import time, so a parent that
    already imported it could not see the arm's overrides, and mutating os.environ in
    this process would leak one arm's configuration into the next.
    """
    code = (
        "import json, os, sys;"
        "sys.path.insert(0, os.environ['EAI_ROOT']);"
        "from config import settings as s;"
        "print(json.dumps({"
        "'ollama_model': s.ollama_model,"
        "'ollama_keep_alive': s.ollama_keep_alive,"
        "'ollama_num_ctx': s.ollama_num_ctx,"
        "'ollama_num_predict': s.ollama_num_predict,"
        "'chunking_mode': s.chunking_mode,"
        "'structure_min_chunk_chars': s.structure_min_chunk_chars,"
        "'parent_expansion_enabled': s.parent_expansion_enabled,"
        "'contextual_retrieval_enabled': s.contextual_retrieval_enabled,"
        "'contextual_min_items': s.contextual_min_items,"
        "'pdf_table_extraction_enabled': s.pdf_table_extraction_enabled,"
        "'inference_backend': s.inference_backend,"
        "'index_dir': str(s.index_dir)}))"
    )
    child_env = dict(os.environ)
    child_env.update(env)
    child_env["EAI_ROOT"] = str(PROJECT_ROOT)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=child_env, encoding="utf-8", errors="replace",
                          cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout or "").strip()[-600:]}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": f"could not parse resolved settings: {proc.stdout.strip()[-300:]}"}


def settings_field_names() -> set[str]:
    """Every field config.Settings exposes, so an arm's env key can be validated."""
    code = (
        "import os, sys; sys.path.insert(0, os.environ['EAI_ROOT']);"
        "from config import Settings; print('\\n'.join(sorted(Settings.model_fields)))"
    )
    child_env = dict(os.environ)
    child_env["EAI_ROOT"] = str(PROJECT_ROOT)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=child_env, encoding="utf-8", errors="replace",
                          cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        return set()
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


# ---------------------------------------------------------------------------
# Index state and backup
# ---------------------------------------------------------------------------
def index_paths() -> list[Path]:
    resolved = resolve_settings({})
    index_dir = Path(str(resolved.get("index_dir") or PROJECT_ROOT / "index"))
    return [index_dir / "store.db", index_dir / "faiss.index"]


def index_state() -> dict[str, int]:
    """What the persisted index actually contains, read from SQLite.

    Read from the database rather than from settings, because the point of the check
    is to catch an arm whose flags did not reach the ingest path. A flag that is on in
    configuration and off in the data would otherwise produce a confident, meaningless
    comparison.
    """
    import sqlite3

    db = index_paths()[0]
    if not db.exists():
        return {"sources": 0, "items": 0, "relationships": 0, "parents": 0,
                "blurbs": 0, "tables": 0}
    con = sqlite3.connect(str(db))
    try:
        def one(sql: str) -> int:
            try:
                return int(con.execute(sql).fetchone()[0])
            except Exception:
                return 0

        return {
            "sources": one("select count(*) from sources"),
            "items": one("select count(*) from items"),
            "relationships": one("select count(*) from relationships"),
            "parents": one("select count(*) from parents"),
            "blurbs": one("select count(*) from items where "
                          "json_extract(metadata,'$.context') is not null"),
            "tables": one("select count(*) from items where "
                          "json_extract(metadata,'$.is_table')=1"),
        }
    finally:
        con.close()


def check_expectations(arm: str, state: dict[str, int]) -> list[str]:
    """Return the expectations an arm's index state failed to meet.

    Expectations are read against the persisted index, not against configuration,
    because the failure this catches is a flag that never reached the ingest path.
    """
    failures = []
    for key, rule_text in ARMS[arm]["expect"].items():
        if rule_text.startswith("=="):
            ok = int(state.get(key, 0)) == int(rule_text[2:])
        elif rule_text.startswith(">="):
            ok = int(state.get(key, 0)) >= int(rule_text[2:])
        else:
            failures.append(f"internal: expectation {rule_text!r} for {key} is not parseable")
            continue
        if not ok:
            failures.append(f"{key} is {state.get(key, 0)}, expected {rule_text}")
    return failures


class IndexBackup:
    """Copy the index aside and put it back, proving the restore with hashes."""

    def __init__(self, stamp: str):
        self.dir = EXPERIMENT_DIR / f"index_backup_{stamp}"
        self.paths = index_paths()
        self.hashes: dict[str, str] = {}
        self.restored = False

    def take(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        for path in self.paths:
            self.hashes[path.name] = sha256(path)
            if path.exists():
                shutil.copy2(path, self.dir / path.name)
        say(f"  index backed up to {self.dir}")
        for name, digest in self.hashes.items():
            say(f"    {name:<12} sha256 {digest[:16]}...")

    def restore(self) -> bool:
        for path in self.paths:
            backup = self.dir / path.name
            if backup.exists():
                shutil.copy2(backup, path)
        after = {path.name: sha256(path) for path in self.paths}
        self.restored = after == self.hashes
        shutil.rmtree(self.dir, ignore_errors=True)
        return self.restored


# ---------------------------------------------------------------------------
# Running children with their output on screen and in a log
# ---------------------------------------------------------------------------
def run_logged(cmd: list[str], env: dict[str, str], log_path: Path,
               title: str) -> tuple[int, float]:
    """Run a child process, echoing its output and teeing it to `log_path`."""
    say(f"\n  --- {title}")
    say(f"      $ {' '.join(str(part) for part in cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        log.write(f"$ {' '.join(str(part) for part in cmd)}\n")
        log.write(f"# started {datetime.now().isoformat(timespec='seconds')}\n\n")
        proc = subprocess.Popen(
            [str(part) for part in cmd], cwd=str(PROJECT_ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            stripped = line.rstrip()
            if stripped:
                say(f"      | {stripped}")
        proc.wait()
        elapsed = time.time() - started
        log.write(f"\n# exit {proc.returncode} after {elapsed:.1f}s\n")
    say(f"      = exit {proc.returncode} in {elapsed:.1f}s  (log: {log_path.name})")
    return proc.returncode, elapsed


def newest_reports(before: set[Path]) -> list[Path]:
    return sorted(set(REPORTS_DIR.glob("report_*.json")) - before)


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def report_flags(report: dict) -> dict:
    """The configuration a report says it was taken with, from either report shape."""
    manifest = report.get("manifest") or {}
    if manifest:
        return {
            "llm": manifest.get("models", {}).get("llm"),
            "chunking_mode": manifest.get("chunking", {}).get("mode"),
            "contextual_retrieval": manifest.get("ingestion", {}).get("contextual_retrieval"),
            "pdf_tables": manifest.get("ingestion", {}).get("pdf_tables"),
            "parent_expansion": manifest.get("chunking", {}).get("parent_expansion_enabled"),
            "git_commit": (report.get("git_commit")
                           or manifest.get("application", {}).get("git_commit")),
            "git_dirty": manifest.get("application", {}).get("git_dirty"),
        }
    config = report.get("configuration") or {}
    return {
        "llm": (report.get("models") or {}).get("llm"),
        "chunking_mode": config.get("chunking_mode"),
        "contextual_retrieval": config.get("contextual_retrieval_enabled"),
        "pdf_tables": config.get("pdf_table_extraction_enabled"),
        "parent_expansion": config.get("parent_expansion_enabled"),
        "git_commit": report.get("git_commit"),
        "git_dirty": None,
    }


def verify_arm_report(report: dict, arm: str, model: str) -> list[str]:
    """Refuse to accept a report that does not prove the arm ran as declared."""
    problems = []
    flags = report_flags(report)
    if flags.get("llm") != model:
        problems.append(f"report says the model was {flags.get('llm')!r}, not {model!r}")
    expected_mode = "structure" if arm == "structure" else None
    if expected_mode and flags.get("chunking_mode") != expected_mode:
        problems.append(f"report says chunking_mode={flags.get('chunking_mode')!r}, "
                        f"expected {expected_mode!r}")
    if arm == "contextual" and flags.get("contextual_retrieval") is not True:
        problems.append("report says contextual_retrieval was not enabled")
    if arm != "contextual" and flags.get("contextual_retrieval") is True:
        problems.append("report says contextual_retrieval was enabled for an arm that "
                        "should not have it")
    if arm != "structure" and flags.get("chunking_mode") == "structure":
        problems.append("report says chunking_mode=structure for an arm that should be "
                        "fixed")
    return problems


# ---------------------------------------------------------------------------
# Comparison and verdict
# ---------------------------------------------------------------------------
def query_status(row: dict) -> str | None:
    """The PASS/CHECK verdict for one query, from the fields the report actually stores.

    A retrieval-only record carries `status` (PASS, CHECK or UNDECIDED). A full record
    does not: evaluation.py computes the same verdict for its console line and never
    persists it, so it is re-derived here with that rule verbatim -- abstention correct,
    correctness at least 0.5, and no forbidden leak. Returns None when a row has neither
    a stored status nor an answer to judge.
    """
    stored = row.get("status")
    if stored:
        return str(stored)
    correctness = row.get("answer_correctness")
    if not isinstance(correctness, (int, float)):
        return None
    abstention_ok = bool(row.get("abstention_correct"))
    forbidden = bool(row.get("forbidden_hits"))
    return "PASS" if abstention_ok and correctness >= 0.5 and not forbidden else "CHECK"


def metric_delta(off: dict, on: dict, metric: str) -> tuple[float | None, float | None]:
    a, b = off.get(metric), on.get(metric)
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None, None
    return float(a), float(b)


def compare_arms(off: dict, on: dict, tolerance: float) -> dict:
    """Paired comparison of two aggregate metric blocks."""
    worse, better, unchanged, absent = [], [], [], []
    for metric in HIGHER_IS_BETTER + LOWER_IS_BETTER:
        a, b = metric_delta(off, on, metric)
        if a is None or b is None:
            absent.append(metric)
            continue
        if metric in HIGHER_IS_BETTER:
            if b < a - tolerance:
                worse.append((metric, a, b))
            elif b > a + tolerance:
                better.append((metric, a, b))
            else:
                unchanged.append((metric, a, b))
        else:
            if b > a + tolerance:
                worse.append((metric, a, b))
            elif b < a - tolerance:
                better.append((metric, a, b))
            else:
                unchanged.append((metric, a, b))
    return {"worse": worse, "better": better, "unchanged": unchanged, "absent": absent}


def per_query_movements(off_report: dict, on_report: dict) -> list[dict]:
    off_rows = {row["id"]: row for row in off_report.get("per_query", [])}
    moves = []
    for row in on_report.get("per_query", []):
        before = off_rows.get(row["id"])
        if not before:
            continue
        deltas = {}
        for metric in PER_QUERY_WATCHED:
            a, b = before.get(metric), row.get(metric)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(b - a) > 1e-9:
                deltas[metric] = (round(float(a), 4), round(float(b), 4))
        before_status, after_status = query_status(before), query_status(row)
        status_changed = before_status != after_status
        if deltas or status_changed:
            moves.append({
                "id": row["id"],
                "query": row.get("query", "")[:90],
                "before_status": before_status,
                "after_status": after_status,
                "status": f"{before_status} -> {after_status}" if status_changed
                          else str(after_status),
                "deltas": deltas,
            })
    return moves


def verdict_for(comparison: dict, gate_exit: int, moves: list[dict],
                dropped_queries: list[str], tolerance: float) -> tuple[str, list[str]]:
    """Apply the flip rule mechanically. Returns (verdict, reasons).

    The rule is the one this project has used for every opt-in feature: a default is
    flipped only when the arm passes its own gate, is not below the same-model
    reference on any gated metric, is strictly better on at least one, and does not
    take any individual query backwards. Tying is not winning - contextual retrieval
    tied 16/16 at 0.5b and still ships off, because a tie buys nothing and costs
    eighteen minutes of ingest.
    """
    reasons = []
    if gate_exit != 0:
        reasons.append(f"the arm's own gate did not pass (evaluation.py exit {gate_exit})")
    if comparison["worse"]:
        reasons.append("below the same-model reference on: "
                       + ", ".join(f"{m} {a:.4f} -> {b:.4f}" for m, a, b in comparison["worse"]))
    if dropped_queries:
        reasons.append("per-query status dropped for: " + ", ".join(dropped_queries))
    # Every metric in PER_QUERY_WATCHED is higher-is-better, so any fall is a regression.
    regressed = sorted({move["id"] for move in moves
                        if any(after < before - tolerance
                               for before, after in move["deltas"].values())})
    if regressed:
        reasons.append("individual queries moved down on a watched metric: "
                       + ", ".join(regressed))
    if reasons:
        return "KEEP OFF", reasons
    if not comparison["better"]:
        return "KEEP OFF", ["not worse than the reference, but not better either: a tie buys "
                            "nothing and this feature costs ingest time on every reindex"]
    return "RECOMMEND ON", [
        "gate passed, no gated metric below the same-model reference, and strictly better on: "
        + ", ".join(f"{m} {a:.4f} -> {b:.4f}" for m, a, b in comparison["better"]),
        "this is a recommendation to a human: flip the default in config.py by hand, re-run "
        "the gate at the shipped configuration, and keep both reports",
    ]


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def ollama_tags(host: str) -> dict | None:
    import urllib.request

    url = host.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def git_state() -> dict:
    def run(args: list[str]) -> str:
        proc = subprocess.run(["git", *args], cwd=str(PROJECT_ROOT), capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
        return proc.stdout.strip()

    return {
        "commit": run(["rev-parse", "--short", "HEAD"]),
        "dirty_files": [line for line in run(["status", "--porcelain"]).splitlines() if line],
    }


def preflight(args, model: str) -> tuple[list[str], list[str], dict]:
    """Return (blocking problems, warnings, facts)."""
    problems: list[str] = []
    warnings: list[str] = []
    facts: dict = {}

    git = git_state()
    facts["git"] = git
    if git["dirty_files"] and not args.allow_dirty:
        problems.append(
            f"the working tree has {len(git['dirty_files'])} uncommitted change(s). A number "
            "that cannot be attributed to a commit is not worth the hours this experiment "
            "costs. Commit or stash first, or pass --allow-dirty and accept that the report "
            "will say git_dirty=true.")

    if not BASELINE_PATH.exists():
        problems.append(f"{BASELINE_PATH} is missing: the reference arm has nothing to be "
                        "compared against.")
    else:
        facts["baseline_sha256"] = sha256(BASELINE_PATH)

    paths = index_paths()
    if not all(p.exists() for p in paths):
        problems.append("no index to back up: " + ", ".join(str(p) for p in paths if not p.exists())
                        + ". Run scripts/reindex.py --yes first.")
    else:
        facts["index_sha256"] = {p.name: sha256(p) for p in paths}
        facts["index_state_before"] = index_state()

    resolved = resolve_settings({"OLLAMA_MODEL": model,
                                 "OLLAMA_KEEP_ALIVE": args.keep_alive})
    facts["resolved"] = resolved
    if "error" in resolved:
        problems.append(f"could not read config.py: {resolved['error']}")
    else:
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        facts["ollama_host"] = host
        tags = ollama_tags(host)
        if tags is None:
            problems.append(f"Ollama is not answering at {host}. Start it (on Windows: "
                            "`ollama serve`, or the Ollama tray app) and re-run.")
        else:
            available = [m.get("name") for m in tags.get("models", [])]
            facts["ollama_models"] = available
            if model not in available:
                problems.append(
                    f"model {model!r} is not installed locally. Available: "
                    f"{', '.join(available) or 'none'}. This project is meant to run "
                    f"offline, so pull it first:  ollama pull {model}")
        num_ctx = int(resolved.get("ollama_num_ctx") or 0)
        if num_ctx >= 8192:
            warnings.append(
                f"OLLAMA_NUM_CTX resolves to {num_ctx}. On a 4 GB card a 3B model plus that "
                "much KV cache may not fit, and Ollama will silently fall back to CPU, which "
                "turns a one-hour experiment into a four-hour one. If latency comes back "
                "minutes per query, re-run with --num-ctx 4096.")
        if "vl" in model.lower():
            warnings.append(
                f"{model} is vision-capable, so image ingestion changes too: images.py builds "
                "the indexed content from the model's own description, which qwen2.5:0.5b "
                "could not produce. Expect the image item's text, and therefore image "
                "retrieval, to differ from the baseline. That is a finding, not a fault, but "
                "it means the reference arm is not a pure generation-quality comparison.")
        if int(resolved.get("ollama_num_predict") or 0) == 0:
            warnings.append("OLLAMA_NUM_PREDICT resolves to 0, which Ollama rejects with a "
                            "500. Leave it at the project default unless you know why.")

    fields = settings_field_names()
    facts["settings_fields"] = len(fields)
    if not fields:
        problems.append("could not introspect config.Settings, so the arm environment keys "
                        "cannot be validated. Fix the Python environment first "
                        "(repair_stack.sh in the build sandbox; the setup guide on Windows).")
    else:
        for arm, spec in ARMS.items():
            for key in spec["delta"]:
                if key.lower() not in fields:
                    problems.append(f"arm {arm!r} sets {key}, which is not a config.Settings "
                                    "field: the flag would be ignored and the arm would "
                                    "measure nothing")

    free_mb = shutil.disk_usage(str(PROJECT_ROOT)).free // (1024 * 1024)
    facts["disk_free_mb"] = free_mb
    if free_mb < 2048:
        warnings.append(f"only {free_mb} MB of disk free; logs, reports and index backups "
                        "need room, and a model pull needs gigabytes.")

    if platform.system() == "Windows":
        facts["platform"] = f"Windows {platform.release()}"
    else:
        facts["platform"] = f"{platform.system()} {platform.release()}"
        warnings.append("not running on Windows: fine for a rehearsal, but the verdict you "
                        "ship should come from the machine the project is deployed on.")
    return problems, warnings, facts


# ---------------------------------------------------------------------------
# Plan / estimate
# ---------------------------------------------------------------------------
def print_plan(args, arms: list[str], facts: dict) -> None:
    model = args.model
    rule()
    say(f"Experiment plan: {model} on {facts.get('platform', '?')}")
    rule()
    say(f"  git commit        {facts.get('git', {}).get('commit', '?')}"
        f"{'  (dirty)' if facts.get('git', {}).get('dirty_files') else '  (clean tree)'}")
    say(f"  ollama            {facts.get('ollama_host', '?')} - models: "
        f"{', '.join(facts.get('ollama_models') or []) or 'none'}")
    say(f"  index before      {json.dumps(facts.get('index_state_before', {}))}")
    say(f"  baseline.json     sha256 {str(facts.get('baseline_sha256', ''))[:16]}... "
        "(this script never writes it)")
    say(f"  keep_alive        {args.keep_alive} for every arm, so latencies are comparable")
    say(f"  judge             {'on (four extra completions per answer)' if args.judge else 'off'}")
    say(f"  fast retrieval    {'on' if not args.no_fast else 'off'} - a retrieval-only pass "
        "after each reindex, which costs ~30s and separates retrieval effects from "
        "generation effects")
    say()
    say("  ARM          CONFIG DELTA                          COMPARED AGAINST")
    say("  " + "-" * 74)
    for arm in arms:
        spec = ARMS[arm]
        delta = ", ".join(f"{k}={v}" for k, v in spec["delta"].items()) or "(none)"
        target = ("evaluation/baseline.json" if spec["compare_against"] == "baseline"
                  else "the reference arm's report")
        if arm != "reference" and args.reference_report:
            target = args.reference_report
        say(f"  {arm:<12} {delta:<41} {target}")
        say(f"               {spec['label']}")
        say(f"               {spec['compare_means']}")
    say()
    resolved = facts.get("resolved", {})
    if "error" in resolved:
        say(f"  could not resolve configuration: {resolved['error']}")
    else:
        say("  Resolved configuration for the reference arm (environment beats .env):")
        for key in ("ollama_model", "ollama_keep_alive", "ollama_num_ctx", "ollama_num_predict",
                    "chunking_mode", "structure_min_chunk_chars", "parent_expansion_enabled",
                    "contextual_retrieval_enabled", "contextual_min_items",
                    "pdf_table_extraction_enabled"):
            say(f"    {key:<32} {resolved.get(key)}")
    say()
    say("  Cost estimate. Measured in the build sandbox on qwen2.5:0.5b, CPU only,")
    say("  keep_alive=0: reindex 23s, contextual reindex 1089s, full gate 235s,")
    say("  retrieval-only 33s. A 3B model on a small GPU is usually faster per token")
    say("  than that CPU sandbox but has far more tokens to think about; on CPU")
    say("  fallback it is several times slower. Budget generously:")
    total_low = total_high = 0.0
    for arm in arms:
        reindex_low = 60 if arm != "contextual" else 300
        reindex_high = 300 if arm != "contextual" else 2400
        gate_low, gate_high = 240, 900
        if args.judge:
            gate_low, gate_high = gate_low * 3, gate_high * 4
        fast = 0 if args.no_fast else 40
        total_low += reindex_low + gate_low + fast
        total_high += reindex_high + gate_high + fast
        say(f"    {arm:<12} reindex {reindex_low}-{reindex_high}s   "
            f"retrieval-only {fast}s   gate {gate_low}-{gate_high}s")
    say(f"    {'TOTAL':<12} {total_low/60:.0f}-{total_high/60:.0f} minutes, if nothing has "
        "to be re-run")
    say()
    say("  What this script will not do: edit config.py, write .env, promote anything to")
    say("  baseline.json, or leave the index in an arm's configuration. The index is")
    say("  restored from a hash-verified backup even if the run is interrupted.")


# ---------------------------------------------------------------------------
# One arm
# ---------------------------------------------------------------------------
def run_arm(arm: str, args, stamp: str, facts: dict, reference_report: Path | None,
            reference_retrieval: Path | None = None) -> dict:
    spec = ARMS[arm]
    model = args.model
    env = dict(os.environ)
    env["OLLAMA_MODEL"] = model
    env["OLLAMA_KEEP_ALIVE"] = args.keep_alive
    if args.num_ctx:
        env["OLLAMA_NUM_CTX"] = str(args.num_ctx)
    env.update(spec["delta"])

    record: dict = {
        "arm": arm,
        "label": spec["label"],
        "delta": spec["delta"],
        "compare_against": spec["compare_against"],
        "compare_means": spec["compare_means"],
        "resolved": resolve_settings(env),
        "started": datetime.now().isoformat(timespec="seconds"),
    }

    rule()
    say(f"ARM {arm}: {spec['label']}")
    say(f"  model {model}   delta {spec['delta'] or '(none)'}   keep_alive {args.keep_alive}")
    rule("-")

    # 1. Reindex. Every arm, including the reference: the flags only reach a corpus
    #    through ingest, and a vision-capable model changes image items at ingest too.
    before = index_state()
    say(f"  index before reindex: {json.dumps(before)}")
    code, seconds = run_logged(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "reindex.py"), "--yes"],
        env, LOG_DIR / f"{arm}_{stamp}_reindex.log", f"{arm}: reindex")
    after = index_state()
    record["reindex"] = {"exit": code, "seconds": round(seconds, 1), "index_after": after}
    say(f"  index after reindex:  {json.dumps(after)}")
    if code != 0:
        record["error"] = f"reindex exited {code}; arm abandoned"
        say(f"  ABANDONED: reindex exited {code}. See the log.")
        return record

    # 2. Did the flag actually reach the data?
    failures = check_expectations(arm, after)
    if failures:
        record["error"] = "index does not match the arm: " + "; ".join(failures)
        say("  ABANDONED: the index does not match what this arm declares.")
        for failure in failures:
            say(f"      - {failure}")
        say("      A flag that is on in configuration and absent from the data would")
        say("      produce a confident, meaningless comparison, so this arm stops here.")
        return record

    if arm == "contextual":
        rate = after["blurbs"] / after["items"] if after["items"] else 0.0
        record["blurb_rate"] = round(rate, 4)
        say(f"  blurbs written: {after['blurbs']} of {after['items']} chunks ({rate:.0%}).")
        say(f"  At qwen2.5:0.5b this was 2 of 4 chunks in project_notes.docx; the rest were")
        say(f"  refusals or restatements that clean_blurb() dropped. This ratio is the")
        say(f"  mechanism the whole decision turns on, so read it before the metrics.")
        say(f"  Ingest cost {seconds:.0f}s against {SANDBOX_COSTS['reindex_s']:.0f}s with the "
            "flag off in the sandbox.")

    # 3. Retrieval-only pass: cheap, no generation, isolates retrieval effects.
    if not args.no_fast:
        snapshot = set(REPORTS_DIR.glob("report_*.json"))
        cmd = [sys.executable, str(PROJECT_ROOT / "evaluation.py"), "--retrieval-only"]
        # The reference arm compares against baseline.json, which is the path the
        # existing fast gate already exercises. A feature arm compares against the
        # reference arm's own retrieval-only report, so that both sides of the
        # comparison are the same mode and the same model.
        compare_target = None
        if spec["compare_against"] == "baseline" and BASELINE_PATH.exists():
            compare_target = BASELINE_PATH
        elif reference_retrieval is not None:
            compare_target = reference_retrieval
        if compare_target:
            cmd += ["--compare", str(compare_target)]
        code, seconds = run_logged(cmd, env, LOG_DIR / f"{arm}_{stamp}_retrieval.log",
                                   f"{arm}: retrieval-only gate")
        new = newest_reports(snapshot)
        record["retrieval_only"] = {
            "exit": code, "seconds": round(seconds, 1),
            "report": new[-1].name if new else None,
        }
        if new:
            rep = load_report(new[-1])
            record["retrieval_only"]["aggregate"] = {
                m: rep["aggregate"].get(m) for m in RETRIEVAL_ONLY}
            record["retrieval_only"]["index"] = rep.get("index")
            if arm == "reference":
                record["reference_retrieval_path"] = str(new[-1])

    # 4. The full gate.
    snapshot = set(REPORTS_DIR.glob("report_*.json"))
    cmd = [sys.executable, str(PROJECT_ROOT / "evaluation.py")]
    if args.judge:
        cmd.append("--judge")
    target: Path | None = None
    if spec["compare_against"] == "baseline":
        target = BASELINE_PATH if BASELINE_PATH.exists() else None
    elif args.reference_report:
        target = Path(args.reference_report)
    else:
        target = reference_report
    if target:
        cmd += ["--compare", str(target)]
        say(f"  comparing against {target.name}: {spec['compare_means']}")
    else:
        say("  no comparison target: this arm will be reported on its own numbers")
    code, seconds = run_logged(cmd, env, LOG_DIR / f"{arm}_{stamp}_gate.log",
                               f"{arm}: full gate")
    new = newest_reports(snapshot)
    if not new:
        record["error"] = f"the gate produced no report (exit {code})"
        say("  ABANDONED: evaluation.py wrote no report.")
        return record

    report_path = new[-1]
    report = load_report(report_path)
    record["gate"] = {
        "exit": code, "seconds": round(seconds, 1), "report": report_path.name,
        "report_path": str(report_path),
        "passed": report.get("comparison", {}).get("passed"),
        "headline": report.get("headline") or report.get("comparison", {}).get("headline"),
        "statuses": {status: sum(1 for row in report.get("per_query", [])
                                 if query_status(row) == status)
                     for status in ("PASS", "CHECK", "UNDECIDED")},
        "aggregate": report.get("aggregate", {}),
        "latency": report.get("latency", {}),
        "index": report.get("index"),
        "flags": report_flags(report),
    }
    if arm == "reference":
        record["reference_report_path"] = str(report_path)

    # 5. Does the report prove the arm ran as declared?
    problems = verify_arm_report(report, arm, model)
    if problems:
        record["error"] = "report contradicts the arm: " + "; ".join(problems)
        say("  REJECTED: the report does not match the arm that was supposed to produce it.")
        for problem in problems:
            say(f"      - {problem}")
        return record

    agg = report.get("aggregate", {})
    say(f"  gate exit {code}  |  " + "  ".join(
        f"{status}={count}" for status, count in record["gate"]["statuses"].items() if count))
    say("  " + "  ".join(
        f"{label} {agg.get(metric, float('nan')):.4f}" for label, metric in
        (("P", "retrieval_precision"), ("R", "retrieval_recall"), ("MRR", "mrr"),
         ("cite", "citation_accuracy"), ("corr", "answer_correctness"),
         ("ground", "groundedness")) if isinstance(agg.get(metric), (int, float))))
    latency = report.get("latency") or {}
    if latency:
        say(f"  latency mean {latency.get('mean_s', 0):.2f}s  p50 {latency.get('p50_s', 0):.2f}s"
            f"  p95 {latency.get('p95_s', 0):.2f}s  (keep_alive={args.keep_alive})")
    say(f"  report: {report_path.name}   git {record['gate']['flags'].get('git_commit')}"
        f"{'  DIRTY' if record['gate']['flags'].get('git_dirty') else ''}")
    record["finished"] = datetime.now().isoformat(timespec="seconds")
    return record


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def fmt(value: object, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"


def write_summary(stamp: str, model: str, args, facts: dict, records: list[dict],
                  comparisons: dict[str, dict]) -> tuple[Path, Path]:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = EXPERIMENT_DIR / f"experiment_{stamp}.json"
    md_path = EXPERIMENT_DIR / f"experiment_{stamp}.md"

    payload = {
        "experiment": "big-model paired arms",
        "stamp": stamp,
        "model": model,
        "keep_alive": args.keep_alive,
        "num_ctx_override": args.num_ctx,
        "judge": args.judge,
        "arms_requested": args.arms.split(","),
        "git": facts.get("git"),
        "platform": facts.get("platform"),
        "baseline_sha256": facts.get("baseline_sha256"),
        "baseline_sha256_after": facts.get("baseline_sha256_after"),
        "index_sha256": facts.get("index_sha256"),
        "index_restored_byte_identical": facts.get("index_restored"),
        "index_state_before": facts.get("index_state_before"),
        "index_state_after": facts.get("index_state_after"),
        "warnings": facts.get("warnings", []),
        "arms": records,
        "comparisons": comparisons,
        "sandbox_reference_costs_s": SANDBOX_COSTS,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    by_arm = {record["arm"]: record for record in records}
    lines = [
        f"# Big-model experiment: {model}",
        "",
        f"- Taken: {stamp} on {facts.get('platform', '?')}, git "
        f"`{facts.get('git', {}).get('commit', '?')}`"
        f"{' (dirty tree)' if facts.get('git', {}).get('dirty_files') else ' (clean tree)'}",
        f"- `OLLAMA_KEEP_ALIVE={args.keep_alive}` for every arm, so the latency column is "
        "internally comparable. It is not comparable to any report taken with a different "
        "keep-alive policy.",
        f"- Judge arm: {'on' if args.judge else 'off'}.",
        f"- Index restored byte-identical: **{facts.get('index_restored')}**. "
        f"`baseline.json` untouched: **{facts.get('baseline_sha256') == facts.get('baseline_sha256_after')}**.",
        "",
        "## Method",
        "",
        "The reference arm is compared against `evaluation/baseline.json`, which was taken "
        "on `qwen2.5:0.5b`: that answers the *model* question. Every other arm is compared "
        "against the reference arm's own report, because a feature can only be judged "
        "against the same model on the same corpus. Mixing the two comparisons is how a "
        "model upgrade gets credited for a feature, or a feature gets blamed for a model.",
        "",
        "## Arms",
        "",
        "| arm | configuration | gate exit | PASS/CHECK | report |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in records:
        gate = record.get("gate") or {}
        statuses = gate.get("statuses") or {}
        counts = f"{statuses.get('PASS', 0)}/{statuses.get('CHECK', 0)}"
        if statuses.get("UNDECIDED"):
            counts += f" +{statuses['UNDECIDED']} undecided"
        report_cell = str(gate.get("report") or record.get("error", "no report"))
        if (gate.get("flags") or {}).get("git_dirty"):
            report_cell += " (dirty tree)"
        lines.append(f"| `{record['arm']}` | {record['label']} | {gate.get('exit', '-')} | "
                     f"{counts} | `{report_cell}` |")
    lines += ["", "## Aggregate metrics", "",
              "| metric | " + " | ".join(record["arm"] for record in records) + " |",
              "| --- |" + " --- |" * len(records)]
    for metric in HIGHER_IS_BETTER + LOWER_IS_BETTER:
        row = [f"`{metric}`"]
        for record in records:
            row.append(fmt((record.get("gate") or {}).get("aggregate", {}).get(metric)))
        lines.append("| " + " | ".join(row) + " |")
    for label, key in (("latency mean (s)", "mean_s"), ("latency p50 (s)", "p50_s"),
                       ("latency p95 (s)", "p95_s")):
        row = [f"`{label}`"]
        for record in records:
            row.append(fmt((record.get("gate") or {}).get("latency", {}).get(key), 2))
        lines.append("| " + " | ".join(row) + " |")
    row = ["`ingest (s)`"]
    for record in records:
        row.append(fmt((record.get("reindex") or {}).get("seconds"), 1))
    lines.append("| " + " | ".join(row) + " |")
    row = ["`index items / parents / blurbs`"]
    for record in records:
        state = (record.get("reindex") or {}).get("index_after") or {}
        row.append(f"{state.get('items', '-')} / {state.get('parents', '-')} / "
                   f"{state.get('blurbs', '-')}")
    lines.append("| " + " | ".join(row) + " |")

    for arm, comparison in comparisons.items():
        lines += ["", f"## Verdict: `{arm}`", ""]
        if comparison.get("error"):
            lines.append(f"**NOT MEASURED** - {comparison['error']}")
            continue
        verdict, reasons = comparison["verdict"], comparison["reasons"]
        lines.append(f"**{verdict}**")
        if comparison.get("reference_report"):
            lines.append("")
            lines.append(f"Compared against `{comparison['reference_report']}`.")
        lines.append("")
        for reason in reasons:
            lines.append(f"- {reason}")
        moves = comparison.get("per_query_movements") or []
        if moves:
            lines += ["", "Per-query movement against the same-model reference:", "",
                      "| query | status | movement |", "| --- | --- | --- |"]
            for move in moves:
                deltas = ", ".join(f"{metric} {a} -> {b}"
                                   for metric, (a, b) in move["deltas"].items()) or "-"
                lines.append(f"| `{move['id']}` {move['query']} | {move['status']} | {deltas} |")
        else:
            lines += ["", "No per-query movement on any watched metric."]
    lines += [
        "",
        "## Reading this honestly",
        "",
        "- A `RECOMMEND ON` verdict is a recommendation to a human, not a change: nothing "
        "here edits `config.py`. Flip the default by hand, re-run the gate at the shipped "
        "configuration, and record the new report next to the old one, which is how the "
        "0.5b measurements were kept.",
        "- Generation is not bit-reproducible, at any model size, even at temperature 0 with "
        "a fixed seed. One run of one arm is one sample. If a verdict turns on a single "
        "query, re-run that arm before believing it.",
        "- Latency here was measured with `OLLAMA_KEEP_ALIVE=" + str(args.keep_alive) + "`"
        + (", which unloads the model after every call, so each query pays the load again."
           if str(args.keep_alive) == "0" else
           ", which keeps the model resident between queries.")
        + " Every arm used the same value, so this column is internally comparable. It is "
        "not comparable to the reports quoted in the setup guide, which were taken under a "
        "different policy in a 2 GB sandbox.",
        "- If this experiment ran on a GPU and the deployment target is CPU-only, the "
        "latency column describes a machine the users do not have. The quality columns "
        "still transfer; the latency column does not.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="The Ollama model to measure, e.g. qwen2.5:3b or qwen2.5vl:3b. "
                             "Must already be pulled: this project runs offline.")
    parser.add_argument("--arms", default="reference,contextual,structure",
                        help=f"Comma-separated subset of {', '.join(ARMS)} (default: all "
                             "three). A feature arm without a reference arm needs "
                             "--reference-report.")
    parser.add_argument("--reference-report", default=None,
                        help="Reuse an existing report as the same-model reference instead "
                             "of running the reference arm again.")
    parser.add_argument("--keep-alive", default="10m",
                        help="OLLAMA_KEEP_ALIVE for every arm (default 10m). Identical "
                             "across arms, or the latency column means nothing.")
    parser.add_argument("--num-ctx", type=int, default=None,
                        help="Override OLLAMA_NUM_CTX for every arm. Try 4096 if a 3B model "
                             "does not fit alongside its KV cache.")
    parser.add_argument("--judge", action="store_true",
                        help="Also score answers with the model as an LLM judge. Advisory, "
                             "and roughly four times slower.")
    parser.add_argument("--no-fast", action="store_true",
                        help="Skip the retrieval-only pass after each reindex.")
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="How far a gated metric may move before it counts (default 0.0).")
    parser.add_argument("--plan", action="store_true",
                        help="Print the matrix, the resolved configuration and the cost "
                             "estimate. Touches nothing, needs no model installed.")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Run anyway with uncommitted changes in the tree.")
    parser.add_argument("--yes", action="store_true",
                        help="Do not ask for confirmation before starting.")
    args = parser.parse_args()

    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        say(f"REFUSING: unknown arm(s) {', '.join(unknown)}. Known: {', '.join(ARMS)}.")
        return 2
    if "reference" not in arms and not args.reference_report:
        say("REFUSING: a feature arm compared against the 0.5b baseline would credit or")
        say("blame the feature for the model. Include 'reference' in --arms, or point")
        say("--reference-report at a report taken with this same model and configuration.")
        return 2
    if args.num_ctx is not None and args.num_ctx <= 0:
        say("REFUSING: --num-ctx must be positive.")
        return 2

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = now_stamp()

    if args.plan:
        # A plan should be printable without a model installed, so preflight problems are
        # reported here rather than treated as fatal.
        problems, warnings, facts = preflight(args, args.model)
        facts["warnings"] = warnings
        print_plan(args, arms, facts)
        if warnings:
            say("\n  Warnings:")
            for warning in warnings:
                say(f"    - {warning}")
        if problems:
            say("\n  A real run would REFUSE to start until these are fixed:")
            for problem in problems:
                say(f"    - {problem}")
        say("\n--plan: nothing was touched.")
        return 0

    problems, warnings, facts = preflight(args, args.model)
    facts["warnings"] = warnings
    rule()
    say(f"Big-model experiment: {args.model}, arms {', '.join(arms)}")
    rule()
    say(f"  git {facts.get('git', {}).get('commit')}  platform {facts.get('platform')}  "
        f"disk free {facts.get('disk_free_mb')} MB")
    for warning in warnings:
        say(f"  [warn] {warning}")
    if problems:
        say("\nREFUSING to start:")
        for problem in problems:
            say(f"  - {problem}")
        say("\nNothing was modified. Fix the above, or re-run with --plan to see the matrix.")
        return 2

    print_plan(args, arms, facts)
    if not args.yes:
        try:
            answer = input("\nStart the experiment? The index will be rebuilt several times "
                           "and restored at the end. [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            say("Cancelled; nothing was modified.")
            return 1

    backup = IndexBackup(stamp)
    backup.take()
    records: list[dict] = []
    comparisons: dict[str, dict] = {}
    reference_report: Path | None = (Path(args.reference_report) if args.reference_report
                                     else None)
    reference_retrieval: Path | None = None
    exit_code = 0
    try:
        for arm in arms:
            record = run_arm(arm, args, stamp, facts, reference_report, reference_retrieval)
            records.append(record)
            if arm == "reference":
                if record.get("reference_report_path"):
                    reference_report = Path(record["reference_report_path"])
                if record.get("reference_retrieval_path"):
                    reference_retrieval = Path(record["reference_retrieval_path"])
            if record.get("error"):
                exit_code = 1

        # Paired comparison and verdict for every feature arm.
        for record in records:
            arm = record["arm"]
            if record.get("error"):
                comparisons[arm] = {"error": record["error"], "verdict": "NOT MEASURED",
                                    "reasons": []}
                continue
            if arm == "reference":
                baseline_moves = []
                if BASELINE_PATH.exists():
                    baseline_moves = per_query_movements(
                        load_report(BASELINE_PATH),
                        load_report(Path(record["gate"]["report_path"])))
                comparisons[arm] = {
                    "verdict": "MODEL DECISION",
                    "reasons": [
                        "compared against evaluation/baseline.json, which was taken on "
                        "qwen2.5:0.5b: read this as 'what does the bigger model do', not as "
                        "a feature verdict",
                        "if this arm's gate failed, switching the default model is itself a "
                        "regression and needs its own decision",
                        "this script does not recommend changing the shipped default model: "
                        "that is a separate decision, and it carries its own re-measurement "
                        "of both features afterwards",
                    ],
                    "reference_report": BASELINE_PATH.name if BASELINE_PATH.exists() else None,
                    "per_query_movements": baseline_moves,
                }
                continue
            if reference_report is None or not reference_report.exists():
                comparisons[arm] = {"error": "no same-model reference report available",
                                    "verdict": "NOT MEASURED", "reasons": []}
                continue
            off = load_report(reference_report)
            on = load_report(Path(record["gate"]["report_path"]))
            comparison = compare_arms(off.get("aggregate", {}), on.get("aggregate", {}),
                                      args.tolerance)
            moves = per_query_movements(off, on)
            dropped = [move["id"] for move in moves
                       if move.get("before_status") == "PASS"
                       and move.get("after_status") in ("CHECK", "FAIL", "UNDECIDED")]
            verdict, reasons = verdict_for(comparison, record["gate"]["exit"], moves,
                                           dropped, args.tolerance)
            comparisons[arm] = {
                "verdict": verdict,
                "reasons": reasons,
                "reference_report": reference_report.name,
                "worse": [list(item) for item in comparison["worse"]],
                "better": [list(item) for item in comparison["better"]],
                "unchanged": [list(item) for item in comparison["unchanged"]],
                "absent": comparison["absent"],
                "per_query_movements": moves,
                "ingest_seconds": {
                    "reference": next((r["reindex"]["seconds"] for r in records
                                       if r["arm"] == "reference" and r.get("reindex")), None),
                    arm: record.get("reindex", {}).get("seconds"),
                },
            }
    except KeyboardInterrupt:
        say("\nInterrupted. Restoring the index and writing what was measured so far.")
        exit_code = 130
    finally:
        facts["index_restored"] = backup.restore()
        facts["index_state_after"] = index_state()
        facts["baseline_sha256_after"] = sha256(BASELINE_PATH)

    rule()
    say("Restore checks")
    rule("-")
    say(f"  index restored byte-identical: {facts.get('index_restored')}")
    say(f"  index state now:               {json.dumps(facts.get('index_state_after'))}")
    baseline_ok = facts.get("baseline_sha256") == facts.get("baseline_sha256_after")
    say(f"  baseline.json untouched:       {baseline_ok}")
    if not facts.get("index_restored"):
        say("  THE INDEX DID NOT RESTORE CLEANLY. Run: python scripts/reindex.py --yes")
        exit_code = 1
    if not baseline_ok:
        say("  baseline.json CHANGED. This script never writes it; investigate before")
        say("  trusting any --compare result from this run.")
        exit_code = 1

    json_path, md_path = write_summary(stamp, args.model, args, facts, records, comparisons)
    say()
    rule()
    say("Verdicts")
    rule("-")
    for arm, comparison in comparisons.items():
        say(f"  {arm:<12} {comparison.get('verdict', '?')}")
        for reason in comparison.get("reasons", []):
            say(f"               - {reason}")
    say()
    say(f"  results  {json_path.relative_to(PROJECT_ROOT)}")
    say(f"  summary  {md_path.relative_to(PROJECT_ROOT)}")
    say(f"  logs     {LOG_DIR.relative_to(PROJECT_ROOT)}/")
    say()
    say("  Next step if a feature arm says RECOMMEND ON: flip the default in config.py by")
    say("  hand, re-run `python evaluation.py --compare` at the shipped configuration, and")
    say("  record both reports. This script measures; it does not change what ships.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
