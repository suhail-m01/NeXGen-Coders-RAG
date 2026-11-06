"""Build the Windows demo kit ZIP from this repository.

The kit is the deliverable people actually receive: everything needed to install,
run, demo and verify Evidence AI on a Windows laptop, with no build step and no
Node toolchain. This script exists so that "regenerate the kit" is one command
with an explicit exclusion list, rather than a hand-assembled archive whose
contents nobody can state.

    python scripts/build_kit.py                       # ../deliverables/EvidenceAI_Windows_Kit.zip
    python scripts/build_kit.py --list                # what would go in, and nothing written
    python scripts/build_kit.py --out C:\\kit.zip --top EvidenceAI
    python scripts/build_kit.py --report report_20261105_200057

What is deliberately NOT in the archive:

- `.env` — it holds a real key path and a machine-specific Tesseract location.
  `.env.example` ships instead, and `setup_windows.ps1` copies it.
- `index/`, `history/`, `auth/` and every `*.db` — generated state. The index is
  built on the target machine during setup, which also proves the pipeline works
  on that hardware; shipping an account database would ship credentials.
- `.git`, `__pycache__`, `.venv`, `node_modules` and editor/OS droppings.
- `evaluation/reports/*` except the one report the documentation quotes, so a
  reviewer can check a claimed score against the artefact instead of trusting it.

What is deliberately IN the archive even though it is generated: `webui/dist`, the
built React bundle. Committing and shipping it means a kit works with no Node
installed, which is the point of serving the UI from the API process.

Every archive also carries `KIT_MANIFEST.json`: the commit it was built from, whether
that tree was clean, the sha256 and size of every entry, the reports it ships and the
ones the documentation cites but this machine does not have. `scripts/verify_kit.py`
reads it, and needs nothing but the standard library to do so - the point is that a
kit can be proved complete on a machine that cannot yet run it. An archive that was
truncated in download, edited afterwards, or assembled by hand fails that check
instead of failing three steps later in a way that looks like a bug in the project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

MANIFEST_NAME = "KIT_MANIFEST.json"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directory names skipped anywhere in the tree.
EXCLUDED_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules", ".vite",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".vscode", ".idea",
    "index", "history", "auth", "uploads", ".cache",
    # Paired-arm experiment output: an operator's run on their own machine, and the
    # reports it cites travel separately. Same reasoning as index/ and history/.
    "experiments",
}

# Relative paths (POSIX separators) skipped wholesale.
EXCLUDED_PATHS = {
    ".env",
    ".streamlit/secrets.toml",
    "deliverables",
}

EXCLUDED_SUFFIXES = {".db", ".faiss", ".index", ".pyc", ".pyo", ".log", ".key", ".pem"}
EXCLUDED_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}

DEFAULT_OUT = PROJECT_ROOT.parent / "deliverables" / "EvidenceAI_Windows_Kit.zip"


def is_excluded(relative: Path) -> bool:
    """One place decides what ships, so the rule can be read and argued with."""
    posix = relative.as_posix()
    if posix in EXCLUDED_PATHS:
        return True
    if relative.name in EXCLUDED_NAMES:
        return True
    if relative.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    # .env.local, .env.production ... but never .env.example.
    if relative.name.startswith(".env.") and relative.name != ".env.example":
        return True
    return any(part in EXCLUDED_DIRS for part in relative.parts)


def collect(report_names: set[str]) -> list[Path]:
    """Every file to ship, sorted, with the chosen evaluation reports kept."""
    keep: list[Path] = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(PROJECT_ROOT)
        if relative.parts[0] == "evaluation" and "reports" in relative.parts:
            # Only the report the documentation quotes travels with the kit.
            if relative.stem not in report_names:
                continue
        if is_excluded(relative):
            continue
        keep.append(relative)
    return keep


# Documentation that travels with the kit, and therefore has to be readable offline:
# every report these files quote by name ships too.
DOC_GLOBS = ("*.md", "*.txt")
REPORT_PATTERN = re.compile(r"report_\d{8}_\d{6}(?:_retrieval)?")


def git_info() -> dict:
    """The commit a kit was built from, and whether that tree was clean.

    Recorded in the archive because a kit is a claim about a state of the source, and
    the state is only checkable if it is named. A dirty tree is not refused - a demo
    kit is sometimes built mid-change - but the manifest says so.
    """
    def run(arguments: list[str]) -> str:
        try:
            proc = subprocess.run(["git", *arguments], cwd=str(PROJECT_ROOT),
                                  capture_output=True, text=True, encoding="utf-8",
                                  errors="replace")
        except OSError:
            return ""
        return proc.stdout.strip()

    return {"git_commit": run(["rev-parse", "--short", "HEAD"]) or None,
            "git_dirty": bool(run(["status", "--porcelain"]))}


def build_manifest(files: list[Path], args, report_names: set[str],
                   cited_missing: list[str]) -> dict:
    """The manifest that travels inside the archive."""
    entries = {}
    for relative in files:
        data = (PROJECT_ROOT / relative).read_bytes()
        entries[relative.as_posix()] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    return {
        "kit": "Evidence AI Windows kit",
        "schema": 1,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "built_from": git_info(),
        "top": args.top,
        "entry_count": len(files),
        "reports": sorted(report_names),
        "cited_but_absent": cited_missing,
        "entries": entries,
        "not_shipped": {
            ".env": "machine-specific; copy .env.example and edit it",
            "index/": "generated; scripts/ingest_demo.py builds it from data/",
            "history/ and auth/": "generated; shipping an account database would ship credentials",
            "webui/node_modules/": "not needed; the built bundle in webui/dist/ is committed",
            "git history": "the archive is a snapshot, not a clone",
        },
        "verify": "python scripts/verify_kit.py --zip <this archive>",
    }


def cited_report_stems() -> set[str]:
    """Every report stem the shipped documentation quotes by name.

    The kit's rule is that a number in the documentation can be opened and read rather
    than taken on trust. Deriving the report list from the documents instead of from a
    flag is what makes that rule survive a distracted build: a new paragraph quoting a
    new report puts that report in the archive with no step to remember.
    """
    cited: set[str] = set()
    for pattern in DOC_GLOBS:
        for doc in sorted(PROJECT_ROOT.glob(pattern)):
            try:
                cited.update(REPORT_PATTERN.findall(
                    doc.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                continue
    return cited


def baseline_stem() -> str | None:
    """The report stem evaluation/baseline.json was promoted from, if it can be derived.

    Derived from the baseline's own timestamp rather than hard-coded, so a citation to
    the baseline run can be explained instead of reported as a missing file.
    """
    path = PROJECT_ROOT / "evaluation" / "baseline.json"
    if not path.exists():
        return None
    try:
        stamp = json.loads(path.read_text(encoding="utf-8")).get("timestamp", "")
        when = datetime.fromisoformat(str(stamp))
    except (OSError, ValueError, TypeError):
        return None
    return f"report_{when.strftime('%Y%m%d_%H%M%S')}"


def resolve_report(requested: str | None) -> tuple[set[str], list[str]]:
    """Report stems to include, plus the cited stems that are not on this machine."""
    reports_dir = PROJECT_ROOT / "evaluation" / "reports"
    available = sorted({p.stem for p in reports_dir.glob("report_*.json")}) if reports_dir.exists() else []
    if requested:
        stems = {name.strip() for name in requested.split(",") if name.strip()}
        missing = sorted(st for st in stems if st not in available)
        if missing:
            raise SystemExit(
                f"Report(s) not found in evaluation/reports: {', '.join(missing)}\n"
                f"Available: {', '.join(available[-5:]) or 'none'}"
            )
        return stems, []
    cited = cited_report_stems()
    missing = sorted(stem for stem in cited if stem not in available)
    stems = {stem for stem in cited if stem in available}
    if not stems:
        # Nothing cited and nothing to cite: fall back to the newest run so a first kit
        # still carries the measurement it was built from.
        stems = {available[-1]} if available else set()
    return stems, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"archive to write (default: {DEFAULT_OUT})")
    parser.add_argument("--top", default="EvidenceAI_Windows_Kit",
                        help="folder name inside the archive")
    parser.add_argument("--report", default=None,
                        help="comma-separated report stems to include (default: every "
                             "report the shipped documentation quotes by name, so a "
                             "number in the guide can always be opened and read)")
    parser.add_argument("--list", action="store_true",
                        help="print what would be shipped and write nothing")
    args = parser.parse_args()

    try:
        report_names, cited_missing = resolve_report(args.report)
    except SystemExit as exc:
        print(exc)
        return 2

    files = collect(report_names)
    if not files:
        print("Nothing to ship - is this the project root?", file=sys.stderr)
        return 2

    total = sum((PROJECT_ROOT / f).stat().st_size for f in files)
    print(f"Report(s) included: {', '.join(sorted(report_names)) or 'none'}")
    if cited_missing:
        promoted = baseline_stem()
        print("Cited in the documentation but not in evaluation/reports:")
        for stem in cited_missing:
            note = ""
            if stem == promoted:
                note = "  <- promoted to evaluation/baseline.json, which does ship"
            print(f"  - {stem}{note}")
        print("  A citation with no file behind it is a claim the reader cannot check.")
    print(f"Files to ship: {len(files)}  ({total / 1024 / 1024:.2f} MB uncompressed)")

    if args.list:
        for relative in files:
            print(f"  {relative.as_posix()}")
        return 0

    # Sanity checks that are cheap and catch the two ways a kit goes wrong:
    # a UI that cannot render, and a secret that should not travel.
    required = {"webui/dist/index.html", ".env.example", "README.md",
                "DEMO_RUNBOOK.md", "WINDOWS_SETUP_GUIDE.md", "READ_ME_FIRST.txt",
                "BIG_MODEL_EXPERIMENT.md", "setup_windows.ps1", "start.ps1",
                "requirements-lock.txt", "scripts/selftest.py", "scripts/apitest.py",
                "scripts/build_kit.py", "scripts/verify_kit.py",
                "scripts/experiment_big_model.py", "scripts/reindex.py",
                "scripts/ingest_demo.py", "api.py", "app.py", "config.py",
                "evaluation.py"}
    have = {f.as_posix() for f in files}
    absent = sorted(required - have)
    if absent:
        print(f"Refusing to build: missing {', '.join(absent)}", file=sys.stderr)
        return 2
    leaked = sorted(p for p in have if p == ".env" or p.endswith(".db"))
    if leaked:
        print(f"Refusing to build: secrets or state in the archive: {leaked}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(files, args, report_names, cited_missing)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in files:
            archive.write(PROJECT_ROOT / relative, f"{args.top}/{relative.as_posix()}")
        archive.writestr(f"{args.top}/{MANIFEST_NAME}", json.dumps(manifest, indent=2))

    with zipfile.ZipFile(args.out) as archive:
        bad = archive.testzip()
        count = len(archive.namelist())
    size = args.out.stat().st_size
    built = manifest["built_from"]
    print(f"Wrote {args.out}")
    print(f"  {count} entries, {size / 1024 / 1024:.2f} MB, integrity "
          f"{'OK' if bad is None else 'CORRUPT at ' + str(bad)}")
    print(f"  built from commit {built['git_commit']}"
          f"{' (DIRTY TREE)' if built['git_dirty'] else ' (clean tree)'}; "
          f"{MANIFEST_NAME} carries a sha256 for every entry")
    print(f"  verify with: python scripts/verify_kit.py --zip {args.out}")
    return 0 if bad is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
