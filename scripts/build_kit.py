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
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directory names skipped anywhere in the tree.
EXCLUDED_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules", ".vite",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".vscode", ".idea",
    "index", "history", "auth", "uploads", ".cache",
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


def resolve_report(requested: str | None) -> set[str]:
    """Report stem(s) to include: the requested one, else the newest present."""
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
        return stems
    return {available[-1]} if available else set()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"archive to write (default: {DEFAULT_OUT})")
    parser.add_argument("--top", default="EvidenceAI_Windows_Kit",
                        help="folder name inside the archive")
    parser.add_argument("--report", default=None,
                        help="comma-separated report stems to include "
                             "(default: the newest report in evaluation/reports)")
    parser.add_argument("--list", action="store_true",
                        help="print what would be shipped and write nothing")
    args = parser.parse_args()

    try:
        report_names = resolve_report(args.report)
    except SystemExit as exc:
        print(exc)
        return 2

    files = collect(report_names)
    if not files:
        print("Nothing to ship - is this the project root?", file=sys.stderr)
        return 2

    total = sum((PROJECT_ROOT / f).stat().st_size for f in files)
    print(f"Report(s) included: {', '.join(sorted(report_names)) or 'none'}")
    print(f"Files to ship: {len(files)}  ({total / 1024 / 1024:.2f} MB uncompressed)")

    if args.list:
        for relative in files:
            print(f"  {relative.as_posix()}")
        return 0

    # Sanity checks that are cheap and catch the two ways a kit goes wrong:
    # a UI that cannot render, and a secret that should not travel.
    required = {"webui/dist/index.html", ".env.example", "README.md",
                "DEMO_RUNBOOK.md", "WINDOWS_SETUP_GUIDE.md", "READ_ME_FIRST.txt",
                "setup_windows.ps1", "start.ps1", "requirements-lock.txt",
                "scripts/selftest.py", "api.py", "app.py"}
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
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in files:
            archive.write(PROJECT_ROOT / relative, f"{args.top}/{relative.as_posix()}")

    with zipfile.ZipFile(args.out) as archive:
        bad = archive.testzip()
        count = len(archive.namelist())
    size = args.out.stat().st_size
    print(f"Wrote {args.out}")
    print(f"  {count} entries, {size / 1024 / 1024:.2f} MB, integrity "
          f"{'OK' if bad is None else 'CORRUPT at ' + str(bad)}")
    return 0 if bad is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
