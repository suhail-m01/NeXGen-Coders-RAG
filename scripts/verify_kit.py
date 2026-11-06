"""
verify_kit.py - prove a Windows kit archive is complete, unmodified and self-consistent.

    python scripts\\verify_kit.py                          # the default deliverable path
    python scripts\\verify_kit.py --zip C:\\Downloads\\EvidenceAI_Windows_Kit.zip
    python scripts\\verify_kit.py --zip kit.zip --extract C:\\EvidenceAI
    python scripts\\verify_kit.py --zip kit.zip --list      # the manifest, nothing checked

Standard library only, on purpose. This is the first thing somebody should run on a
machine that has nothing but Python installed - before the virtual environment, before
Ollama, before Tesseract - because "download the kit" is a step that can silently
truncate, and every later failure would otherwise look like a bug in the project. It
imports nothing from this repository, so it can verify an archive of a project it cannot
run.

What it checks:

  1. the archive opens and passes zipfile's own CRC test
  2. it carries the manifest build_kit.py wrote into it
  3. every file the manifest lists is present, with the sha256 and size it was built with
  4. every file in the archive is listed - nothing was added afterwards
  5. the files a kit cannot work without are there (the built React bundle, .env.example,
     the guides, selftest, the API and the Streamlit app)
  6. nothing that must not travel is there: no .env, no *.db, no index/, history/, auth/,
     no .git, no node_modules
  7. every asset webui/dist/index.html asks the browser for is inside the archive - the
     classic broken kit, where the HTML loads and the hashed JS file does not exist
  8. every evaluation report the shipped documentation quotes by name is inside the
     archive, so a reader can check a claimed score against the artefact
  9. with --extract, the unpacked files are re-hashed on disk, and any entry whose name
     would escape the target directory is refused before it is written

Exit 0 when every check passes, 1 when one fails, 2 when the archive cannot be read at
all. The last thing it prints is how to rebuild a kit that fails: scripts\\build_kit.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

MANIFEST_NAME = "KIT_MANIFEST.json"

# A kit without these does not install, run or demo. Same list build_kit.py refuses to
# build without, repeated here because the verifier must not import the builder: it runs
# on machines where the project's dependencies do not exist.
REQUIRED = (
    "webui/dist/index.html", ".env.example", "README.md", "DEMO_RUNBOOK.md",
    "WINDOWS_SETUP_GUIDE.md", "READ_ME_FIRST.txt", "setup_windows.ps1", "start.ps1",
    "requirements-lock.txt", "scripts/selftest.py", "scripts/build_kit.py",
    "scripts/reindex.py", "scripts/ingest_demo.py", "scripts/apitest.py",
    "scripts/verify_kit.py",
    "scripts/experiment_big_model.py", "api.py", "app.py", "config.py",
    "evaluation.py", "BIG_MODEL_EXPERIMENT.md",
)

# Generated state, secrets and vendored trees. A kit that carries any of these is either
# leaking something or lying about being reproducible from source.
FORBIDDEN_SUFFIXES = (".db", ".faiss", ".index", ".key", ".pem", ".sqlite", ".sqlite3")
FORBIDDEN_NAMES = (".env", "secrets.toml")
FORBIDDEN_DIRS = ("index", "history", "auth", ".git", "node_modules", ".venv", "venv",
                  "__pycache__", "uploads", "experiments")

# Documentation that travels with the kit, scanned for report citations.
DOC_SUFFIXES = (".md", ".txt")
REPORT_PATTERN = re.compile(r"report_\d{8}_\d{6}(?:_retrieval)?")

# Vite emits <script type="module" crossorigin src="/assets/index-abc123.js"> and a
# matching stylesheet link. Both must be in the archive or the UI loads a blank page.
ASSET_PATTERN = re.compile(r"""(?:src|href)=["']([^"']+\.(?:js|css))["']""")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def top_level(names: list[str]) -> str | None:
    """The single folder every entry lives under, or None.

    Requires *every* entry to sit under that folder. An archive that mixes root-level
    files with a wrapper folder returns None and is then read as flat, which is the safe
    reading: guessing a prefix here would silently drop the root-level entries from every
    check below, and a verifier that quietly examines fewer files than the archive holds
    is worse than no verifier.
    """
    if not names or "/" not in names[0]:
        return None
    first = names[0].split("/")[0]
    return first if all(name.startswith(first + "/") for name in names) else None


def relative_names(names: list[str], top: str | None) -> list[str]:
    """Archive entry names with the wrapper folder removed, directories dropped."""
    prefix = f"{top}/" if top else ""
    return sorted(name[len(prefix):] for name in names
                  if not name.endswith("/") and name.startswith(prefix))


def asset_references(html: str) -> set[str]:
    """Local assets an index.html asks the browser for, as archive-relative paths."""
    found = set()
    for reference in ASSET_PATTERN.findall(html):
        if reference.startswith(("http://", "https://", "//", "data:")):
            continue
        found.add(reference.lstrip("./").lstrip("/"))
    return found


def forbidden_entries(names: list[str]) -> list[str]:
    """Archive entries that must never travel in a kit."""
    bad = []
    for name in names:
        parts = name.split("/")
        base = parts[-1]
        if base in FORBIDDEN_NAMES:
            bad.append(name)
        elif name.lower().endswith(FORBIDDEN_SUFFIXES):
            bad.append(name)
        elif any(part in FORBIDDEN_DIRS for part in parts[:-1]):
            bad.append(name)
    return sorted(bad)


def cited_reports(doc_texts: dict[str, str]) -> set[str]:
    """Every report stem the shipped documentation quotes by name."""
    cited: set[str] = set()
    for text in doc_texts.values():
        cited.update(REPORT_PATTERN.findall(text))
    return cited


def compare_manifest(manifest: dict, digests: dict[str, tuple[str, int]]) -> dict:
    """Diff a manifest against what the archive actually holds.

    `digests` maps relative name -> (sha256, size). Returns the three ways an archive and
    its manifest can disagree, because each one means something different: a missing file
    means the archive is incomplete, a mismatch means it was modified or truncated, and an
    unaccounted file means something was added after the kit was built.
    """
    entries = manifest.get("entries") or {}
    missing, mismatched = [], []
    for name, record in sorted(entries.items()):
        actual = digests.get(name)
        if actual is None:
            missing.append(name)
            continue
        expected = (str(record.get("sha256", "")), int(record.get("bytes", -1)))
        if actual != expected:
            mismatched.append({"name": name, "expected": expected, "actual": actual})
    unaccounted = sorted(set(digests) - set(entries) - {MANIFEST_NAME})
    return {"missing": missing, "mismatched": mismatched, "unaccounted": unaccounted}


def unsafe_destination(name: str, target: Path) -> bool:
    """True when unpacking `name` would write outside `target` (zip-slip)."""
    if name.startswith("/") or ".." in Path(name).parts:
        return True
    try:
        (target / name).resolve().relative_to(target.resolve())
    except ValueError:
        return True
    return False


CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"\n           {detail}" if detail and not ok else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    default_zip = Path(__file__).resolve().parent.parent.parent / "deliverables" \
        / "EvidenceAI_Windows_Kit.zip"
    parser.add_argument("--zip", type=Path, default=default_zip,
                        help=f"archive to verify (default: {default_zip})")
    parser.add_argument("--extract", type=Path, default=None,
                        help="also unpack here and re-verify every file on disk")
    parser.add_argument("--list", action="store_true",
                        help="print the manifest and check nothing")
    args = parser.parse_args()

    if not args.zip.exists():
        print(f"Cannot read {args.zip}")
        print("Build one with: python scripts/build_kit.py")
        return 2

    print(f"Verifying {args.zip}")
    print("=" * 74)
    try:
        archive = zipfile.ZipFile(args.zip)
    except zipfile.BadZipFile as exc:
        print(f"  [FAIL] the archive is not a readable zip: {exc}")
        return 2

    with archive:
        corrupt = archive.testzip()
        check("zipfile CRC test", corrupt is None, f"corrupt entry: {corrupt}")
        names = [n for n in archive.namelist() if not n.endswith("/")]
        top = top_level(archive.namelist())
        relative = relative_names(names, top)
        print(f"  ....  {len(relative)} entries"
              + (f" under {top}/" if top else " (flat archive)"))

        if MANIFEST_NAME not in relative:
            check(f"{MANIFEST_NAME} is present", False,
                  "this archive was built before the manifest existed, or built by hand; "
                  "rebuild it with scripts/build_kit.py")
            return 1
        check(f"{MANIFEST_NAME} is present", True)

        try:
            manifest = json.loads(archive.read(f"{top}/{MANIFEST_NAME}" if top
                                               else MANIFEST_NAME).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            check("the manifest is readable JSON", False, str(exc))
            return 1
        check("the manifest is readable JSON", True)

        if args.list:
            print(json.dumps(manifest, indent=2)[:4000])
            return 0

        built = manifest.get("built_from") or {}
        tree = "dirty tree" if built.get("git_dirty") else "clean tree"
        print(f"  ....  built {manifest.get('built_at', '?')} from commit "
              f"{built.get('git_commit', '?')} ({tree})")

        digests = {}
        for name in relative:
            if name == MANIFEST_NAME:
                continue
            data = archive.read(f"{top}/{name}" if top else name)
            digests[name] = (sha256_bytes(data), len(data))
        result = compare_manifest(manifest, digests)
        check("every file the manifest lists is in the archive",
              not result["missing"], ", ".join(result["missing"][:8]))
        check("every listed file matches its sha256 and size",
              not result["mismatched"],
              "; ".join(f"{m['name']} expected {m['expected'][0][:12]} got {m['actual'][0][:12]}"
                        for m in result["mismatched"][:5]))
        check("the archive holds nothing the manifest does not list",
              not result["unaccounted"], ", ".join(result["unaccounted"][:8]))

        absent = sorted(name for name in REQUIRED if name not in digests)
        check("every file a kit cannot work without is present", not absent,
              ", ".join(absent))

        leaked = forbidden_entries(relative)
        check("no secrets, databases or generated state travelled", not leaked,
              ", ".join(leaked[:8]))

        html_name = "webui/dist/index.html"
        if html_name in digests:
            html = archive.read(f"{top}/{html_name}" if top else html_name).decode(
                "utf-8", errors="replace")
            wanted = asset_references(html)
            # Vite writes /assets/... with a leading slash; the archive stores
            # webui/dist/assets/... so both spellings are accepted.
            present = set(digests)
            missing_assets = []
            for asset in sorted(wanted):
                candidates = {f"webui/dist/{asset}", asset,
                              f"webui/dist/{Path(asset).name}"}
                if not candidates & present:
                    missing_assets.append(asset)
            check("every asset the built UI asks the browser for is in the archive",
                  not missing_assets,
                  "the page would load and render nothing: " + ", ".join(missing_assets))

        docs = {}
        for name in relative:
            if name.lower().endswith(DOC_SUFFIXES) and "/" not in name:
                docs[name] = archive.read(f"{top}/{name}" if top else name).decode(
                    "utf-8", errors="replace")
        cited = cited_reports(docs)
        shipped = {name.split("/")[-1].rsplit(".", 1)[0] for name in relative
                   if name.startswith("evaluation/reports/")}
        uncited = sorted(cited - shipped)
        check("every report the documentation quotes is in the archive", not uncited,
              "a citation with no file behind it is a claim the reader cannot check: "
              + ", ".join(uncited))
        print(f"  ....  {len(shipped)} reports shipped, {len(cited)} cited by the docs")

    if args.extract:
        print(f"\nExtracting to {args.extract}")
        print("-" * 74)
        target = args.extract
        target.mkdir(parents=True, exist_ok=True)
        refused, written, drifted = [], 0, []
        with zipfile.ZipFile(args.zip) as archive:
            top = top_level(archive.namelist())
            for name in archive.namelist():
                if name.endswith("/"):
                    continue
                relative_name = name[len(top) + 1:] if top else name
                if unsafe_destination(relative_name, target):
                    refused.append(name)
                    continue
                destination = target / relative_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(name))
                written += 1
        check("no entry would write outside the target directory", not refused,
              "zip-slip: " + ", ".join(refused[:5]))
        with zipfile.ZipFile(args.zip) as archive:
            manifest = json.loads(archive.read(f"{top}/{MANIFEST_NAME}" if top
                                               else MANIFEST_NAME).decode("utf-8"))
        for name, record in sorted((manifest.get("entries") or {}).items()):
            path = target / name
            if not path.exists():
                drifted.append(f"{name} (absent)")
            elif sha256_bytes(path.read_bytes()) != record.get("sha256"):
                drifted.append(f"{name} (hash differs on disk)")
        check("every extracted file still matches the manifest on disk", not drifted,
              "; ".join(drifted[:5]))
        print(f"  ....  {written} files written")
        print(f"\nNext: cd {target} and follow READ_ME_FIRST.txt")

    print()
    failed = [entry for entry in CHECKS if not entry[1]]
    print("=" * 74)
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("A kit that fails any of these should be rebuilt rather than repaired:")
        print("  python scripts/build_kit.py")
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
