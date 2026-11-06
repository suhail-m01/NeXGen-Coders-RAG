# Recovery: how to rebuild this project from nothing

This project has been destroyed twice and rebuilt from its own artefacts both times, so
the procedure below is not theory. It is what was actually run, with the checks that
proved each step worked.

Two files in `deliverables/`, and they are not interchangeable:

| artefact | contains | gets you back |
| --- | --- | --- |
| `EvidenceAI_Windows_Kit.zip` | every file in the tree that is not generated, plus `KIT_MANIFEST.json`: source, documentation, the v2.0 dataset and its 32 mirrors, `evaluation/baseline.json`, the committed React bundle, and every evaluation report the documentation cites by name | a **working tree** |
| `EvidenceAI_repo.bundle` | the git objects, `git bundle create --all` | the **commit history**, including the per-batch narrative a flat archive loses |

Deliberately in neither:

* `.env` — machine-specific. `copy .env.example .env` and edit it.
* `index/` — generated. `scripts\ingest_demo.py` rebuilds it from `data\` in under a
  minute, and building it on the target machine is part of proving the pipeline works
  there. A binary index in an archive is something the reader can neither verify nor
  reproduce.
* `auth/`, `history/` — generated, and shipping an account database would ship
  credentials.
* `webui/node_modules/` — not needed; `webui/dist/` is committed and served by the API.
* The Python, HuggingFace and Ollama caches — rebuilt by `setup_windows.ps1` on Windows,
  by `repair_stack.sh` in the build sandbox.

## 1. Verify before you trust

```powershell
python scripts\verify_kit.py --zip deliverables\EvidenceAI_Windows_Kit.zip
```

Standard library only, imports nothing from this project, runs on a machine that has
Python and nothing else. Ten checks: the CRC, every entry's sha256 and size against the
manifest, that nothing was added after the build, that the files a kit cannot work
without are present, that no secret or generated database travelled, that every asset
`webui/dist/index.html` asks the browser for is inside the archive, and that every report
the shipped documentation quotes is inside it too. Exit 0 means the archive is what
`build_kit.py` wrote; anything else means rebuild it rather than repair it.

## 2. Restore a working tree

```powershell
python scripts\verify_kit.py --zip deliverables\EvidenceAI_Windows_Kit.zip --extract C:\EvidenceAI
cd C:\EvidenceAI
copy .env.example .env          # then edit: model, Tesseract path, index directory
python scripts\selftest.py      # 258 checks, no model, no network, throwaway index
python scripts\ingest_demo.py   # builds index\ from data\
python scripts\apitest.py       # 78 checks on the API and the React UI
python evaluation.py --retrieval-only --compare    # fast gate, ~30 s, no LLM
```

`--extract` re-hashes every file on disk after unpacking and refuses any entry whose name
would write outside the target directory.

## 3. Restore the history as well

```powershell
git clone deliverables\EvidenceAI_repo.bundle EvidenceAI-restored
```

or, into a clone whose objects were lost:

```powershell
git fetch C:\path\EvidenceAI_repo.bundle "refs/*:refs/*"
git reset --hard <commit>
```

`git bundle verify` should say *the bundle records a complete history*. Then re-apply any
file newer than the bundle — the archive and the bundle are only as fresh as the last time
you made a recovery point.

## 4. Make a recovery point

After every commit, in this order, so the manifest names a commit that exists and the
bundle contains it:

```powershell
python scripts\build_kit.py                                        # ZIP + manifest
git bundle create deliverables\EvidenceAI_repo.bundle --all        # history
git bundle verify  deliverables\EvidenceAI_repo.bundle
python scripts\verify_kit.py                                       # prove the ZIP
```

A kit built from a dirty tree is not refused — a demo kit is sometimes built mid-change —
but `KIT_MANIFEST.json` records `git_dirty: true`, and `verify_kit.py` prints it. Prefer
clean.

## How to tell you need this

Symptoms observed in both incidents, in the order they are usually noticed:

* `git log --oneline -1` shows the original upstream clone and none of your commits, and
  `git cat-file -t <a commit you made>` says *not a valid object name*.
* `ModuleNotFoundError: No module named 'pydantic_core._pydantic_core'` — the compiled
  extensions are gone even though the package directory is there.
* `evaluation\reports\` holds one old report instead of the fifteen the documentation
  cites.
* `webui\dist\` and `webui\src\` are missing, or `evaluation\dataset\queries.json` has six
  queries instead of sixteen.

The working tree usually survives; the version control, the gitignored outputs and the
installed binaries usually do not.

## The two incidents, and what each one cost

| lost | recovered by | proved by |
| --- | --- | --- |
| every commit object (twice) | not recoverable the first time; the bundle covers it now | `git bundle verify`: *records a complete history* |
| 15 evaluation reports, 45 files | the ZIP | sha256 per entry, `verify_kit.py` |
| dataset v2.0: `queries.json` + 32 mirrors | the ZIP | 16 queries, `dataset_version 2.0`, and a gate run that reproduced the report of record exactly |
| `webui/src` and the committed `webui/dist` | the ZIP | `/ui` serves, and `apitest.py` checks the hashed asset paths resolve |
| Python extensions, HF cache, Ollama engine | `repair_stack.sh` | `selftest.py` 258/258, `apitest.py` 78/78, a full gate at 16 PASS |
| `index/`, `baseline.json`, `auth/`, `history/` | never lost | hash before equals hash after |

The restored tree was then installed as a stranger would install it - unpacked into a
clean directory, `.env` copied from `.env.example`, index built from `data/` in 28 s -
and reproduced the report of record exactly: sixteen PASS, no gated failure, all ten
gated metrics identical and no per-query drift across the sixteen questions
(`report_20261106_011329`, shipped in the archive). That is the check that the recovery
gave back a working project rather than a plausible-looking directory tree.

Two rules came out of it, and both are now enforced rather than remembered:

1. **A restore must take a protect-list.** The second incident happened while
   `scripts/build_kit.py` had uncommitted edits *newer than the archive*. A restore that
   overwrites everything would have destroyed the newest work in the tree while "fixing"
   it. The recovery that was actually run restored 86 files, skipped that one, and printed
   what it skipped.
2. **Commit, then archive, then bundle.** In that order the manifest names a commit that
   exists, the archive matches the tree, and the bundle contains both. Out of that order
   you get an archive that says `git_dirty: true` and a bundle that is missing the commit
   the archive was built from — which is recoverable, but only by knowing to look.

The reports taken before the first incident name commit hashes that no longer resolve
(`08856ef`, `9c49fc8`, `605dcf4` and the rest). The measurements are still valid: they
describe this tree, which is the tree those commits described. `WINDOWS_SETUP_GUIDE.md`
says so where the numbers are quoted, and `BIG_MODEL_EXPERIMENT.md` §9 repeats it, because
a reader who opens one of those JSON files will notice.
