# NeXGen RAG — Strict Grounding Enforcement Extension

## What this extension changes

1. Removes the citation builder's nearest-neighbour fallback. A sentence that does not meet the configured citation similarity threshold receives no citation.
2. Connects the existing `grounding_audit.py` ledger to `rag_pipeline.py`.
3. Fails closed: if any answer sentence has no qualifying citation, the pipeline returns the existing insufficient-evidence abstention response instead of presenting the answer as grounded.
4. Creates backups before changing either source file.

This is a citation-coverage safeguard, **not a proof of semantic entailment or factual truth**. Embedding similarity can still match evidence that does not logically support a claim.

## Apply it

Copy `apply_grounding_enforcement.py` into the root of your NeXGen-Coders-RAG project. Ensure the project root already contains `grounding_audit.py`, `citations.py`, and `rag_pipeline.py`.

From the VS Code PowerShell terminal opened at the project root, run:

```powershell
python .\apply_grounding_enforcement.py
```

Or specify the project folder:

```powershell
python .\apply_grounding_enforcement.py --project "C:\path\to\NeXGen-Coders-RAG"
```

The script checks every expected code block before writing. If the repository differs from the expected version, it stops without changing files.

Backups are created as:
- `citations.py.before_grounding_enforcement.bak`
- `rag_pipeline.py.before_grounding_enforcement.bak`

## Test

The tests use the existing `grounding_audit.py` module:

```powershell
python -m pytest tests\test_grounding_enforcement.py -q
```

Then run the complete suite:

```powershell
python -m pytest -q
```

If pytest is missing:

```powershell
python -m pip install pytest
```

## Review before commit

```powershell
git diff --check
git diff -- citations.py rag_pipeline.py
git status
```

Confirm the changes are limited to the intended files and tests. Do not stage `.env`, API keys, local data, model caches, or generated indexes.

## Roll back

If you need to undo only this patch, restore the backups:

```powershell
Copy-Item .\citations.py.before_grounding_enforcement.bak .\citations.py -Force
Copy-Item .\rag_pipeline.py.before_grounding_enforcement.bak .\rag_pipeline.py -Force
```

## Suggested commit

```powershell
git add citations.py rag_pipeline.py tests/test_grounding_enforcement.py apply_grounding_enforcement.py docs/INTEGRATION_GROUNDING_ENFORCEMENT.md
git commit -m "Enforce sentence-level citation coverage"
git push origin main
```
