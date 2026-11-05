# Evidence AI — Grounding Coverage Upgrade

This package adds an optional sentence-level Grounding Coverage Ledger to the
existing offline multimodal RAG project.

## Package contents

```text
grounding_audit.py
tests/
  test_grounding_audit.py
docs/
  INTEGRATION.md
```

## Quick start

1. Extract this ZIP into a temporary folder.
2. Copy `grounding_audit.py` to the root of your Evidence AI repository.
3. Copy `tests/test_grounding_audit.py` into its `tests/` directory.
4. Follow `docs/INTEGRATION.md` to add the strict citation gate to
   `rag_pipeline.py` and remove the weak nearest-evidence citation fallback
   from `citations.py`.
5. Run `python -m pytest tests/test_grounding_audit.py -q`.

The integration is deliberately documented as a small manual patch so the
existing pipeline is not overwritten by a stale copy of a large source file.

## Scope and honesty

The standalone module and tests are included. The full application integration
is not applied automatically and has not been run against your local Windows
environment. Run the tests and existing application checks before claiming
the integrated feature is verified.
