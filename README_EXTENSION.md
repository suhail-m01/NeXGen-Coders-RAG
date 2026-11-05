# Strict Grounding Enforcement Extension

An incremental extension for the existing NeXGen-Coders-RAG / Evidence AI codebase.

- Removes unsupported nearest-neighbour citation fallback.
- Uses the existing `grounding_audit.py` module to audit sentence-to-citation coverage.
- Abstains if any answer sentence has no citation above the configured threshold.
- Includes a safe patch script, backups, integration guide, and tests.

See `docs/INTEGRATION_GROUNDING_ENFORCEMENT.md` before applying.

Important: citation coverage is a traceability check. It does not establish semantic entailment or prove factual correctness.
