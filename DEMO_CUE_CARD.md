# Demo cue card — Evidence AI

One page. Print it. Every timing here was measured on **2026-11-06** against a copy of
`EvidenceAI_Windows_Kit.zip` unpacked into a clean directory, index built from scratch by
`scripts\ingest_demo.py`, model resident, `qwen2.5:0.5b` on CPU. The longer script — what
to install, what to say about each panel, what to do when something goes wrong — is
`DEMO_RUNBOOK.md`. This card is the part you need in your hand.

---

## T-15 minutes, before anyone is watching

| # | Do | Why |
| --- | --- | --- |
| 1 | `.\start.ps1` | API + React UI on :8000, Streamlit on :8501, opens the browser |
| 2 | Log in — or fill the same form to create the first administrator | The bootstrap door closes after one use; do not discover that live |
| 3 | Ask **"What is the top risk carried into 2025?"** and let it finish | **A cold process takes 26 s** for the first answer while the model loads. Every answer after it is 8–13 s. The audience must never see the first one |
| 4 | Leave the tab on **Ask**, zoom 100%, Streamlit open in a second tab | Beat 8 uses the operator UI |
| 5 | If anything looks wrong: `scripts\selftest.py` (258 checks, no model), `scripts\apitest.py` (74), `scripts\dump_chunks.py --all` (6 sources) | All three are offline and take under a minute together |

---

## The seven lines, in order, with what each costs

| # | Ask this, exactly | Expect | Measured |
| --- | --- | --- | --- |
| 1 | *log in, do not ask anything* | role badge next to your name | — |
| 2 | **How many records did the ingestion platform process in 2024?** | "1.2 million records", five citations including `project_report.pdf · p.2` | **11.2 s** |
| 3 | **What does the project dashboard show about team size?** | "6 engineers" from `project_dashboard.png`, OCR confidence ~90 % in the evidence panel | **12.6 s** |
| 4 | **What did the quarterly review recording say about accuracy?** | "99.4 %" cited to `project_meeting.wav · 00:00–00:21`, speaker labels | **12.1 s** |
| 5 | **What was Project Alpha's budget in 2040?** | refused, zero citations — open **Pipeline trace** and read the last two rows aloud | **0.49 s** |
| 6 | **What was Project Beta's team size?** | refused — `entity guard: question asks about Project Beta; evidence only mentions Alpha` | **0.65 s** |
| 7 | **What percentage of the budget was utilised?** | **"87 %"**, one citation: `project_notes.docx · Q4 Review Meeting` | **8.5 s** |

Model time for the whole demo: about **45 seconds**. Speaking time is the rest.

Answers **stream**: the text appears as the model produces it, and the citations, scores
and pipeline trace arrive with the final result. A refusal streams nothing and lands in
half a second, which is the beat worth pointing at — *"it did not start writing and then
change its mind; it never called the model."* For the corpus numbers before the first
question, the **Overview** tab (Alt+5) reads them from the service: sources, indexed
items, relationships, collections, the configured model and whether it is installed, and
the recorded sessions.

**Beat 1, say:** *"Local accounts, PBKDF2-hashed, role-gated, lockout after five failures —
and a viewer never sees the delete button at all, rather than seeing it fail."*

**Beat 2, then click a citation.** The answer prose may carry no inline `[1]` — a small
model often does not emit markers, and this one did not in rehearsal. The **numbered chips
directly under the answer are always there**, and so are the chips on each evidence block.
Click one: the matching block scrolls into view and outlines itself. Click again to
release. Say: *"Interactive citations — the claim and the sentence it came from are one
click apart, in both directions."*

**Beat 3, point at the OCR confidence.** *"Ninety percent confidence, four of five words
kept, and the one unreliable numeric token was withheld rather than guessed. Low
confidence is reported, not laundered."*

**Beat 5, the differentiator.** Read the trace rows out loud:

```
coverage guard: question numbers absent from evidence: 2040
End to end 490 ms   abstained (INSUFFICIENT_EVIDENCE)
```

*"Half a second — the language model was never called. The trace lists every stage,
including the ones that did not run and why. Abstention accuracy is 1.0 on the sixteen
question harness, and this is that number for one question."*

**Beat 6, say:** *"It does not just refuse, it names the constraint that failed. Near-miss
questions are the ones that make RAG systems lie confidently."*

---

## Numbers you may quote

From the report of record on dataset v2.0 (16 questions, `report_20261106_000901`):
abstention accuracy **1.0**, answer correctness **1.0**, forbidden-leak **0.0**,
groundedness **0.9375**, citation accuracy **0.5635**, retrieval precision **0.3531**,
MRR **0.6458**.

**Explain precision before somebody else does.** Five passages are retrieved for every
answer and the harness counts all five, so a correct answer supported by one passage and
four neighbours scores 0.2. Precision here measures how tight the retrieved set is, not
whether the answer is right — correctness and abstention measure that, and both are 1.0.

---

## Three things not to do

1. **Do not ask a question you have not watched it answer.** The guards are tuned to this
   corpus; an untested near-miss is the one thing that can produce a confident wrong
   answer live.
2. **Do not let the first cold answer be seen.** 26 s of spinner reads as a broken demo.
3. **Do not stage a failure.** The guards fire or they do not. A rehearsed stumble reads
   as rehearsed.

---

## If it goes wrong in the room

| Symptom | Say | Then |
| --- | --- | --- |
| An answer takes > 40 s | *"The model unloaded itself — one moment."* | Re-ask; the second answer is fast. `curl 127.0.0.1:11434/api/tags` shows whether Ollama still holds it |
| Refused a question that should answer | *"That is the coverage guard; let me show you why."* | Open the trace and read the decisive row — a refusal with a reason is the feature, not the bug |
| UI blank at `/ui` | *"Serving the bundle from the API — let me check the build."* | `scripts\apitest.py` verifies the hashed asset paths resolve; `scripts\verify_kit.py` proves the archive they came from |
| Asked "is it really offline?" | *"Yes — and it is tested, not claimed."* | `scripts\verify_offline.py` blocks every non-loopback socket and then answers a question |
| Asked what the model is | Name it, and its size | `scripts\manifest.py` prints the exact build; the Health tab says `degraded` and why if the configured model is not the one serving |

---

## The honest limits — say them if asked, never volunteer

* The timings above are `qwen2.5:0.5b` on CPU, the model that fits a sub-4 GB machine and
  the one this kit was measured with. The shipped default is `qwen2.5vl:3b`, which also
  reads images natively instead of falling back to OCR; at 0.5 b the dashboard image is
  OCR-only and says so in the evidence panel.
* Citation accuracy 0.5635 means the answer cites more passages than it strictly needs.
  It does not mean the citations are wrong.
* Slide 11 of the deck OCR'd at 90 % confidence and one numeric token was withheld. That
  is in the ingestion warnings, and it is the behaviour being demonstrated, not a defect
  being hidden.
* Ollama is not bit-reproducible at temperature 0. Five identical runs of the shipped
  configuration produced identical gated metrics, which is strong evidence for this corpus
  and not a claim of determinism.

---

## Provenance

The timings above were measured on 2026-11-06 against `EvidenceAI_Windows_Kit.zip`
unpacked into a clean directory: `.env` copied from `.env.example`, index built from
`data/` by `scripts\ingest_demo.py` in 28 s, seven questions through the ASGI app with
the model resident. In the same tree afterwards: `selftest.py` 258/258, `apitest.py`
78/78 on its first run against that fresh index, `smoke_test.py` 4/4, `verify_offline.py`
PASSED, and `evaluation.py --compare` **16 PASS with 0 gated failures** — every one of the
ten gated metrics identical to the report of record `report_20261106_000901`, with no
per-query drift across the sixteen questions (`report_20261106_011329`, in
`evaluation\reports\`). Mean latency there was 10.69 s against the record's 15.77 s,
because the record ran with `OLLAMA_KEEP_ALIVE=0` and paid a model load on every question;
the card's numbers are the ones a demo actually experiences.

`RECOVERY.md` covers rebuilding that tree from the archive if it is ever lost.
