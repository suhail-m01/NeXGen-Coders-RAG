"""
config.py — single source of truth for every tunable in Evidence AI.

Values resolve in this order: environment variable -> .env file -> default
below. Import `settings` anywhere; it is constructed once at import time.

Nothing here contacts the network. The two HF_* environment variables are
set before any HuggingFace import so that model loading stays local and,
on Windows without Developer Mode, does not attempt unprivileged symlinks.
"""
from __future__ import annotations

import os
from pathlib import Path

# Must be set before huggingface_hub / sentence_transformers are imported
# anywhere in the process. Windows without Developer Mode cannot create
# symlinks (WinError 1314), so the hub is told to copy blobs instead.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent


def _bridge_streamlit_secrets() -> None:
    """Copy Streamlit Cloud secrets into the environment.

    Hosted Streamlit supplies secrets through `st.secrets`, which
    pydantic-settings does not read. Bridging them here means the same
    Settings class works locally (.env) and hosted (platform secrets) with
    no code change. Existing environment variables always win, and nothing
    is logged -- only key names are ever touched.
    """
    try:
        import streamlit as st

        for name, value in st.secrets.items():
            if isinstance(value, str) and name not in os.environ:
                os.environ[name] = value
    except Exception:
        # Not running under Streamlit, or no secrets configured. Both fine.
        pass


_bridge_streamlit_secrets()


class Settings(BaseSettings):
    """Runtime configuration. Every field is overridable from .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Inference backend ---------------------------------------------
    # "offline" -> local Ollama;  "online" -> cloud API.
    # This is the backend the UI starts on; the user can switch at runtime.
    inference_backend: str = "offline"

    # ---- Online (cloud) inference ---------------------------------------
    # Read from the environment / .env only. Never hard-code a key here.
    openai_api_key: str = ""
    online_model: str = "gpt-4o-mini"
    # Any OpenAI-compatible endpoint works (Azure OpenAI, OpenRouter, a
    # self-hosted gateway) without touching provider code.
    online_base_url: str = "https://api.openai.com/v1"
    online_timeout: int = 120
    online_model_supports_vision: bool = True
    # Named in user-facing setup messages so they know exactly what to set.
    online_api_key_env: str = "OPENAI_API_KEY"

    # ---- Local LLM (Ollama) ------------------------------------------
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5vl:3b"
    ollama_timeout: int = 300
    # Keeps the model resident between queries. Ollama's 5-minute default
    # causes a measured ~14s reload penalty mid-demo.
    ollama_keep_alive: str = "30m"
    # One image costs ~1030 prompt tokens; the 4096 default leaves too
    # little room for text evidence alongside it.
    ollama_num_ctx: int = 8192
    # Hard cap on generated tokens for an answer. Measured, not cosmetic: with no
    # cap a 0.5b model that falls into a repetition loop keeps generating until
    # the context fills, which on 2 CPUs took over five minutes and made Ollama
    # itself answer 500 after its own write timeout -- one bad query could stall
    # a whole demo. Every answer this system produces is far shorter than this.
    ollama_num_predict: int = 512
    ollama_temperature: float = 0.0
    # A fixed seed makes answers reproducible run-to-run. Essential for an
    # evidence tool and for comparing evaluation reports. Set <= 0 to let the
    # server pick (non-reproducible sampling).
    ollama_seed: int = 42

    # ---- Local models -------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    whisper_model_size: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    # Local speaker diarization (MFCC embeddings + clustering). Adds speaker
    # labels to audio segments; degrades gracefully to plain timestamps when
    # librosa/scikit-learn are unavailable.
    diarization_enabled: bool = True
    diarization_max_speakers: int = 6
    # 4GB VRAM is fully committed to Ollama; PyTorch stays on CPU.
    torch_device: str = "cpu"

    # ---- Chunking -----------------------------------------------------
    # "fixed" splits on a character window with overlap (the original behaviour).
    # "structure" splits on headings, paragraphs and table rows and keeps the
    # heading path with each chunk (chunking.py). chunk_size is a cap in both
    # modes; chunk_overlap only applies where a block had to be cut, because
    # structure-aware chunking already breaks at seams.
    #
    # Default is "fixed" on measurement, not preference -- and the measurement has
    # now been made twice. The known fix for structure-aware chunking is built:
    # parents.py collapses retrieved chunks into their complete section
    # (small-to-big), the per-source fan-out is wider under "structure", and the
    # fragment floor is 200 characters instead of 40. With all of that in place
    # structure mode retrieves clearly better than fixed -- precision .3635 vs
    # .3115, MRR .7031 vs .6146, citation accuracy .7323 vs .4135 -- and still
    # loses the gate at 13/16 PASS against 16/16: a question whose answer is
    # absent from the corpus was answered "87%" instead of refused, the deck
    # fragments behind one paper question scored -5.5 (inside the calibrated -6.0
    # abstention threshold, so no refusal) and the model then denied the
    # limitation exists, and cross-modal recall fell .75 -> .50.
    # The lesson is that richer context helps a small model find a fact and helps
    # it confabulate at the same time, so structure mode ships opt-in rather than
    # as the default: set CHUNKING_MODE=structure for a long, well-headed corpus
    # asked lookup questions, and re-measure with evaluation.py --compare.
    # Changing this changes how every document splits: run scripts/reindex.py
    # afterwards, since ingestion skips already-indexed files by content hash.
    chunking_mode: str = "fixed"
    chunk_size: int = 800
    chunk_overlap: int = 100
    min_chunk_chars: int = 40
    # Structure-aware chunking needs a much higher floor. Measured on the demo
    # deck: with the shared floor of 40 a 22-page slide deck produced 63 items
    # with a median of 85 characters, and two queries failed because of it -- the
    # title slide split so the university name reached the model without its
    # label (it invented "UC Berkeley"), and the limitations paragraph split so
    # no fragment scored above the abstention threshold (it refused). A fragment
    # that small carries a heading but not the sentence that answers anything.
    structure_min_chunk_chars: int = 200

    # ---- PDF tables ---------------------------------------------------
    # Lift a detected table out of a PDF page and index it as a markdown grid
    # instead of leaving it to plain text extraction. Plain extraction reads a
    # table in block order, which on a real page scatters one row across several
    # blocks: on the demo deck's literature survey the number, the paper title
    # and that row's findings land in different chunks, so a question about one
    # row retrieves a fragment of it. The grid keeps a row together.
    #
    # On by default because it was measured, not because it sounds good. Same
    # corpus, same 16 queries, flag off then on: retrieval precision .3115 ->
    # .3531, citation accuracy .551 -> .5635, and Q16 (asked of the slide deck)
    # went from retrieving three items for one answer to retrieving exactly the
    # right one, P .33 -> 1.00 with its citation .80 -> 1.00. Everything else was
    # unchanged, and the run held 16 PASS / 0 CHECK. It also retired the last
    # reported-only per-query regression in the gate.
    #
    # The bounded risk, stated plainly: PyMuPDF's detector is a layout heuristic,
    # so on a scanned or heavily ruled page it can call prose a table and move real
    # sentences into a grid. Three size floors below reject the small cases, and
    # words are claimed one cell at a time and handed back whenever a candidate is
    # rejected, so nothing is withheld from the page -- a mis-detected region reads
    # worse, it does not read missing. `scripts/selftest.py` asserts that property
    # word by word against the shipped deck. Set this to false and run
    # scripts/reindex.py to go back to plain page text.
    pdf_table_extraction_enabled: bool = True
    # A detected region has to look like a table before it is treated as one.
    # These floors are what keeps a two-column paragraph from being lifted out of
    # the prose: it comes back as 2x2 with two filled cells and stays where it is.
    pdf_table_min_rows: int = 2
    pdf_table_min_cols: int = 2
    pdf_table_min_cells: int = 3

    # ---- Contextual retrieval -----------------------------------------
    # One local LLM call per chunk at ingest writes a sentence or two saying where
    # the chunk sits in its document, and that sentence is prepended to the text
    # that gets embedded and lexically indexed (contextual.py). Anthropic's
    # contextual retrieval, September 2024.
    #
    # Off by default, and the decision is measured on this corpus rather than
    # argued: see §10 of WINDOWS_SETUP_GUIDE.md for both evaluation runs. The blurb
    # never reaches a screen -- `ContentItem.content` is untouched, and citations,
    # excerpts and every guard read `content` -- so the worst a bad blurb can do is
    # mis-rank retrieval. What it always costs is one generation per chunk at
    # ingest, which is why this is a per-corpus decision and not a free switch.
    contextual_retrieval_enabled: bool = False
    # How much of the document is sent with each chunk. It has to fit the model's
    # window beside the chunk and the reply: 12,000 characters is roughly 3,000
    # tokens, which leaves room in a 4,096-token context and is a quarter of an
    # 8,192-token one. Head and tail are kept, not just the head.
    contextual_document_chars: int = 12000
    # Cap on the reply. A blurb is a sentence or two; an uncapped small model asked
    # for one will keep going.
    contextual_max_tokens: int = 60
    # Cap on what is stored and indexed, cut at a word boundary.
    contextual_max_chars: int = 300
    # A source with fewer items than this is skipped: one chunk is the whole
    # document, so there is nothing to situate it in and the blurb could only
    # paraphrase the text it is attached to.
    contextual_min_items: int = 2

    # ---- Retrieval ----------------------------------------------------
    top_k: int = 10
    rerank_top_k: int = 5
    # Hybrid lexical leg: BM25 fused with the dense ranking via RRF. Catches
    # exact tokens (IDs, percentages, names) that semantic search smooths over.
    hybrid_enabled: bool = True
    hybrid_rrf_k: int = 60
    hybrid_lexical_k: int = 20
    # Length normalisation: the cross-encoder mildly prefers long chunks, so
    # chunks above the threshold take a fixed penalty before ranking.
    rerank_length_norm: bool = True
    rerank_length_threshold_chars: int = 600
    rerank_length_penalty: float = 0.2
    # MMR diversity re-selection over an over-fetched rerank pool. The per-source
    # cap keeps one document from crowding out other modalities.
    mmr_enabled: bool = True
    mmr_lambda: float = 0.7
    mmr_max_per_source: int = 2
    # Structure-aware chunking cuts a table into one chunk per row, so a cap of
    # two rows per file can leave the row that answers the question behind. Under
    # "structure" the cap is wider, and parents.py then collapses the selected
    # chunks into complete sections, so the extra chunks cost context only when
    # they come from different sections.
    mmr_max_per_source_structure: int = 4
    # Small-to-big retrieval: build and expand parent sections. Inert in fixed
    # chunking, where no parent rows exist.
    parent_expansion_enabled: bool = True
    parent_max_chars: int = 1800

    # ---- Relationships ------------------------------------------------
    related_to_threshold: float = 0.70
    max_relationship_expansion: int = 3

    # ---- Generation context -------------------------------------------
    max_context_items: int = 5
    max_context_chars: int = 6000
    max_excerpt_chars: int = 400
    # At most one image may be passed to the vision model per query.
    max_inspect_images: int = 1

    # ---- Grounding / abstention ----------------------------------------
    # Cross-encoder logit below which evidence is considered unsupportive.
    abstention_threshold: float = -6.0
    # OCR-derived image evidence scores ~6 logits lower than text evidence, so
    # it needs its own (lower) threshold to be judged on its own scale.
    abstention_threshold_image: float = -9.0
    # Deterministic coverage guard: numbers/years the question commits to must
    # appear in the evidence. Years only by default; all-numbers mode can
    # refuse legitimate rephrasings ("1.2 million" vs "1200000").
    coverage_guard_enabled: bool = True
    # Stream answer tokens into the UI instead of one batched block.
    streaming_enabled: bool = True
    coverage_guard_all_numbers: bool = False
    # Numeric verification guard (post-generation). Any figure in the answer
    # that appears nowhere in the retrieved evidence gets a visible caveat.
    # It never deletes or refuses an answer -- a false positive costs one
    # sentence, a silently wrong number costs trust.
    numeric_guard_enabled: bool = True
    # Cut a looping generation at its first duplicated paragraph. A 0.5b model at
    # temperature 0 lands in repetition loops on some calls; see repetition_guard.
    repetition_guard_enabled: bool = True
    # Question-constraint guards (constraint_guard.py). The score gate cannot see
    # a near-miss period or a renamed entity: both retrieve at high confidence
    # and both produce a fluent, grounded, wrong answer.
    entity_guard_enabled: bool = True      # "Project Beta" vs a corpus of Alpha
    temporal_guard_enabled: bool = True    # 2023 asked, 2024's figures answered
    period_targeting_enabled: bool = True  # show period-matching evidence first
    # Evidence-completion aid for broad multi-source questions: appends the
    # corroborated figures the answer omitted, each labelled with the files that
    # state it. A prompt-level version of this was measured first and produced a
    # byte-identical answer at qwen2.5:0.5b, so the aid is applied to the answer
    # instead of the prompt. Fires only on synthesis phrasing (see
    # synthesis.SYNTHESIS_MARKERS), never on a lookup question.
    synthesis_completion_enabled: bool = True
    # Figures shorter than this are ignored ("3 items", "page 5"): short
    # numbers are too easy to paraphrase legitimately to police.
    numeric_guard_min_digits: int = 2
    # Cosine similarity required to attribute an answer sentence to evidence.
    citation_threshold: float = 0.35

    # ---- OCR quality ---------------------------------------------------
    # Tesseract reports a confidence per word. Keeping only trustworthy words
    # and telling the LLM when the rest is guesswork is the difference between
    # "the dashboard shows 22,024" and "OCR could not read this reliably".
    ocr_confidence_enabled: bool = True
    # Prose words below this Tesseract confidence (0-100) are dropped.
    ocr_min_word_confidence: float = 40.0
    # Tokens containing a digit need more. Measured on the demo dashboard:
    # prose scored 86-94 while digits averaged 52.6, and the digit that caused
    # a confident hallucination ("22024", the team size 22 fused with the year
    # 2024) scored 51. A misread letter reads as noise; a misread digit
    # silently becomes a different fact, so it is held to a higher bar.
    ocr_min_digit_confidence: float = 60.0
    # Mean confidence of the kept words below which the passage is annotated
    # as unreliable and reported to the user as an ingestion warning.
    ocr_low_confidence_threshold: float = 60.0
    # Tesseract's sweet spot is roughly a 30px cap height. Screenshots and
    # dashboards are usually rendered well below it: on the demo dashboard,
    # OCR at native size scored a 63.6% mean confidence and fused the team
    # size into a nonexistent "22024", while a 2x upscale scored 90.5% and
    # read the real values ("6" engineers, 320k, 240k). Small images are
    # therefore upscaled toward this target; large scans already clear it and
    # are left alone, since upscaling them only costs time.
    ocr_target_longest_px: int = 2200
    ocr_max_upscale: int = 4
    # A page that loses one digit out of forty is fine; a page where a quarter
    # of its numbers were unreadable is not quotable. Proportional, so long
    # scanned documents do not warn on a single bad token.
    ocr_digit_drop_ratio: float = 0.25

    # ---- Local accounts (UI gate) --------------------------------------
    # PBKDF2-HMAC-SHA256 iterations per password check. 200k is roughly
    # 100-200 ms on a laptop CPU: slow enough to make guessing expensive, fast
    # enough that logging in does not feel broken.
    auth_pbkdf2_iterations: int = 200_000
    auth_min_password_length: int = 10
    auth_max_failed_attempts: int = 5
    auth_lockout_minutes: int = 5
    # Bearer tokens for api.py, checked against the same accounts and roles as
    # the UI. Turning this off makes the API an open door on whatever interface
    # it is bound to -- leave it on unless the port is loopback-only and the
    # machine is trusted.
    api_auth_enabled: bool = True

    # ---- Ingestion limits (security) -----------------------------------
    max_file_bytes: int = 200 * 1024 * 1024
    max_query_chars: int = 2000

    # ---- Rate limiting (model-running routes only) --------------------
    # Token bucket on /query, /query_file and /ingest, which are the routes that
    # load and run local models. Read-only endpoints (/health, /sources,
    # /manifest, /history) are not limited. `burst` requests may arrive at once,
    # after which the per-minute rate refills the bucket. An /ingest request is
    # charged one token per file. Counted in process memory, so two uvicorn
    # workers mean twice the effective limit -- see rate_limit.py.
    # The defaults are deliberately generous: a presenter clicking through demo
    # questions, or a folder of slides dropped on the upload box, must not meet a
    # 429. What they stop is a retry loop or a polling client hammering a service
    # where every call costs seconds of GPU time.
    rate_limit_enabled: bool = True
    rate_limit_queries_per_minute: float = 30.0
    rate_limit_queries_burst: int = 6
    rate_limit_ingest_per_minute: float = 60.0
    rate_limit_ingest_burst: int = 20
    # Only turn this on behind a proxy that you control and that sets
    # X-Forwarded-For. Off means the socket peer is the client. On means the
    # rightmost forwarded hop is, which is the only entry in that header a client
    # cannot forge; anything to its left is whatever the client chose to send.
    rate_limit_trust_forwarded_for: bool = False

    # ---- Paths ---------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    index_dir: Path = PROJECT_ROOT / "index"
    evaluation_dir: Path = PROJECT_ROOT / "evaluation"

    # ---- External binaries ---------------------------------------------
    tesseract_cmd: str = Field(default=r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    soffice_cmd: str = Field(default=r"C:\Program Files\LibreOffice\program\soffice.exe")

    # ---- Derived paths --------------------------------------------------
    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def faiss_path(self) -> Path:
        return self.index_dir / "faiss.index"

    @property
    def db_path(self) -> Path:
        return self.index_dir / "store.db"

    @property
    def auth_db(self) -> Path:
        """Accounts live beside chat history and outside the index: rebuilding
        the corpus must not log everybody out or delete the role assignments."""
        return self.data_dir.parent / "auth" / "users.db"

    @property
    def history_db(self) -> Path:
        """Chat sessions deliberately live outside the index: rebuilding or
        deleting the index must not destroy a reviewer's conversation."""
        return self.data_dir.parent / "history" / "chat_history.db"

    def ensure_dirs(self) -> None:
        for path in (
            self.documents_dir, self.images_dir, self.audio_dir,
            self.index_dir, self.evaluation_dir / "reports",
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()

# Point pytesseract at the installed binary if the default path exists.
if Path(settings.tesseract_cmd).exists():
    os.environ.setdefault("TESSERACT_CMD", settings.tesseract_cmd)
