"""
scripts/verify_env.py -- prove every dependency Evidence AI needs is importable.

Called by setup.ps1 / setup_windows.ps1 right after the pip installs. Exits 0
only if every import succeeded, so a broken or partially-installed environment
fails the setup instead of surfacing later as a confusing runtime error.

Run manually any time:
    .venv\\Scripts\\python scripts\\verify_env.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Allow running as `python scripts/verify_env.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# (import name, pip distribution name, hard requirement?)
DEPENDENCIES = [
    ("pydantic", "pydantic", True),
    ("pydantic_settings", "pydantic-settings", True),
    ("dotenv", "python-dotenv", True),
    ("torch", "torch", True),
    ("sentence_transformers", "sentence-transformers", True),
    ("numpy", "numpy", True),
    ("faiss", "faiss-cpu", True),
    ("fitz", "pymupdf", True),
    ("docx", "python-docx", True),
    ("pandas", "pandas", True),
    ("PIL", "pillow", True),
    ("cv2", "opencv-python-headless", True),
    ("pytesseract", "pytesseract", True),
    ("faster_whisper", "faster-whisper", True),
    ("av", "av", True),
    ("ctranslate2", "ctranslate2", True),
    ("fastapi", "fastapi", True),
    ("uvicorn", "uvicorn[standard]", True),
    ("multipart", "python-multipart", True),
    ("streamlit", "streamlit", True),
    ("requests", "requests", True),
    ("pyttsx3", "pyttsx3", False),   # demo speech only; Windows SAPI
]

# Project modules -- these catch config/schema breakage, not just pip problems.
PROJECT_MODULES = [
    "config",
    "schemas",
    "embeddings",
    "vector_store",
    "retriever",
    "reranker",
    "documents",
    "images",
    "audio",
    "citations",
    "relationships",
    "llm",
    "providers",
    "rag_pipeline",
]


def _version(module) -> str:
    for attr in ("__version__", "VERSION"):
        value = getattr(module, attr, None)
        if isinstance(value, str):
            return value
    try:
        from importlib.metadata import version

        return version(module.__name__)
    except Exception:
        return "unknown"


AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".opus"}


def check_audio_path() -> tuple[bool, str]:
    """Decode a real audio file through faster-whisper's own decoder.

    This exercises the exact call that breaks when PyAV is too new, instead of
    inspecting a signature (av.open is a C builtin with no introspectable one).
    Falls back to a version check when no audio file exists yet.
    """
    try:
        import av
    except Exception as exc:
        return False, f"PyAV is not importable ({exc})"

    version = str(getattr(av, "__version__", "unknown"))

    sample: Path | None = None
    try:
        from config import settings

        audio_dir = settings.audio_dir
        if audio_dir.exists():
            for path in sorted(audio_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
                    sample = path
                    break
    except Exception:
        sample = None

    if sample is not None:
        try:
            from faster_whisper.audio import decode_audio

            frames = decode_audio(str(sample), sampling_rate=16000)
            seconds = len(frames) / 16000.0
            return True, (f"decoded {sample.name} -> {seconds:.1f}s of audio "
                          f"(av {version} + faster-whisper)")
        except TypeError as exc:
            return False, (f"av {version} rejected faster-whisper's call: {exc}. "
                           "Install the pinned builds:  pip install av==18.1.0 "
                           "faster-whisper==1.2.1 ctranslate2==4.8.1")
        except Exception as exc:
            return False, f"{exc.__class__.__name__}: {exc}"

    try:
        major = int(version.split(".")[0])
    except ValueError:
        major = None
    if major is not None and major >= 19:
        return False, (f"av {version} is on the release line that dropped the "
                       "'metadata_errors' argument faster-whisper passes, and no "
                       "audio file was found to test against. Fix with:  "
                       "pip install av==18.1.0")
    return True, (f"av {version} predates the breaking 19.x line "
                  "(no audio file present to decode yet)")


def main() -> int:
    failures: list[str] = []
    optional_missing: list[str] = []

    print(f"Python {sys.version.split()[0]}  ({sys.executable})")
    print("-" * 66)
    print("Third-party packages")

    for import_name, dist_name, required in DEPENDENCIES:
        try:
            module = importlib.import_module(import_name)
            print(f"  [OK]   {dist_name:<26} {_version(module)}")
        except Exception as exc:
            if required:
                failures.append(f"{dist_name}: {exc}")
                print(f"  [FAIL] {dist_name:<26} {exc}")
            else:
                optional_missing.append(dist_name)
                print(f"  [SKIP] {dist_name:<26} optional - {exc.__class__.__name__}")

    print("-" * 66)
    print("Project modules")
    for name in PROJECT_MODULES:
        try:
            importlib.import_module(name)
            print(f"  [OK]   {name}")
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            print(f"  [FAIL] {name}: {exc}")

    print("-" * 66)
    # The single most common silent breakage: PyAV 19 removed the
    # `metadata_errors=` keyword that faster-whisper 1.2.1 passes at
    # faster_whisper/audio.py:46, so audio ingestion dies only when an audio
    # file is first ingested. Test the real code path instead of guessing.
    audio_ok, audio_detail = check_audio_path()
    if audio_ok:
        print(f"  [OK]   audio decode path: {audio_detail}")
    else:
        failures.append(f"audio decode path: {audio_detail}")
        print(f"  [FAIL] audio decode path: {audio_detail}")

    if optional_missing:
        print(f"\nOptional and missing (demo audio generation only): "
              f"{', '.join(optional_missing)}")

    if failures:
        print(f"\n{len(failures)} problem(s) found:")
        for item in failures:
            print(f"  - {item}")
        print("\nFix with:  .venv\\Scripts\\python -m pip install -r requirements-lock.txt")
        return 1

    print("\nEnvironment verified - all required imports succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
