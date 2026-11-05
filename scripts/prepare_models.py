"""
scripts/prepare_models.py -- download every local model BEFORE going offline.

Evidence AI needs four models. Three come from HuggingFace into the local hub
cache and one is an Ollama model. This script downloads all of them and then
runs a tiny real inference with each, so a truncated or corrupt download fails
here (while you still have internet) rather than mid-demo.

Run from the project root:
    .venv\\Scripts\\python scripts\\prepare_models.py
    .venv\\Scripts\\python scripts\\prepare_models.py --ollama-model qwen2.5:1.5b
    .venv\\Scripts\\python scripts\\prepare_models.py --skip-ollama

Approximate download sizes:
    sentence-transformers/all-MiniLM-L6-v2        ~90 MB   (embeddings)
    cross-encoder/ms-marco-MiniLM-L-6-v2          ~90 MB   (reranking)
    faster-whisper base (int8, CTranslate2)      ~145 MB   (audio -> text)
    Ollama qwen2.5vl:3b                         ~3.2 GB   (answer generation)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402


def _say(text: str) -> None:
    print(f"\n=== {text} ===", flush=True)


def _ok(text: str) -> None:
    print(f"  [OK]   {text}", flush=True)


def _bad(text: str) -> None:
    print(f"  [FAIL] {text}", flush=True)


def _warn(text: str) -> None:
    print(f"  [WARN] {text}", flush=True)


def cache_location() -> Path:
    import os

    for name in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(name)
        if value:
            return Path(value)
    return Path.home() / ".cache" / "huggingface"


def prepare_embedding() -> bool:
    _say(f"Embedding model  {settings.embedding_model}")
    start = time.time()
    try:
        from embeddings import embed_texts, embedding_dimension

        vectors = embed_texts(["offline multimodal retrieval test"])
        dim = embedding_dimension()
        shape = tuple(vectors.shape)
        _ok(f"loaded and encoded in {time.time() - start:.1f}s")
        _ok(f"dimension={dim} (config expects {settings.embedding_dim}) shape={shape}")
        if dim != settings.embedding_dim:
            _warn("Reported dimension differs from EMBEDDING_DIM. If the index "
                  "was built with a different model, delete the index folder and "
                  "re-ingest.")
        return True
    except Exception as exc:
        _bad(f"{exc.__class__.__name__}: {exc}")
        return False


def prepare_reranker() -> bool:
    _say(f"Reranker  {settings.reranker_model}")
    start = time.time()
    try:
        from reranker import get_model

        model = get_model()
        score = float(model.predict([("what is the budget?", "the budget is 87% used")])[0])
        _ok(f"loaded and scored in {time.time() - start:.1f}s (sample logit {score:.3f})")
        return True
    except Exception as exc:
        _bad(f"{exc.__class__.__name__}: {exc}")
        return False


def prepare_whisper() -> bool:
    _say(f"Speech-to-text  faster-whisper '{settings.whisper_model_size}' "
         f"({settings.whisper_device}/{settings.whisper_compute_type})")
    start = time.time()
    try:
        # Loading the model downloads it into the HF cache on first use.
        import audio

        audio._get_model()  # module-private loader; loads the CTranslate2 model
        _ok(f"model downloaded and loaded in {time.time() - start:.1f}s")
        _warn("A real transcription test needs an audio file. Ingest the demo "
              "WAV (scripts\\ingest_demo.py) to exercise it end to end.")
        return True
    except Exception as exc:
        _bad(f"{exc.__class__.__name__}: {exc}")
        _bad("If this is a PyAV/av error, install the pinned build: "
             "pip install av==18.1.0 faster-whisper==1.2.1 ctranslate2==4.8.1")
        return False


def ollama_reachable() -> bool:
    try:
        with urllib.request.urlopen(
            f"{settings.ollama_host}/api/version", timeout=5
        ) as response:
            payload = json.loads(response.read().decode())
        _ok(f"Ollama server responding, version {payload.get('version')}")
        return True
    except Exception as exc:
        _warn(f"Ollama server not responding at {settings.ollama_host} ({exc})")
        _warn("Start it with:  ollama serve   (or open the Ollama app)")
        return False


def installed_ollama_models() -> list[str]:
    try:
        with urllib.request.urlopen(
            f"{settings.ollama_host}/api/tags", timeout=5
        ) as response:
            payload = json.loads(response.read().decode())
        return [m.get("name", "") for m in payload.get("models", [])]
    except Exception:
        return []


def _tag_matches(installed: list[str], model: str) -> bool:
    """True if `model` is already present, tolerating a missing tag.

    Ollama treats a bare name as ':latest', and /api/tags returns fully
    qualified tags, so 'qwen2.5vl' must match 'qwen2.5vl:latest'.
    """
    wanted = model if ":" in model else f"{model}:latest"
    wanted_name = wanted.split(":")[0]
    for name in installed:
        if name == wanted or name == model:
            return True
        if name.split(":")[0] == wanted_name and ":" not in model:
            return True
    return False


def prepare_ollama(model: str) -> bool:
    _say(f"Answer model (Ollama)  {model}")

    # The CLI is only needed to pull or to start a local server. If the server
    # already answers and already has the model, no CLI is required at all -
    # which matters when Ollama runs as a Windows service, in a container, or
    # on another machine pointed at by OLLAMA_HOST.
    have_cli = shutil.which("ollama") is not None

    if not ollama_reachable():
        if have_cli:
            _warn("Trying to start 'ollama serve' in the background...")
            try:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                time.sleep(6)
            except Exception as exc:
                _warn(f"Could not start it automatically: {exc}")
            ollama_reachable()
        else:
            _bad(f"No Ollama server at {settings.ollama_host} and the 'ollama' "
                 "command is not on PATH.")
            _bad("Start Ollama (open the Ollama app, or run: ollama serve), or "
                 "install it from https://ollama.com/download")
            return False

    installed = installed_ollama_models()
    if _tag_matches(installed, model):
        _ok(f"already present locally: {model}")
    else:
        if installed:
            print(f"  Installed models: {', '.join(installed)}")
        if not have_cli:
            _bad(f"{model} is not installed and the 'ollama' command is not on "
                 "PATH, so it cannot be pulled automatically.")
            _bad(f"On the machine running Ollama, run:  ollama pull {model}")
            return False
        print(f"  Pulling {model} (one time, may take several minutes)...", flush=True)
        start = time.time()
        result = subprocess.run(["ollama", "pull", model])
        if result.returncode != 0:
            _bad(f"ollama pull {model} failed")
            return False
        _ok(f"pulled in {time.time() - start:.0f}s")

    # Real generation, so a corrupt pull is caught now.
    try:
        body = json.dumps(
            {"model": model, "prompt": "Reply with the single word: READY",
             "stream": False, "options": {"num_ctx": 2048}}
        ).encode()
        request = urllib.request.Request(
            f"{settings.ollama_host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        start = time.time()
        with urllib.request.urlopen(request, timeout=settings.ollama_timeout) as response:
            payload = json.loads(response.read().decode())
        reply = (payload.get("response") or "").strip()
        _ok(f"generated a reply in {time.time() - start:.1f}s: {reply[:60]!r}")
        return True
    except Exception as exc:
        _bad(f"generation test failed: {exc.__class__.__name__}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama-model", default=settings.ollama_model,
                        help="Ollama model tag to pull and test "
                             "(default: %(default)s)")
    parser.add_argument("--skip-ollama", action="store_true",
                        help="only prepare the HuggingFace models")
    parser.add_argument("--skip-whisper", action="store_true",
                        help="skip the speech-to-text model")
    args = parser.parse_args()

    print(f"HuggingFace cache: {cache_location()}")
    print(f"Torch device:      {settings.torch_device}")
    print("Downloading models - keep the internet connected until this finishes.")

    results = {
        "embedding": prepare_embedding(),
        "reranker": prepare_reranker(),
    }
    results["whisper"] = True if args.skip_whisper else prepare_whisper()
    results["ollama"] = True if args.skip_ollama else prepare_ollama(args.ollama_model)

    _say("Summary")
    for name, passed in results.items():
        print(f"  {'[OK]  ' if passed else '[FAIL]'} {name}")

    if all(results.values()):
        print("\nAll models are cached locally. You can now disconnect from the")
        print("internet and prove it with:  .venv\\Scripts\\python scripts\\verify_offline.py")
        return 0

    print("\nOne or more models failed to prepare. Re-run after fixing the "
          "errors above; downloads resume rather than restarting.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
