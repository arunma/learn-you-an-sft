"""English-only filter using fasttext's lid.176 language identifier.

fasttext lid.176 is the de-facto language ID for short text. Three
properties make it the right choice here:

  - Fast: ~1ms per snippet on CPU, no GPU needed.
  - Small: ~125 MB binary, loads once.
  - Accurate enough for "is this English?" — the exact false-positive
    rate doesn't matter much because we drop on either side, and
    English vs near-English mistakes don't hurt training.

We check both prompt and response. A pair where either side is in
another language is dropped — codeswitch examples teach the model
weird habits.

The model isn't bundled (it's 125 MB) — the user downloads it once.
The error message in get_model() shows the exact curl command.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import fasttext

from ..ingest.schema import Pair

MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
DEFAULT_MODEL_PATH = Path.home() / ".cache" / "fasttext" / "lid.176.bin"

# Module-level cache: load the 125 MB model once per process, not per pair.
_model = None


def get_model(path: Path = DEFAULT_MODEL_PATH):
    global _model
    if _model is None:
        if not path.exists():
            raise SystemExit(
                f"fasttext lid.176 not found at {path}.\n\n"
                f"Download once with:\n"
                f"  mkdir -p {path.parent}\n"
                f"  curl -L {MODEL_URL} -o {path}\n"
            )
        _model = fasttext.load_model(str(path))
    return _model


def is_english(text: str, threshold: float = 0.6) -> bool:
    """Return True iff fasttext's top-1 prediction is English with prob >= threshold."""
    model = get_model()
    # fasttext requires single-line input; truncate to bound predict cost.
    sanitised = text.replace("\n", " ")[:1000]
    if not sanitised.strip():
        return False
    labels, probs = model.predict(sanitised, k=1)
    label = labels[0].replace("__label__", "")
    return label == "en" and probs[0] >= threshold


def filter_english(pairs: Iterable[Pair]) -> Iterable[Pair]:
    """Yield only pairs where both prompt and response are confidently English."""
    for pair in pairs:
        if is_english(pair.prompt) and is_english(pair.response):
            yield pair
