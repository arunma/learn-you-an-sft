"""Text normalisation: fix Unicode, scrub URLs/emails, normalise whitespace.

Two principles:

1. Be aggressive about WEIRD characters (mojibake from old IRC clients,
   smart quotes from Reddit's WYSIWYG editor, zero-width joiners, BOMs).
   They don't help the model learn humor and they bloat tokenizer
   vocabulary use.

2. Be CONSERVATIVE about meaning-preserving content. Don't lowercase,
   don't strip punctuation, don't unify spelling. Sarcasm leans
   heavily on capitalisation ("OH GREAT") and punctuation
   ("...sure.") for tone — destroying those would erase the very
   thing we're trying to teach.

We do replace URLs and emails with `<URL>` / `<EMAIL>` placeholders.
Real URLs in jokes are usually irrelevant noise (the joke would land
the same with any URL), and they're a common mojibake/PII vector.
"""
from __future__ import annotations

import re
from typing import Iterable

import ftfy

from ..ingest.schema import Pair

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
MULTI_SPACE = re.compile(r" {2,}")
MULTI_NEWLINE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """The standard pipeline: ftfy → URL/email scrub → whitespace normalise."""
    text = ftfy.fix_text(text)
    text = URL_RE.sub("<URL>", text)
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = MULTI_SPACE.sub(" ", text)
    text = MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def normalize(pairs: Iterable[Pair]) -> Iterable[Pair]:
    """Stream-clean a sequence of Pairs. Drops pairs that go empty after cleanup."""
    for pair in pairs:
        cleaned_prompt = clean_text(pair.prompt)
        cleaned_response = clean_text(pair.response)
        if not cleaned_prompt or not cleaned_response:
            continue
        # New Pair, not in-place mutation — keeps the upstream object intact
        # in case anything else holds a reference.
        yield Pair(
            prompt=cleaned_prompt,
            response=cleaned_response,
            source=pair.source,
            score=pair.score,
            meta=pair.meta,
        )
