"""Two-stage deduplication: exact hash + MinHash near-duplicate.

Stage 1 — exact dedup via SHA1 of the response text.
  Catches verbatim duplicates (the same response generated multiple
  times). O(N) memory, O(N) time. The bulk of duplicates fall here.

Stage 2 — near-duplicate via MinHash LSH (Locality Sensitive Hashing).
  Catches paraphrases — small word substitutions, format variations,
  near-identical retellings.

  MinHash works by hashing each document into a signature of K hash
  values such that the Jaccard similarity of two signatures
  approximates the Jaccard similarity of the underlying token sets.
  LSH bands those signatures so candidate-pair lookup is sub-linear
  instead of O(N^2).

  Library: datasketch — battle-tested, simple API. Resist the urge
  to roll your own; getting LSH banding right is fiddly.

Threshold: 0.7 Jaccard similarity on 5-grams. Catches near-duplicates
and trivial edits but lets through *structurally* similar pairs (same
setup, different completion) — variants ARE the training signal.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

from datasketch import MinHash, MinHashLSH

from ..ingest.schema import Pair

NUM_PERM = 128            # MinHash signature length; 128 = standard quality/speed trade
JACCARD_THRESHOLD = 0.7
NGRAM_SIZE = 5


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _ngrams(text: str, n: int = NGRAM_SIZE) -> set[str]:
    """Return the set of word-level n-grams.

    n=5 is a sweet spot: short enough that paraphrases share many
    n-grams, long enough that random short matches don't trigger a
    false positive.
    """
    tokens = text.lower().split()
    if len(tokens) < n:
        # Document shorter than n words — fall back to one bag-of-words shingle.
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _build_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for ng in _ngrams(text):
        m.update(ng.encode("utf-8"))
    return m


def dedupe(pairs: Iterable[Pair]) -> Iterable[Pair]:
    """Yield pairs with both exact and near-duplicates removed.

    Memory: O(N) for the hash set + O(N * NUM_PERM * 8 bytes) for the
    LSH index. At 1M pairs and NUM_PERM=128 → ~1 GB; manageable.
    """
    seen_hashes: set[str] = set()
    lsh = MinHashLSH(threshold=JACCARD_THRESHOLD, num_perm=NUM_PERM)

    for i, pair in enumerate(pairs):
        # Exact dedup on the response text — the prompt may repeat
        # across pairs (templated questions), so duplicate prompts
        # don't necessarily indicate duplicate content.
        h = _sha1(pair.response)
        if h in seen_hashes:
            continue

        m = _build_minhash(pair.response)
        if list(lsh.query(m)):
            # Some near-duplicate is already in the index — drop this one.
            continue

        seen_hashes.add(h)
        lsh.insert(str(i), m)
        yield pair
