"""Load 20 Newsgroups posts and turn them into token lists."""

import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_WORD = re.compile(r"[a-z]+")


def load_newsgroup(category="sci.space", n_docs=None, subset="train", seed=42):
    """Return the raw text of up to `n_docs` posts from one newsgroup."""
    from sklearn.datasets import fetch_20newsgroups

    data = fetch_20newsgroups(
        subset=subset,
        categories=[category],
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=seed,
    )
    return data.data[:n_docs] if n_docs else data.data


def tokenize(text):
    """Lowercase, keep letters only, drop stop words and single letters."""
    return [
        w for w in _WORD.findall(text.lower())
        if len(w) > 1 and w not in ENGLISH_STOP_WORDS
    ]


def filter_corpus(token_lists, min_df=2, min_words=20):
    """Drop rare words, then drop posts that are too short to say anything.

    min_df: keep words that appear in at least this many posts.
    min_words: keep posts with at least this many words left.
    Returns (kept_indices, kept_token_lists).
    """
    token_lists = list(token_lists)
    df = Counter(w for tokens in token_lists for w in set(tokens))
    kept = []
    for i, tokens in enumerate(token_lists):
        tokens = [w for w in tokens if df[w] >= min_df]
        if len(tokens) >= max(min_words, 1):
            kept.append((i, tokens))
    return [i for i, _ in kept], [t for _, t in kept]
