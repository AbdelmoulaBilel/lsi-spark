"""Term-document TF-IDF matrix, written by hand (no sklearn vectorizer)."""

import numpy as np


def count_matrix(token_lists, vocab=None):
    """Return (vocab, counts) where counts[t, d] = occurrences of term t in doc d."""
    if vocab is None:
        vocab = sorted({w for tokens in token_lists for w in tokens})
    index = {w: i for i, w in enumerate(vocab)}

    counts = np.zeros((len(vocab), len(token_lists)))
    for d, tokens in enumerate(token_lists):
        for w in tokens:
            if w in index:
                counts[index[w], d] += 1
    return vocab, counts


def idf(counts):
    """idf(t) = log(N / df(t)), where df(t) = number of docs containing t."""
    n_docs = counts.shape[1]
    df = np.count_nonzero(counts, axis=1)
    return np.log(n_docs / df)


def tfidf_matrix(token_lists):
    """Return (vocab, idf, X) where X is the terms x docs TF-IDF matrix.

    tf(t, d) = count(t, d) / len(d)
    """
    vocab, counts = count_matrix(token_lists)
    doc_lengths = counts.sum(axis=0)
    weights = idf(counts)
    X = (counts / doc_lengths) * weights[:, None]
    return vocab, weights, X
