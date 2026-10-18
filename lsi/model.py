"""Latent Semantic Indexing: TF-IDF + truncated SVD, with query search."""

import numpy as np

from .preprocessing import filter_corpus, tokenize
from .tfidf import count_matrix, tfidf_matrix


def choose_rank(s, max_error=0.38):
    """Smallest k whose rank-k approximation has relative Frobenius error < max_error.

    For a rank-k SVD approximation, ||X - X_k||_F / ||X||_F
    = sqrt(sum(s[k:]^2) / sum(s^2)), so no reconstruction is needed.
    """
    energy = s ** 2
    tail = np.sqrt(np.clip(1 - np.cumsum(energy) / energy.sum(), 0, None))  # k = 1, 2, ...
    below = np.flatnonzero(tail < max_error)
    return int(below[0]) + 1 if below.size else len(s)


def _flip_signs(U, Vt):
    """Make each concept's largest term weight positive (SVD signs are arbitrary)."""
    signs = np.sign(U[np.abs(U).argmax(axis=0), range(U.shape[1])])
    signs[signs == 0] = 1
    return U * signs, Vt * signs[:, None]


class LSI:
    """Fit on a list of raw documents, then search them with free-text queries.

    k: number of concepts to keep. If None, it is picked with `choose_rank`.
    min_df, min_words: see `filter_corpus`.
    """

    def __init__(self, k=None, max_error=0.38, min_df=2, min_words=20):
        self.k = k
        self.max_error = max_error
        self.min_df = min_df
        self.min_words = min_words

    def fit(self, docs):
        # Skipped posts are left out of the model; doc_ids maps rows back to `docs`.
        ids, tokens = filter_corpus(map(tokenize, docs), self.min_df, self.min_words)
        self.doc_ids = np.array(ids)
        self.vocab, self.idf, self.X = tfidf_matrix(tokens)

        U, s, Vt = np.linalg.svd(self.X, full_matrices=False)
        k = min(self.k or choose_rank(s, self.max_error), len(s))
        self.U, self.Vt = _flip_signs(U[:, :k], Vt[:k])
        self.s = s[:k]
        self.k = k
        return self

    def search(self, query, top=5):
        """Return [(doc_index, cosine_score), ...] best match first.

        The query is folded into concept space: q_k = q^T U_k S_k^-1,
        then compared with each document's row of V_k.
        """
        _, q = count_matrix([tokenize(query)], vocab=self.vocab)
        q = q[:, 0] * self.idf
        if not q.any():
            return []

        q_k = (q @ self.U) / self.s
        docs_k = self.Vt.T
        scores = docs_k @ q_k / (np.linalg.norm(docs_k, axis=1) * np.linalg.norm(q_k) + 1e-12)
        best = np.argsort(scores)[::-1][:top]
        return [(int(self.doc_ids[i]), float(scores[i])) for i in best]

    def concepts(self, n_terms=10):
        """Top terms of each concept, strongest concept first."""
        return [
            [self.vocab[t] for t in np.argsort(col)[::-1][:n_terms]]
            for col in self.U.T
        ]

    def doc_coordinates(self):
        """Documents in concept space (one row per doc): V_k * S_k."""
        return self.Vt.T * self.s
