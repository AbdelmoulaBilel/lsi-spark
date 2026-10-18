"""Latent Semantic Indexing on 20 Newsgroups, with a Spark SVD benchmark."""

from .model import LSI, choose_rank
from .preprocessing import load_newsgroup, tokenize
from .tfidf import tfidf_matrix

__all__ = ["LSI", "choose_rank", "load_newsgroup", "tokenize", "tfidf_matrix"]
