import numpy as np
import pytest

from lsi import LSI, choose_rank, tokenize, tfidf_matrix
from lsi.preprocessing import filter_corpus


def fit(k):
    # The toy docs are tiny, so turn the corpus filters off.
    return LSI(k=k, min_df=1, min_words=1).fit(DOCS)


DOCS = [
    "The rocket launch was delayed by bad weather at the launch pad.",
    "NASA plans a new rocket launch to the moon next year.",
    "The moon orbit and lunar landing were a big success for NASA.",
    "My car engine needs new oil and the brakes are worn.",
    "Engine oil and brake pads: car maintenance tips.",
    "",  # empty after cleaning, must be skipped
]


def test_tokenize_drops_stopwords_digits_and_punctuation():
    assert tokenize("The 3 rockets, launched!") == ["rockets", "launched"]


def test_tfidf_matches_formula():
    vocab, idf, X = tfidf_matrix([["a", "a", "b"], ["b", "c"]])
    assert vocab == ["a", "b", "c"]
    np.testing.assert_allclose(idf, [np.log(2), 0, np.log(2)])
    np.testing.assert_allclose(X[:, 0], [2 / 3 * np.log(2), 0, 0])


def test_choose_rank():
    assert choose_rank(np.array([10.0, 1.0, 0.1]), max_error=0.38) == 1
    assert choose_rank(np.array([1.0, 1.0, 1.0]), max_error=0.38) == 3


def test_svd_reconstruction_with_full_rank():
    model = fit(5)
    np.testing.assert_allclose((model.U * model.s) @ model.Vt, model.X, atol=1e-10)


def test_search_finds_topic():
    model = fit(2)
    assert {doc for doc, _ in model.search("rocket moon", top=3)} == {0, 1, 2}
    assert {doc for doc, _ in model.search("car oil", top=2)} == {3, 4}


def test_search_unknown_words():
    assert fit(2).search("xyzzy") == []


def test_empty_docs_skipped():
    model = fit(2)
    assert list(model.doc_ids) == [0, 1, 2, 3, 4]
    assert model.doc_coordinates().shape == (5, 2)


def test_spark_matches_numpy():
    pytest.importorskip("pyspark")
    from lsi.spark_svd import spark_svd, start_spark

    _, _, X = tfidf_matrix([tokenize(d) for d in DOCS[:5]])
    _, s, _ = spark_svd(start_spark("local[1]"), X, k=3)
    np.testing.assert_allclose(s, np.linalg.svd(X, compute_uv=False)[:3], rtol=1e-6)


def test_filter_corpus():
    ids, tokens = filter_corpus([["a", "b"], ["a", "c"], ["c"], []], min_df=2, min_words=1)
    assert ids == [0, 1, 2]
    assert tokens == [["a"], ["a", "c"], ["c"]]
