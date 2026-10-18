"""Command line entry point: python -m lsi <command> --help"""

import argparse
import time
from pathlib import Path

import numpy as np

from .model import LSI
from .preprocessing import filter_corpus, load_newsgroup, tokenize
from .tfidf import count_matrix, tfidf_matrix


def cmd_search(args):
    docs = load_newsgroup(args.category, args.docs)
    model = LSI(k=args.k).fit(docs)
    print(f"{len(model.doc_ids)} docs, {len(model.vocab)} terms, k={model.k} concepts\n")

    results = model.search(args.query, top=args.top)
    if not results:
        print("No query word appears in the corpus.")
    for rank, (doc, score) in enumerate(results, 1):
        snippet = " ".join(docs[doc].split())[:100]
        print(f"{rank}. doc {doc:<4} score {score:.3f}  {snippet}...")


def cmd_concepts(args):
    model = LSI(k=args.k).fit(load_newsgroup(args.category, args.docs))
    for i, terms in enumerate(model.concepts(args.terms)[: args.n], 1):
        print(f"concept {i}: {', '.join(terms)}")


def cmd_stats(args):
    _, tokens = filter_corpus(map(tokenize, load_newsgroup(args.category, args.docs)))
    vocab, counts = count_matrix(tokens)
    _, _, X = tfidf_matrix(tokens)

    def top(values):
        return ", ".join(vocab[i] for i in np.argsort(values)[::-1][:10])

    print(f"docs: {len(tokens)}   vocabulary: {len(vocab)} terms")
    print(f"longest doc: {int(counts.sum(axis=0).max())} words")
    print(f"most frequent: {top(counts.sum(axis=1))}")
    print(f"highest tf-idf: {top(X.max(axis=1))}")


def cmd_plot(args):
    import matplotlib.pyplot as plt

    model = LSI(k=max(args.k or 2, 2)).fit(load_newsgroup(args.category, args.docs))
    xy = model.doc_coordinates()[:, :2]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(xy[:, 0], xy[:, 1], s=14)
    for doc, (x, y) in zip(model.doc_ids, xy):
        ax.annotate(str(doc), (x, y), fontsize=7, alpha=0.6)
    ax.set_xlabel("concept 1")
    ax.set_ylabel("concept 2")
    ax.set_title(f"{args.category}: documents in concept space")
    _save(fig, args.out / "concepts_2d.png")


def cmd_benchmark(args):
    import matplotlib.pyplot as plt

    corpus = load_newsgroup(args.category, max(args.sizes))
    sc = None
    if args.spark:
        from .spark_svd import spark_svd, start_spark
        sc = start_spark(args.spark)

    rows = []
    for n in args.sizes:
        _, _, X = tfidf_matrix(filter_corpus(map(tokenize, corpus[:n]))[1])
        t0 = time.perf_counter()
        np.linalg.svd(X, full_matrices=False)
        row = [n, X.shape[0], time.perf_counter() - t0]
        if sc:
            t0 = time.perf_counter()
            spark_svd(sc, X, k=min(args.k, X.shape[1]))
            row.append(time.perf_counter() - t0)
        rows.append(row)
        print("docs {:>4}  terms {:>5}  numpy {:.3f}s".format(*row[:3])
              + (f"  spark {row[3]:.3f}s" if sc else ""))

    rows = np.array(rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rows[:, 0], rows[:, 2], "o-", label="LSI (numpy)")
    if sc:
        ax.plot(rows[:, 0], rows[:, 3], "o-", label=f"MR-LSI (Spark, k={args.k})")
    ax.set_xlabel("number of documents")
    ax.set_ylabel("SVD time (s)")
    ax.set_title("SVD time vs corpus size")
    ax.legend()
    _save(fig, args.out / "benchmark.png")


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser(prog="python -m lsi", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, func, help_text, docs=300):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--category", default="sci.space", help="20 Newsgroups category")
        p.add_argument("--docs", type=int, default=docs, help="number of posts to use")
        p.add_argument("--k", type=int, help="number of concepts (default: automatic)")
        p.add_argument("--out", type=Path, default=Path("results"), help="folder for figures")
        p.set_defaults(func=func)
        return p

    p = add("search", cmd_search, "rank documents for a free-text query")
    p.add_argument("query")
    p.add_argument("--top", type=int, default=5)

    p = add("concepts", cmd_concepts, "print the top terms of each concept")
    p.add_argument("-n", type=int, default=5, help="number of concepts to show")
    p.add_argument("--terms", type=int, default=8, help="terms per concept")

    add("stats", cmd_stats, "corpus statistics")
    add("plot", cmd_plot, "plot documents on the first two concepts", docs=100)

    p = add("benchmark", cmd_benchmark, "time numpy SVD vs Spark SVD")
    p.add_argument("--sizes", type=int, nargs="+", default=list(range(30, 301, 30)))
    p.add_argument("--spark", metavar="MASTER", nargs="?", const="local[*]",
                   help="also run Spark; optional master URL (default local[*])")
    p.set_defaults(k=50)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
