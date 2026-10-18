"""Distributed SVD with Spark MLlib (the "MR-LSI" side of the benchmark)."""

import numpy as np


def start_spark(master="local[*]"):
    """Start (or reuse) a SparkContext.

    master: "local[*]" to use every local core, or a standalone cluster URL
    such as "spark://127.0.0.1:7077" (see docs/spark-cluster.md).
    """
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master(master).appName("lsi-svd").getOrCreate()
    return spark.sparkContext


def spark_svd(sc, X, k, partitions=None):
    """SVD of X (terms x docs) with RowMatrix.computeSVD.

    Each term row becomes one record of a distributed RDD. Returns numpy
    arrays (U, s, Vt) shaped like np.linalg.svd(X, full_matrices=False)
    truncated to k.
    """
    from pyspark.mllib.linalg.distributed import RowMatrix

    rows = sc.parallelize(X.tolist(), numSlices=partitions)
    svd = RowMatrix(rows).computeSVD(k, computeU=True)
    U = np.array(svd.U.rows.map(lambda r: r.toArray()).collect())
    return U, svd.s.toArray(), svd.V.toArray().T
