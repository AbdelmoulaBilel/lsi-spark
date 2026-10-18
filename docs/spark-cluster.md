# Running the benchmark on a Spark standalone cluster (Windows)

These are the steps used for the original experiments.

## Machine used

- CPU: Intel Core i5-8300H, 4 cores / 8 threads, 2.3 GHz (4 GHz turbo)
- RAM: 8 GB
- OS: Windows 10, 64-bit
- Spark 3.0.3 (pre-built for Hadoop 2.7), installed in `C:\spark\spark-3.0.3-bin-hadoop2.7`

## 1. Start the master

From the Spark folder:

```bat
cd C:\spark\spark-3.0.3-bin-hadoop2.7
bin\spark-class2.cmd org.apache.spark.deploy.master.Master
```

Open http://localhost:8080 and copy the master URL, e.g. `spark://192.168.43.78:7077`.

## 2. Start one or more workers

Run this once per worker, each in its own terminal:

```bat
bin\spark-class2.cmd org.apache.spark.deploy.worker.Worker -c 1 -m 1G spark://192.168.43.78:7077
```

- `-c 1`: CPU cores given to the worker
- `-m 1G`: RAM given to the worker

## 3. Run the benchmark against the cluster

```bash
python -m lsi benchmark --spark spark://192.168.43.78:7077
```

Use `--spark` with no URL to run Spark in local mode (`local[*]`) instead.

## 4. Watch it run

- http://localhost:8080: the master. Applications, cores, memory per executor, duration.
- http://localhost:8081, 8082, ...: one page per worker.
- http://localhost:4040: the running application. Jobs, stages, executors.

## How the distributed SVD works

- The TF-IDF matrix (terms x documents) is split by rows into an RDD: `sc.parallelize(rows)`.
- `RowMatrix(rows).computeSVD(k, computeU=True)` computes the top `k` singular values.
  Spark builds the Gram matrix `XᵀX` with a map/reduce over the rows, then takes its eigen-decomposition.
