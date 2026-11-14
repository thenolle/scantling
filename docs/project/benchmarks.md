---
title: Benchmarks
section: Project
order: 3
description: Measured matrix product, elementwise and linear layer throughput on the development machine.
---

# Benchmarks

Measured with `bun run bench` on an Intel Core i5-12450H (12 logical cores), 15.7 GiB RAM, Bun 1.4.4, Windows 11. Figures vary by roughly 5 to 10% between runs.

## Matrix products, one thread

| Size      | `nn` GFLOP/s | `nt` GFLOP/s | `tn` GFLOP/s |
| --------- | ------------ | ------------ | ------------ |
| 64 x 64   | 2.20         | 1.79         | 2.17         |
| 128 x 128 | 2.28         | 1.95         | 2.38         |
| 256 x 256 | 2.29         | 2.01         | 2.53         |
| 512 x 512 | 2.45         | 2.13         | 2.58         |

## Matrix products, worker pool (`nn`, GFLOP/s)

| Workers | 128 x 128 | 256 x 256 | 512 x 512 |
| ------- | --------- | --------- | --------- |
| 1       | 2.30      | 2.33      | 2.51      |
| 2       | 3.99      | 3.99      | 4.44      |
| 4       | 6.71      | 6.49      | 6.08      |
| 8       | 5.60      | 7.93      | 8.87      |
| 12      | 5.71      | 7.90      | 9.78      |

## Small products

A second run measured smaller sizes. Results vary by 20% or more between runs at these sizes.

| Workers | 64 x 64 | 128 x 128 | 256 x 256 |
| --- | --- | --- | --- |
| 1 | 1.41 | 2.18 | 2.49 |
| 2 | 2.98 | 3.95 | 4.05 |
| 4 | 2.40 | 5.14 | 6.38 |
| 8 | 2.59 | 5.71 | 7.81 |
| 12 | 1.87 | 4.18 | 7.77 |

Products below 2 MFLOP stay on the calling thread, and one extra worker is used per MFLOP up to the pool size.

At 512 x 512 the best result is 4.0 times a single thread. A pool of one worker matches the calling thread, so the copy into shared memory is not a measurable cost at these sizes.

## Other operations

| Operation                              | Result                 |
| -------------------------------------- | ---------------------- |
| Elementwise add, 1M elements           | 795 Melem/s            |
| Softmax, 512 x 1024                    | 125,938 rows/s         |
| Linear layer 512 x 128 to 512, forward | 2.27 GFLOP/s (29.6 ms) |
| Linear layer, forward and backward     | 3.69 GFLOP/s (54.5 ms) |

## Training speed ceiling

The ceiling divides the best measured throughput (9.78 GFLOP/s) by `6 * parameters`. Parameter counts are estimates until the model exists, and real training will be slower because normalisation, activations and memory-bound work do not run at matrix speed.

| Configuration | Parameters (estimate) | Ceiling         |
| ------------- | --------------------- | --------------- |
| Micro         | 0.15 M                | 10,866 tokens/s |
| Tiny          | 1.3 M                 | 1,254 tokens/s  |
| Small         | 14 M                  | 116 tokens/s    |

At the ceiling, 10 million tokens take about 2.2 hours for Tiny and about 24 hours for Small.
