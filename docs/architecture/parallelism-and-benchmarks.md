---
title: Parallelism and benchmarks
section: Architecture
order: 7
description: How matrix products are split across worker threads and how performance is measured.
---

# Parallelism and benchmarks

## Parallel matrix products

`ParallelMatmul` splits the rows of the output across a pool of worker threads. The operands are copied into `SharedArrayBuffer`s, each worker computes its row ranges with the same kernels as the single-threaded `matmul`, and the result is copied back. The arithmetic order within a row is unchanged, so the results are bit-identical to the single-threaded path.

Copying costs roughly `m*k + k*n` values, while the product costs `m*k*n` multiply-adds, so parallelism only pays off above some size. Products below a FLOP threshold stay on the calling thread. The threshold is set from measurements, not guessed.

Calls are serialised because they share the staging buffers. A worker that crashes is replaced and its job is rejected.

## Measuring

```bash
bun run bench              # default sizes up to 512
bun run bench -- --quick   # small sizes
bun run bench -- --full    # adds 1024
bun run bench -- --markdown
```

The report starts with the processor, memory, Bun version and operating system, so every number is attributable to a machine. A JSON copy is saved under `data/benchmarks/`.

| Section             | What it measures                                              |
| ------------------- | ------------------------------------------------------------- |
| Matmul, one thread  | GFLOP/s of the three transpose kernels at several sizes       |
| Matmul, worker pool | GFLOP/s and speedup against one thread for several pool sizes |
| Elementwise         | Addition and softmax throughput                               |
| Linear layer        | Forward and forward plus backward through the autodiff engine |
| Ceiling tokens/s    | An upper bound for training speed                             |

> [!NOTE]
> The tokens-per-second ceiling divides the best measured matrix throughput by `6 * parameters`. Real training also runs activations, normalisation and memory-bound operations, so it will be slower.
