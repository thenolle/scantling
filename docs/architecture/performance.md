---
title: Performance work
section: Architecture
order: 15
description: What limits speed, which optimisations were made and how each is measured.
---

# Performance work

Every change here is judged by a benchmark run before and after. Speed claims are recorded in [Benchmarks](../project/benchmarks.md) only after they have been measured.

## What limits each workload

| Workload                  | Limit                                       | Reasoning                                                                                                                               |
| ------------------------- | ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Matrix products           | Instruction throughput in scalar JavaScript | About 2.4 GFLOP/s is far below what L2 bandwidth supports, so cache tiling would gain nothing yet                                       |
| Dot-product (`nt`) kernel | Floating-point add latency                  | One accumulator serialises every add, and the kernel was the slowest at about 2.0 GFLOP/s                                               |
| Decode of Tiny            | Instruction throughput                      | The 5.3 MB of weights fit in cache, so memory bandwidth is not the limit                                                                |
| Decode of Tiny and Small  | Instruction throughput                      | Int8 weights, which cut memory traffic by 4x, were slower on Small, so memory bandwidth is not the limit on this machine                |
| Parallel training         | Per-core slowdown when all cores are busy   | Synchronisation is 1.5 to 3% of a step; one worker needs 581 ms for a micro-batch and eight concurrent workers need about 1,470 ms each |

## What changed

- **Matrix kernels.** `nn` and `tn` process two output rows and unroll the shared dimension by four, which cuts accumulator loads and stores by a factor of four. `nt` computes four outputs together with four independent accumulators. Each row's arithmetic order does not depend on how rows are grouped, so splitting work across workers is still bit-exact; a test checks every split point.
- **Scratch reuse.** The kernels' accumulator is reused across calls instead of allocated per product.
- **Fast decoder.** A dedicated single-token path with preallocated buffers, a preallocated KV cache and transposed weights, so one decode step allocates nothing.
- **Int8 weights.** Optional per-row symmetric quantisation of every projection and the output head. The scale is applied once per output, not per weight.
- **Training synchronisation.** All micro-batches of a step are dispatched at once with one gradient slot each, so there is no barrier between rounds. Results are bit-identical to the previous loop. Each phase of a step is timed.

## Measuring

```bash
bun run bench -- --quick
bun run bench:inference
bun run train:scaling
```

## Not done, and why

Cache tiling and activation buffer pooling are deferred until a profile shows they matter. Weight quantisation is not applied to training.

## Profile of a training step

Profiling a forward and backward pass of the Tiny model on 2 sequences of 128 tokens showed that matrix products take 91% of the 808 ms step (229 ms forward, 510 ms backward). Cross-entropy, softmax, SiLU and everything else together are about 9%. Optimising elementwise code can therefore gain a few percent at most, so the work went into the matrix kernels.

The tied output head (a 4096 by 128 table used as a matrix on every token) accounts for about 40% of the Tiny model's compute.

## SIMD matrix kernel

A WebAssembly module, assembled by a small encoder in TypeScript, computes `C = A * B` in 4 by 8 blocks with 128-bit vectors. Operands are copied into WebAssembly memory, padded to whole blocks and transposed during the copy when needed, so one kernel serves all transpose modes. Accumulation is in float32 with a fixed order, so results are deterministic and independent of how rows are grouped, but they differ from the scalar kernels in the last digits.

The path is used for products with at least 4 rows and 16,384 multiply-adds when the runtime supports SIMD. Set `SCANTLING_SIMD=0` to disable it.
