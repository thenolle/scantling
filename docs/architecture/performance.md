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
