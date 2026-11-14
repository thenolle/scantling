---
title: Parallel training
section: Architecture
order: 13
description: How training is split across worker threads and what that guarantees.
---

# Parallel training

Training is data-parallel. Splitting the batch is a much coarser grain than splitting each matrix product, so thread overhead stays small relative to the work.

## One step

1. The main thread copies the master weights into shared memory.
2. Each worker, holding its own copy of the model, takes one micro-batch, runs forward and backward, and writes its gradient into its own slot of a shared buffer.
3. The main thread sums the slots in slot order and stores the result as the parameter gradients.
4. The usual trainer logic then clips the global norm, skips the update if anything is not finite, and applies the optimizer.

One step averages `workers * rounds` micro-batches, so increasing `rounds` is gradient accumulation on top of parallelism.

## Guarantees

- Micro-batches are drawn by the main thread in a fixed order and the slots are summed in slot order, so results do not depend on which worker finishes first. A run is reproducible for a given worker count.
- A test requires the result to match single-threaded accumulation of the same micro-batches within float rounding, and `workers = 1, rounds = 2` to match `workers = 2, rounds = 1`.
- A failing job is reported as an error and the failed worker is replaced.

## Memory

The estimate, excluding activations, is `4 * parameters * (5 + 3 * workers)` bytes. Pass `maxMemoryBytes` to refuse configurations that exceed a budget.

## Measuring

```bash
bun run train:scaling
```

This prints tokens per second for the Tiny configuration with the single-threaded trainer and with 1, 2, 4, 8 and 12 workers.

## Limits

Each step copies the weights out and sums the slots on the main thread, which costs O(workers * parameters). The optimizer state is not sharded.
