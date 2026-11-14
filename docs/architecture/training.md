---
title: Training
section: Architecture
order: 10
description: Optimizers, learning-rate schedules, gradient accumulation, safety checks and the overfit test.
---

# Training

## Optimizers

AdamW keeps bias-corrected first and second moments of each gradient:

```text
m = b1 * m + (1 - b1) * g          v = b2 * v + (1 - b2) * g^2
p = p - lr * ( (m / (1 - b1^t)) / (sqrt(v / (1 - b2^t)) + eps)  +  wd * p )
```

Decay is decoupled from the gradient and applied only to parameters of rank 2 or more, so norm scales are never decayed. The defaults are `b1 = 0.9`, `b2 = 0.95`, `eps = 1e-8` and `wd = 0.1`. SGD with momentum (`v = mu * v + g`) is available as a baseline for comparison.

## Learning-rate schedule

Linear warmup to a peak, then cosine decay to a floor of 10% of the peak:

```text
lr = min + (peak - min) * 0.5 * (1 + cos(pi * progress))
```

## A training step

1. Zero the gradients.
2. For each of `accumulationSteps` micro-batches, run the forward pass and call `backward` with a seed of `1 / accumulationSteps`, so the accumulated gradient is the mean.
3. Clip the global gradient norm to `maxGradNorm` and record the norm measured before clipping.
4. If the loss or the norm is not finite, skip the update and count it. Otherwise apply the optimizer with the scheduled learning rate.

## Evaluation and control

`evaluate` averages the loss over fixed, non-overlapping validation windows without recording a graph, and reports perplexity (`exp(loss)`). `run` can be stopped with an `AbortSignal`, which ends the run cleanly after the current step.

## Overfit check

```bash
bun run train:overfit
```

This trains a small model from random weights on a repeating 16-token sequence. A model that has learned nothing scores `ln(vocabulary)`; learning anything drives the loss well below that. The measured run is recorded below once it has been executed.
