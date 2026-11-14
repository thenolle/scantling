---
title: Checkpoints
section: Architecture
order: 12
description: What a checkpoint contains, how it is written safely and how a checkpoint becomes stable.
---

# Checkpoints

## Contents

A checkpoint is a directory named after its step, for example `step-00001200`:

| File          | Holds                                                                                                                                       |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `tensors.bin` | The model weights followed by the optimizer state, as raw 32-bit floats                                                                     |
| `meta.json`   | The model configuration, step counters, the sampler's random state, metrics, and a SHA-256 checksum for every tensor and for the whole file |

Resuming restores all of it, so a resumed run produces exactly the same weights as one that was never interrupted. A test trains 12 steps straight, then 6 steps, saves to disk, reloads into fresh objects and trains 6 more, and requires bit-identical weights.

## Writing safely

A checkpoint is assembled in a temporary directory with every file flushed to disk, then renamed into place. A crash leaves either the previous state or a complete new checkpoint, never a half-written one. Existing checkpoints are never overwritten, and loading verifies every checksum and refuses a corrupt file.

## Promotion

Only evaluation can make a checkpoint the stable one. The gate requires:

- A finite evaluation loss.
- An evaluation loss no more than 0.5% worse than the current stable checkpoint.
- Every regression check passed, and at least one provided.

Training loss is not considered. Every decision, accepted or rejected, is appended to `promotions.jsonl`. The previous stable checkpoint remains on disk, and `rollback` makes it stable again. Pruning keeps the newest checkpoints plus the stable one and every checkpoint that was ever promoted.

## Inspecting

```bash
bun run checkpoint -- list
bun run checkpoint -- verify --id step-00001200
bun run checkpoint -- history
```

> [!NOTE]
> Checkpoints are not compressed yet. That is added with the performance work.
