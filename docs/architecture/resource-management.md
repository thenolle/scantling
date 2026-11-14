---
title: Resource management
section: Architecture
order: 3
description: How Scantling detects hardware and budgets memory, workers and disk.
---

# Resource management

The `hardware` package takes a snapshot of the machine and turns it into a budget that every long-running job must respect.

## What is detected

| Resource                                                          | Source                                            |
| ----------------------------------------------------------------- | ------------------------------------------------- |
| CPU model, architecture, logical cores                            | Bun and `node:os`                                 |
| Total and available memory                                        | `node:os`                                         |
| Disk total and free space                                         | `fs.statfs` on the volume of each configured path |
| Disk write speed                                                  | Optional 64 MiB synced write, run only on request |
| NVIDIA GPU, VRAM, driver, compute capability, CUDA driver version | `nvidia-smi` when it is installed                 |

Logical cores are reported as they are. Bun does not expose which cores are performance or efficiency cores.

## How the budget is computed

| Value            | Rule                                                                          |
| ---------------- | ----------------------------------------------------------------------------- |
| Usable memory    | The smaller of available memory minus a 3 GiB reserve and 60% of total memory |
| Inference memory | The smaller of usable memory and 25% of total memory                          |
| Worker threads   | Logical cores minus 2, at most 8, at least 1                                  |
| Disk headroom    | Free space minus 10 GiB on the tightest checked volume                        |

The defaults can be overridden and are validated. The worker default is provisional until it is benchmarked.

## Refusing oversized jobs

Before a training job starts, its estimated memory and disk needs are checked with `assertMemoryFits` and `assertDiskFits`. A job that does not fit is rejected with a `ResourceExceededError` that states the required and budgeted amounts, so a large configuration never takes down the application.

> [!NOTE]
> A detected GPU is reported but not used. The accelerator in the budget is always `cpu` until a GPU backend exists and produces results that match the CPU reference.

## Inspect your machine

```bash
bun run hardware
bun run hardware -- --storage --path F:/
bun run hardware -- --json
```
