---
title: Transformer
section: Architecture
order: 9
description: The decoder-only Transformer, its components, formulas, initialisation and parameter counts.
---

# Transformer

The model is a decoder-only Transformer. Every component is differentiated by the autodiff engine and checked against finite differences, both per operation and through the whole network.

## Components

| Component        | Definition                                                                              |
| ---------------- | --------------------------------------------------------------------------------------- |
| Token embedding  | A `[vocab, dim]` table; tied with the output projection by default                      |
| RMSNorm          | `y = x / sqrt(mean(x^2) + eps) * w` over the last axis                                  |
| Rotary embedding | Pairs `(x[2i], x[2i+1])` rotated by `t * base^(-2i/headDim)`                            |
| Attention        | `softmax(mask(Q K^T / sqrt(headDim))) V`, with optional grouped-query key/value sharing |
| SwiGLU           | `down(silu(gate(x)) * up(x))`, hidden size about 8/3 of the width                       |
| Block            | `x + attention(norm(x))`, then `x + feedForward(norm(x))`                               |
| Output           | `logits = norm(x) @ E^T` when tied, otherwise a separate projection                     |

## Initialisation

All weights are normal with standard deviation 0.02 from a seeded generator. Projections that write into the residual stream (attention output and feed-forward down) use `0.02 / sqrt(2 * layers)` so the stream's variance does not grow with depth. Norm scales start at 1. The same configuration and seed always produce the same initial model.

## Sizes

Parameter counts follow from the configuration, with a vocabulary of 4096:

| Preset | Width | Layers | Heads | Context | Parameters           |
| ------ | ----- | ------ | ----- | ------- | -------------------- |
| Micro  | 64    | 2      | 4     | 128     | see `parameterCount` |
| Tiny   | 128   | 4      | 4     | 256     | 1,315,968            |
| Small  | 384   | 6      | 6     | 512     | about 12.2 million   |

## Design choices and how they are validated

RoPE, RMSNorm, SwiGLU, grouped-query attention and weight tying are standard modern choices, adopted on published evidence rather than measured here. Ablation runs on this machine will compare each against its plain alternative once training works, and the results will be recorded in the project documentation.

> [!NOTE]
> There is no dropout and no KV cache yet. The cache is added with the inference engine, and dropout when overfitting is measured.
