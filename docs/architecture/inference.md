---
title: Inference
section: Architecture
order: 14
description: Cached decoding, sampling, stopping and context handling.
---

# Inference

## Cached decoding

An `InferenceSession` keeps the attention keys and values of every layer for the tokens already processed. A prompt goes through in one pass, and each further token costs one position. No autodiff graph is built. A test requires the cached logits to match full recomputation for prefill, for one-token steps, and for standard, grouped-query and untied models.

## Sampling

For each step the pipeline applies, in this order:

1. Bans, then the repetition, frequency and presence penalties over the recent tokens.
2. Temperature, then softmax. Temperature 0 selects the most likely token, with the lowest id winning ties, and uses no randomness.
3. Top-k, then top-p, then renormalisation.
4. A draw from a seeded generator, so a seed reproduces a generation exactly.

Every step can report its most likely candidates and their probabilities.

## Stopping

Generation ends at the length limit, on a stop token, on a stop sequence, on cancellation through an `AbortSignal`, or when the window is full and the policy is `stop`. Text that could still become a stop sequence is held back, so a partial match is never shown.

## Context window

A prompt longer than the window is cut to its last tokens. When the window fills, the `slide` policy re-feeds the most recent half of the tokens. Cached keys carry the rotary positions they were computed at, so shifting the cache itself would change what every key means.

## Measuring

```bash
bun run bench:inference
bun run generate -- --untrained --prompt "function"
```

The benchmark reports prefill time, decode speed with the cache, decode speed when recomputing every step, and the ratio.
