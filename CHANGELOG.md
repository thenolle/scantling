# Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Client-rendered documentation site built from the Markdown files in `docs/`.
- Automatic publishing of the site to the `website` branch.
- Hardware detection and resource budgeting for memory, workers and disk, with a CLI report.
- Validated layered configuration with environment overrides and a local-only server default.
- Structured logging with secret redaction, file rotation and an in-memory buffer.
- Tensor library with broadcasting, reductions, matrix products, softmax and seeded random numbers.
- Documentation pages are prerendered with canonical URLs, Open Graph and X metadata, per-page social images, structured data, a sitemap and a web manifest.
- Documentation styling follows shadcn's neutral theme tokens.
- Reverse-mode automatic differentiation with broadcasting-aware gradients, gradient clipping and a finite-difference gradient checker.
- Differentiable activations, softmax, cross-entropy, embeddings, slicing, concatenation, masking, dropout, RMS normalisation and rotary embeddings.
- Worker-parallel matrix products on shared memory and a benchmark suite that records the machine profile.
- Byte-level BPE tokenizer with a code-aware pre-tokenizer, deterministic training, streaming decode and validated serialisation.
- Measured benchmark results for matrix products on the development machine.
- Decoder-only Transformer with rotary embeddings, grouped-query attention, SwiGLU, RMS normalisation, weight tying and a whole-model gradient check.
- AdamW and SGD optimizers, warmup-cosine scheduling, gradient accumulation with clipping and skip-on-non-finite, and a cancellable trainer.
- Dataset pipeline that excludes sensitive files and credentials, deduplicates, splits deterministically, checks contamination and stores packed tokens in a streamable binary format.
- Atomic, checksummed checkpoints with exact resume, and a promotion gate that requires evaluation and regression results and keeps the previous stable checkpoint recoverable.
- Data-parallel training across worker threads, reproducible for a given worker count and equivalent to gradient accumulation.
- KV-cached inference with seeded sampling, stop handling, cancellation and context sliding.
- Training runner that builds the model from a dataset, trains in parallel, evaluates against a unigram baseline, checkpoints and applies the promotion gate.
- Allocation-free single-token decoder with optional int8 weights.
- WebAssembly SIMD matrix kernel assembled from TypeScript, with automatic fallback, a profiler for training steps and memory diagnostics in the training runner.
- Model quality checks: next-token accuracy, a greedy completion test against held-out code, and an `evaluate` command.

### Changed

- Matrix kernels are unrolled and register-blocked; data-parallel training dispatches all micro-batches at once and reports phase timings.
- Dataset scanning flags only secret-looking literals in source code and skips generated files.

### Fixed

- Dependency audit no longer flags string literals that contain the word import.

## [0.1.0] - 2026-11-14

### Added

- Bun workspace with strict TypeScript configuration.
- Dependency audit covering manifests, imports, lockfile baseline, pretrained-model formats and external AI endpoints.
- Continuous integration on Linux and Windows.
- Release automation driven by the version in `package.json`.
- Commit message linting.
