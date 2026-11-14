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

### Fixed

- Dependency audit no longer flags string literals that contain the word import.

## [0.1.0] - 2026-11-14

### Added

- Bun workspace with strict TypeScript configuration.
- Dependency audit covering manifests, imports, lockfile baseline, pretrained-model formats and external AI endpoints.
- Continuous integration on Linux and Windows.
- Release automation driven by the version in `package.json`.
- Commit message linting.
