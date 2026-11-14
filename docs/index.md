---
title: Scantling
seoTitle: Scantling | Local-first AI coding assistant, from scratch
section: Overview
order: 1
description: What Scantling is and where the project stands.
---

# Scantling

Scantling is a local-first coding assistant built around a language model that is written, trained and run entirely inside this repository. The tensor library, automatic differentiation, tokenizer, Transformer and training loop are all TypeScript running on Bun. There are no pretrained weights and no external AI services.

> [!NOTE]
> Scantling is in early development. This site documents what exists today and marks planned work as planned.

## Principles

- No third-party runtime dependencies. [The dependency policy](architecture/dependency-policy.md) is enforced by an automated audit.
- Every model starts from random initialization.
- Conversations, memory and training data stay on your machine.
- Capabilities and benchmarks are documented only after they have been measured.

## Where to go next

- [Getting started](guide/getting-started.md) to install and verify the project.
- [Development workflow](guide/development.md) for scripts, commits and releases.
- [Architecture overview](architecture/overview.md) for the planned system layout.
- [Limitations](project/limitations.md) for what a locally trained model can and cannot do.
