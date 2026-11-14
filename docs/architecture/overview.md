---
title: Architecture overview
section: Architecture
order: 1
description: Planned domains and what is implemented today.
---

# Architecture overview

Scantling is a Bun workspace written in strict TypeScript. The model stack is built in layers, with each layer tested before the next one depends on it.

## Status of each domain

| Domain                                       | Status      |
| -------------------------------------------- | ----------- |
| Workspace, strict TypeScript, CI, releases   | Implemented |
| Dependency audit                             | Implemented |
| Documentation site                           | Implemented |
| Hardware detection and resource budgets      | Implemented |
| Tensor library and automatic differentiation | Planned     |
| Tokenizer and Transformer                    | Planned     |
| Training and evaluation                      | Planned     |
| Mathematics engine                           | Planned     |
| Repository analysis and coding agent         | Planned     |
| Search, retrieval and fact-checking          | Planned     |
| Memory and conversations                     | Planned     |
| Application UI                               | Planned     |

## Boundaries

- Model and training code runs only on the server, in workers.
- The browser receives a typed API client, never training engines or database code.
- Retrieved web pages, repository contents and model output are treated as untrusted data.
- Memory and training are separate systems. Remembering something never changes model weights.
