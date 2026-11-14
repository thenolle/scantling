---
title: Dependency policy
section: Architecture
order: 2
description: What the dependency audit enforces.
---

# Dependency policy

Scantling uses no third-party runtime libraries. `bun run audit` checks this on every push.

## Allowed

- Bun and Node-compatible built-ins (`bun:*`, `node:*`).
- The Gambrel framework packages under `@gambrel/*`.
- `typescript` and `@types/bun` as development dependencies.

## What the audit rejects

| Rule         | Rejected                                                                              |
| ------------ | ------------------------------------------------------------------------------------- |
| `dependency` | A package outside the allowlist, or `overrides`, `resolutions`, `trustedDependencies` |
| `lifecycle`  | `preinstall`, `install` or `postinstall` scripts                                      |
| `import`     | A bare import that is not allowed                                                     |
| `endpoint`   | A hardcoded host of an external AI service                                            |
| `model-file` | Pretrained formats such as `.safetensors`, `.gguf` and `.onnx`                        |
| `lockfile`   | A locked package missing from `tools/audit.baseline.json`                             |

The import check is lexical, so it can occasionally flag text that only resembles an import.

## Changing the baseline

When a reviewed change adds a package to `bun.lock`:

```bash
bun tools/audit-deps.ts --write-baseline
git diff tools/audit.baseline.json
```

Review the diff before committing it.
