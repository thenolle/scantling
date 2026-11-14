# Scantling

[![CI](https://github.com/thenolle/scantling/actions/workflows/ci.yml/badge.svg)](https://github.com/thenolle/scantling/actions/workflows/ci.yml)

A local-first coding assistant built on its own language model, written entirely in TypeScript and run with Bun. The tensor library, automatic differentiation, tokenizer, Transformer, training loop and inference engine live in this repository. Nothing is downloaded, wrapped or fine-tuned from another model.

[Documentation](https://thenolle.github.io/scantling/)

## Status

Early development. The repository currently provides the workspace, strict TypeScript configuration and a dependency audit. The model, training pipeline and application are being added step by step. See the [changelog](./CHANGELOG.md) for what has shipped.

## Principles

- **No third-party runtime dependencies.** The only exceptions are the Gambrel UI framework and, for development, TypeScript and Bun's type definitions. `bun run audit` enforces this.
- **No pretrained weights.** Every model starts from random initialization.
- **Local by default.** Conversations, memory and training data stay on your machine.
- **Measured claims.** Benchmarks and capabilities are documented only after they have been run.

## Requirements

- [Bun](https://bun.com) 1.4 or newer
- Windows 11 is the primary target. Linux and macOS are exercised in CI where applicable.

## Getting started

```bash
git clone https://github.com/thenolle/scantling.git
cd scantling
bun install
bun run check
```

`bun run check` runs the type check, the dependency audit and the tests.

## Scripts

| Command                                 | Purpose                                    |
| --------------------------------------- | ------------------------------------------ |
| `bun run typecheck`                     | Strict TypeScript check                    |
| `bun run audit`                         | Dependency and policy audit                |
| `bun test`                              | Run the test suite                         |
| `bun run check`                         | All of the above                           |
| `bun run release <major\|minor\|patch>` | Prepare a version bump and changelog entry |
| `bun run hooks`                         | Enable the commit message hook             |

## Contributing

Read [CONTRIBUTING.md](./CONTRIBUTING.md) first. Report vulnerabilities as described in [SECURITY.md](./SECURITY.md).

## Support

If this project is useful to you, you can [sponsor its development](https://github.com/sponsors/thenolle).

## Licence

[PolyForm Strict License 1.0.0](./LICENSE). It permits noncommercial use and does not permit redistribution or derivative works. Contact the maintainer for any rights it does not grant.
