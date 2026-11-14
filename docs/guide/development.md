---
title: Development workflow
section: Guide
order: 2
description: Scripts, commit conventions and how releases are made.
---

# Development workflow

## Scripts

| Command                                 | Purpose                                   |
| --------------------------------------- | ----------------------------------------- |
| `bun run check`                         | Type check, audit and tests               |
| `bun run docs`                          | Serve the documentation site              |
| `bun run docs:build`                    | Build the static site into `website/dist` |
| `bun run hooks`                         | Enable the commit message hook            |
| `bun run release <major\|minor\|patch>` | Prepare a version bump                    |

## Code conventions

- Strict TypeScript with ESM, tab indentation, no semicolons and single quotes.
- Small cohesive modules, with JSDoc on public APIs.
- No new dependencies. See the [dependency policy](../architecture/dependency-policy.md).

## Commit messages

Use `type(scope): description` in lower case, without a trailing period, up to 72 characters. Allowed types are `feat`, `fix`, `docs`, `chore`, `ci`, `build`, `test`, `perf`, `refactor` and `revert`. The hook installed by `bun run hooks` and the CI check apply the same rules.

## Releases

1. Add entries under `Unreleased` in `CHANGELOG.md`.
2. Run `bun run release minor` (or `major` or `patch`).
3. Commit the result as `chore(release): vX.Y.Z` and push to `master`.

A workflow then creates the GitHub release from the matching changelog section. Versions below 1.0.0 are published as pre-releases.
