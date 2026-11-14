---
title: Getting started
section: Guide
order: 1
description: Install the tooling and verify a fresh checkout.
---

# Getting started

## Requirements

- [Bun](https://bun.com) 1.4 or newer.
- Git.
- Windows 11 is the primary target. The CI runs on Windows and Linux.

## Install

```bash
git clone https://github.com/thenolle/scantling.git
cd scantling
bun install
```

## Verify

```bash
bun run check
```

The command runs three things in order:

| Step   | Command             | Purpose                                      |
| ------ | ------------------- | -------------------------------------------- |
| Types  | `bun run typecheck` | Strict TypeScript check across the workspace |
| Policy | `bun run audit`     | Dependency and pretrained-model audit        |
| Tests  | `bun test`          | Bun's built-in test runner                   |

## Browse the documentation locally

```bash
bun run docs
```

The site is served at `http://localhost:4000`. Set `DOCS_PORT` to use another port.
