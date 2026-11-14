---
title: Dataset pipeline
section: Architecture
order: 11
description: How files become safe, deduplicated, packed token streams for training.
---

# Dataset pipeline

The pipeline turns files you choose into token streams. It never reads anything you did not pass, and it writes nothing until you approve.

## Stages

1. **Collect.** UTF-8 text files, in a stable order. Binary files, files over 1 MiB and build or dependency folders are skipped.
2. **Exclude.** Files with sensitive names (`.env`, private keys, certificates, `.npmrc`) and files in which a credential pattern is found are dropped whole. Credentials are never redacted in place. Email addresses are replaced with `user@example.com`.
3. **Deduplicate.** Files with identical normalised content are kept once.
4. **Split.** A hash of the file path decides train or validation, so a file never changes sides between runs.
5. **Check contamination.** A validation file is dropped if at least half of its lines of 30 or more characters also appear in the training files.
6. **Pack.** Each file becomes `<|file|> tokens <|endfile|>` in one continuous stream. Training windows may span a file boundary.
7. **Store.** Tokens are written as little-endian unsigned 16-bit values behind a 32-byte header, so vocabularies up to 65,536 entries fit and any window can be read without loading the whole file.

## Approval

```bash
bun run dataset -- build --input packages --input docs --out data/datasets/repo
bun run dataset -- build --input packages --input docs --out data/datasets/repo --approve
```

The first command is a dry run that reports counts and the reason for every exclusion. Only the second writes files. The manifest lists every path with its disposition and reason and stores no file contents.

## Limits

Deduplication is exact, so near-duplicates remain. Credential detection is pattern-based and favours excluding, which can drop files that merely mention words like `password`. Windows can span two files; masking attention across `<|file|>` boundaries is not implemented.
