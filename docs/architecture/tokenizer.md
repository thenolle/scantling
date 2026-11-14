---
title: Tokenizer
section: Architecture
order: 8
description: The byte-level BPE tokenizer, its pre-tokenizer, special tokens and guarantees.
---

# Tokenizer

The tokenizer is byte-level BPE, written from scratch. Its base vocabulary is the 256 byte values, so every input is representable and there is no unknown token.

## Token ids

| Range        | Meaning                                                                        |
| ------------ | ------------------------------------------------------------------------------ |
| `0 .. S-1`   | Special tokens: pad, bos, eos, file, endfile, system, user, assistant, endturn |
| `S .. S+255` | The 256 byte values                                                            |
| `S+256 ..`   | Learned merges, in the order they were learned                                 |

## Pre-tokenization

Text is split into pieces before merging, and merges never cross a piece boundary. Pieces are identifiers and words (underscores included), single digits, punctuation runs and whitespace runs, each with at most one leading space. Whitespace runs stay together, so indentation can become a single token. Pieces over 1024 bytes are chunked to bound the cost of merging.

## Guarantees

- Decoding the encoding of any valid text returns the original text. A lone UTF-16 surrogate cannot be encoded as UTF-8 and is the one exception.
- Training is deterministic: the same corpus and options always produce the same file.
- Text such as `<|endturn|>` in user input is encoded as ordinary text unless the caller explicitly allows special tokens, so a document cannot inject control tokens.
- Loading validates the format, version and every merge.

## Training and inspecting

```bash
bun run tokenizer -- train --input packages --input docs --vocab 4096 --out data/tokenizer.json
bun run tokenizer -- inspect --tokenizer data/tokenizer.json --input website
```

Only the paths you pass are read. Binary files, files over 1 MiB and common build folders are skipped.
