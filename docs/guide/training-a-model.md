---
title: Training a model
section: Guide
order: 5
description: Build a tokenizer and dataset from your own code, train the Tiny model and read the results.
---

# Training a model

## 1. Tokenizer

```bash
bun run dataset -- tokenizer --input F:/web --vocab 4096 --out data/tokenizer.json
```

The tokenizer is trained on the same text that survives the dataset exclusions, so a credential repeated across files cannot end up in the vocabulary.

## 2. Dataset

```bash
bun run dataset -- build --input F:/web --tokenizer data/tokenizer.json --out data/datasets/web --verbose
bun run dataset -- build --input F:/web --tokenizer data/tokenizer.json --out data/datasets/web --approve
```

The first command is a dry run that lists every excluded file and why. Only the second writes anything.

## 3. Train

```bash
bun run train -- --dataset data/datasets/web --steps 300
bun run train -- --dataset data/datasets/web --steps 3000 --name tiny-web
```

The run prints a plan first: parameters, tokens, tokens per step and how many passes over the data the run makes. It logs the training loss, the validation loss and perplexity, bits per byte, and the loss of a unigram baseline. Press Ctrl+C to stop: the run finishes the current step and saves a checkpoint. Add `--resume` to continue.

## 4. Read the results

| Number                                | Meaning                                                         |
| ------------------------------------- | --------------------------------------------------------------- |
| Uniform loss (`ln` of the vocabulary) | A model that knows nothing                                      |
| Unigram baseline                      | A model that knows only how common each token is                |
| Validation loss                       | How well the model predicts text it has not trained on          |
| Bits per byte                         | Validation loss in a form that does not depend on the tokenizer |

A checkpoint becomes the stable one only if its validation loss is finite, no worse than the stable one, and below the unigram baseline.

## 5. Generate

```bash
bun run generate -- --prompt "export function"
```
