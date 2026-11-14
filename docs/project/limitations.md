---
title: Limitations
section: Project
order: 2
description: Honest limits of training a model locally from scratch.
---

# Limitations

Scantling trains its model from random weights on a single laptop, written in TypeScript. That sets firm limits.

- **Model size.** Realistic models are in the range of millions to a few tens of millions of parameters. This is far below the size of assistants trained on large clusters.
- **Speed.** Training runs on the CPU first. GPU acceleration is an experiment and will only be claimed after it runs and matches the CPU results.
- **Quality.** A model this small learns patterns from its training data. It will not match large pretrained assistants at general coding or reasoning.
- **Facts.** Search results and model output are never treated as verified. The fact-checking system reports uncertainty instead of hiding it.

> [!WARNING]
> No benchmark or capability is listed here until it has been measured on real runs.
