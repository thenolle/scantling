---
title: Tensor library
section: Architecture
order: 5
description: Memory layout, broadcasting, precision rules and the operations the tensor package provides.
---

# Tensor library

The `tensor` package is the numerical base for automatic differentiation and the language model.

## Layout

A tensor is a contiguous, row-major `Float32Array` plus a shape. `reshape` returns a view over the same memory. `permute`, `slice` and `concat` copy. Keeping every tensor contiguous means no operation has to track strides, and the matrix kernels read memory in order.

## Broadcasting

Shapes are aligned from the trailing dimension. Each pair of dimensions must be equal, or one of them must be 1. Broadcast dimensions are read with stride 0. The gradient of a broadcast is `sumToShape`, which sums the result back down to the source shape.

## Precision

Values are stored as 32-bit floats. Reductions, matrix products, `softmax`, `logSoftmax` and `logSumExp` accumulate in 64-bit floats and round once when the result is stored. Softmax subtracts the maximum before exponentiating:

```text
softmax(x)_i = exp(x_i - m) / sum_j exp(x_j - m),   m = max_j x_j
logsumexp(x) = m + log(sum_j exp(x_j - m))
```

A slice that is entirely `-Infinity` has no defined softmax and produces NaN. Use `assertFinite` to catch non-finite values early.

## Matrix products

`matmul(a, b)` multiplies the last two axes and broadcasts the leading batch axes, so a `[batch, time, width]` input can be multiplied by a `[width, out]` weight directly. The `transposeA` and `transposeB` options read an operand as stored transposed. Backpropagation uses them to compute `dC @ Bᵀ` and `Aᵀ @ dC` without copying.

## Randomness

`Random` is a seeded xoshiro128** generator. The same seed gives the same numbers on every machine, and `state()` and `restore()` let a training run resume with exactly the same stream.

## Limits

- CPU only, single thread. Worker parallelism and benchmarks come after the autodiff engine.
- Only `float32`. Lower-precision and quantized storage are considered once there are measurements to justify them.
