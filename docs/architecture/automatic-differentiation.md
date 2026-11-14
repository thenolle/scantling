---
title: Automatic differentiation
section: Architecture
order: 6
description: How the reverse-mode engine records a graph, propagates gradients and is validated against finite differences.
---

# Automatic differentiation

The `autodiff` package computes gradients of scalar objectives with reverse-mode differentiation.

## How it works

Every operation returns a `Variable` that remembers its parents and a function that maps the gradient of its output to gradients for its inputs. `backward(root)` orders the graph topologically, then walks it from the root, applying the chain rule at each node:

```text
dL/dx = sum over consumers y of x:  dL/dy * dy/dx
```

When a value feeds several operations, the contributions are added. When an operation broadcasts an input, the gradient is summed back to the input's shape with `sumToShape`.

## Gradient rules

| Operation                                 | Gradient with respect to inputs                                                                              |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `a + b`                                   | `g` to both, summed to each input's shape                                                                    |
| `a * b`                                   | `g * b` and `g * a`                                                                                          |
| `a / b`                                   | `g / b` and `-g * a / b^2`                                                                                   |
| `exp(a)`                                  | `g * exp(a)`                                                                                                 |
| `log(a)`                                  | `g / a`                                                                                                      |
| `sqrt(a)`                                 | `g / (2 * sqrt(a))`                                                                                          |
| `A @ B`                                   | `g @ B^T` and `A^T @ g`, using transpose flags instead of copies                                             |
| `sum`, `mean`                             | `g` expanded to the input shape, divided by the count for `mean`                                             |
| `reshape`, `permute`, `transpose`         | `g` reshaped or permuted back                                                                                |
| `relu`, `sigmoid`, `tanh`, `silu`, `gelu` | `g` times the pointwise derivative                                                                           |
| `softmax`                                 | `y * (g - sum(g * y))` along the axis                                                                        |
| `logSoftmax`                              | `g - softmax * sum(g)` along the axis                                                                        |
| `crossEntropy`                            | `(softmax - onehot) / count`, with no one-hot matrix                                                         |
| `embedding`                               | The rows of `g` added back into a zero table                                                                 |
| `sliceAxis`, `concat`                     | `g` zero-padded, or `g` sliced per input                                                                     |
| `maskedFill`                              | `g` where the mask is zero, otherwise 0                                                                      |
| `rmsNorm`                                 | `dx = r * (dx̂ - x̂ * mean(dx̂ * x̂))` and `dw = sum(g * x̂)`, where `r = 1 / sqrt(mean(x²) + ε)` and `x̂ = x * r` |
| `rope`                                    | `g` rotated by the negative angle                                                                            |

## Safety properties

- Gradients are never added in place into a tensor that another node may share. Only a parameter's own `grad` buffer is accumulated in place.
- Every gradient is checked against its input's shape during `backward`.
- `noGrad(() => ...)` builds no graph, which keeps inference cheap.
- `clipGradNorm` scales gradients to a maximum global norm and leaves them untouched when the norm is not finite, so the caller can skip the step.

## Validation

`gradcheck` compares the analytic gradient with central differences for every element of every input:

```text
numeric_i = (L(x_i + h) - L(x_i - h)) / (2h)
```

For a non-scalar output, the loss is a fixed random weighted sum of the outputs, accumulated in 64-bit floats. Because tensors are Float32, the default step is `h = 0.01` and the default tolerance is `1e-2` relative plus `1e-3` absolute. A deliberately wrong gradient is caught by a test.

## Not yet implemented

Gradient checkpointing is added once the Transformer exists and its activation memory can be measured. Operations are added only when a model component needs them, each with a finite-difference check.
