import * as T from '@scantling/tensor'
import type { Variable } from './graph.ts'
import { record } from './graph.ts'

/**
 * Softmax along `axis`. The gradient is `y * (g - sum(g * y))` along that axis.
 */
export function softmax(a: Variable, axis = -1): Variable {
	const out = T.softmax(a.value, axis)
	const resolved = T.normalizeAxis(axis, a.shape.length)
	return record(out, [a], (g) => [T.mul(out, T.sub(g, T.sum(T.mul(g, out), { axes: resolved, keepDims: true })))], 'softmax')
}

/**
 * Log-softmax along `axis`. The gradient is `g - softmax * sum(g)` along that axis.
 */
export function logSoftmax(a: Variable, axis = -1): Variable {
	const out = T.logSoftmax(a.value, axis)
	const resolved = T.normalizeAxis(axis, a.shape.length)
	return record(out, [a], (g) => [T.sub(g, T.mul(T.exp(out), T.sum(g, { axes: resolved, keepDims: true })))], 'logSoftmax')
}

export interface CrossEntropyOptions {
	/** Positions whose target equals this value are excluded from the loss. */
	readonly ignoreIndex?: number
}

/**
 * Mean negative log-likelihood of `targets` under `logits` of shape `[..., vocab]`.
 * `targets` holds one class id per row. The result is a scalar; the gradient is
 * `(softmax - onehot) / count` and no one-hot matrix is ever built.
 */
export function crossEntropy(logits: Variable, targets: ArrayLike<number>, options: CrossEntropyOptions = {}): Variable {
	const vocab = logits.shape[logits.shape.length - 1]
	if (vocab === undefined || vocab === 0) throw new RangeError('crossEntropy needs logits with a non-empty last axis')
	const rows = logits.value.size / vocab
	if (targets.length !== rows) throw new RangeError(`Expected ${rows} targets, received ${targets.length}`)
	const ids = Array.from(targets)
	const logp = T.logSoftmax(logits.value.reshape([rows, vocab]), -1)
	let total = 0
	let count = 0
	ids.forEach((target, row) => {
		if (target === options.ignoreIndex) return
		if (!Number.isInteger(target) || target < 0 || target >= vocab) throw new RangeError(`Target ${target} at position ${row} is outside the vocabulary of ${vocab}`)
		total -= logp.data[row * vocab + target]!
		count += 1
	})
	return record(T.Tensor.scalar(count === 0 ? 0 : total / count), [logits], (g) => {
		const scale = count === 0 ? 0 : g.item() / count
		const grad = new Float32Array(rows * vocab)
		ids.forEach((target, row) => {
			if (target === options.ignoreIndex) return
			const base = row * vocab
			for (let index = 0; index < vocab; index += 1) grad[base + index] = Math.exp(logp.data[base + index]!) * scale
			grad[base + target] = grad[base + target]! - scale
		})
		return [new T.Tensor(logits.shape, grad)]
	}, 'crossEntropy')
}
