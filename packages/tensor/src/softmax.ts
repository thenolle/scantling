import { normalizeAxis } from './shape.ts'
import { Tensor } from './tensor.ts'

interface Layout {
	readonly outer: number
	readonly length: number
	readonly inner: number
}

function layoutOf(t: Tensor, axis: number): Layout {
	const resolved = normalizeAxis(axis, t.rank)
	let outer = 1
	for (let index = 0; index < resolved; index += 1) outer *= t.shape[index]!
	let inner = 1
	for (let index = resolved + 1; index < t.rank; index += 1) inner *= t.shape[index]!
	return { outer, length: t.shape[resolved]!, inner }
}

function maxAlong(x: Float32Array, base: number, length: number, inner: number): number {
	let best = Number.NEGATIVE_INFINITY
	for (let k = 0; k < length; k += 1) {
		const value = x[base + k * inner]!
		if (value > best) best = value
	}
	return best
}

/**
 * Softmax along `axis`: `exp(x_i - m) / sum_j exp(x_j - m)` with `m = max(x)`.
 * Subtracting the maximum keeps the exponentials in range for any finite input.
 * A slice that is entirely `-Infinity` has no defined softmax and yields NaN.
 */
export function softmax(t: Tensor, axis = -1): Tensor {
	const { outer, length, inner } = layoutOf(t, axis)
	const x = t.data
	const out = new Float32Array(t.size)
	for (let o = 0; o < outer; o += 1) {
		for (let i = 0; i < inner; i += 1) {
			const base = o * length * inner + i
			const top = maxAlong(x, base, length, inner)
			let total = 0
			for (let k = 0; k < length; k += 1) {
				const index = base + k * inner
				const e = Math.fround(Math.exp(x[index]! - top))
				out[index] = e
				total += e
			}
			for (let k = 0; k < length; k += 1) {
				const index = base + k * inner
				out[index] = out[index]! / total
			}
		}
	}
	return new Tensor(t.shape, out)
}

/**
 * Log-softmax along `axis`: `x_i - m - log(sum_j exp(x_j - m))`, stable for any finite input.
 */
export function logSoftmax(t: Tensor, axis = -1): Tensor {
	const { outer, length, inner } = layoutOf(t, axis)
	const x = t.data
	const out = new Float32Array(t.size)
	for (let o = 0; o < outer; o += 1) {
		for (let i = 0; i < inner; i += 1) {
			const base = o * length * inner + i
			const top = maxAlong(x, base, length, inner)
			let total = 0
			for (let k = 0; k < length; k += 1) total += Math.exp(x[base + k * inner]! - top)
			const offset = top + Math.log(total)
			for (let k = 0; k < length; k += 1) {
				const index = base + k * inner
				out[index] = x[index]! - offset
			}
		}
	}
	return new Tensor(t.shape, out)
}

/**
 * `log(sum(exp(x)))` along `axis`, computed as `m + log(sum(exp(x - m)))`.
 */
export function logSumExp(t: Tensor, axis = -1, keepDims = false): Tensor {
	const { outer, length, inner } = layoutOf(t, axis)
	const resolved = normalizeAxis(axis, t.rank)
	const x = t.data
	const out = new Float32Array(outer * inner)
	for (let o = 0; o < outer; o += 1) {
		for (let i = 0; i < inner; i += 1) {
			const base = o * length * inner + i
			const top = maxAlong(x, base, length, inner)
			let total = 0
			for (let k = 0; k < length; k += 1) total += Math.exp(x[base + k * inner]! - top)
			out[o * inner + i] = top + Math.log(total)
		}
	}
	const shape = keepDims ? t.shape.map((dimension, index) => (index === resolved ? 1 : dimension)) : t.shape.filter((_, index) => index !== resolved)
	return new Tensor(shape, out)
}
