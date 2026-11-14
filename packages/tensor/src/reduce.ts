import { mulScalar } from './elementwise.ts'
import type { Shape } from './shape.ts'
import { normalizeAxis, shapesEqual } from './shape.ts'
import { Tensor } from './tensor.ts'

export interface ReduceOptions {
	/** One axis, several axes, or all axes when omitted. Negative axes count from the end. */
	readonly axes?: number | readonly number[]
	/** Keep reduced axes as size 1 instead of removing them. */
	readonly keepDims?: boolean
}

type Kind = 'sum' | 'max' | 'min'

function resolveAxes(rank: number, axes: number | readonly number[] | undefined): number[] {
	if (axes === undefined) return Array.from({ length: rank }, (_, index) => index)
	const list = (typeof axes === 'number' ? [axes] : axes).map((axis) => normalizeAxis(axis, rank))
	return [...new Set(list)].sort((a, b) => b - a)
}

function reduceOne(t: Tensor, axis: number, kind: Kind): Tensor {
	const length = t.shape[axis]!
	let outer = 1
	for (let index = 0; index < axis; index += 1) outer *= t.shape[index]!
	let inner = 1
	for (let index = axis + 1; index < t.rank; index += 1) inner *= t.shape[index]!
	const shape = t.shape.filter((_, index) => index !== axis)
	if (length === 0 && kind !== 'sum') throw new RangeError(`Cannot take the ${kind} of an empty axis`)
	const x = t.data
	const out = new Float32Array(outer * inner)
	const acc = new Float64Array(inner)
	for (let o = 0; o < outer; o += 1) {
		const base = o * length * inner
		if (kind === 'sum') {
			acc.fill(0)
			for (let k = 0; k < length; k += 1) {
				const row = base + k * inner
				for (let i = 0; i < inner; i += 1) acc[i] = acc[i]! + x[row + i]!
			}
		} else {
			for (let i = 0; i < inner; i += 1) acc[i] = x[base + i]!
			for (let k = 1; k < length; k += 1) {
				const row = base + k * inner
				for (let i = 0; i < inner; i += 1) {
					const value = x[row + i]!
					const current = acc[i]!
					if (kind === 'max' ? value > current || value !== value : value < current || value !== value) acc[i] = value
				}
			}
		}
		out.set(acc, o * inner)
	}
	return new Tensor(shape, out)
}

function reduce(t: Tensor, kind: Kind, options: ReduceOptions): Tensor {
	const axes = resolveAxes(t.rank, options.axes)
	if (axes.length === 0) return t.clone()
	let result = t
	for (const axis of axes) result = reduceOne(result, axis, kind)
	if (options.keepDims === true) return result.reshape(t.shape.map((dimension, index) => (axes.includes(index) ? 1 : dimension)))
	return result
}

/** Sum, accumulated in 64-bit floats. */
export const sum = (t: Tensor, options: ReduceOptions = {}): Tensor => reduce(t, 'sum', options)

/** Maximum. NaN propagates. Throws on an empty axis. */
export const max = (t: Tensor, options: ReduceOptions = {}): Tensor => reduce(t, 'max', options)

/** Minimum. NaN propagates. Throws on an empty axis. */
export const min = (t: Tensor, options: ReduceOptions = {}): Tensor => reduce(t, 'min', options)

/** Arithmetic mean over the given axes. */
export function mean(t: Tensor, options: ReduceOptions = {}): Tensor {
	const axes = resolveAxes(t.rank, options.axes)
	const count = axes.reduce((product, axis) => product * t.shape[axis]!, 1)
	return mulScalar(sum(t, options), 1 / count)
}

/**
 * Index of the largest element along `axis`, returned as a tensor of whole numbers.
 * Ties resolve to the lowest index.
 */
export function argmax(t: Tensor, axis = -1): Tensor {
	const resolved = normalizeAxis(axis, t.rank)
	const length = t.shape[resolved]!
	if (length === 0) throw new RangeError('Cannot take the argmax of an empty axis')
	let outer = 1
	for (let index = 0; index < resolved; index += 1) outer *= t.shape[index]!
	let inner = 1
	for (let index = resolved + 1; index < t.rank; index += 1) inner *= t.shape[index]!
	const x = t.data
	const out = new Float32Array(outer * inner)
	for (let o = 0; o < outer; o += 1) {
		for (let i = 0; i < inner; i += 1) {
			const base = o * length * inner + i
			let best = 0
			let bestValue = x[base]!
			for (let k = 1; k < length; k += 1) {
				const value = x[base + k * inner]!
				if (value > bestValue) {
					bestValue = value
					best = k
				}
			}
			out[o * inner + i] = best
		}
	}
	return new Tensor(t.shape.filter((_, index) => index !== resolved), out)
}

/**
 * Sums a broadcast result back down to `target`, the shape it was broadcast from.
 * This is the gradient of broadcasting.
 */
export function sumToShape(t: Tensor, target: Shape): Tensor {
	if (shapesEqual(t.shape, target)) return t
	const lead = t.rank - target.length
	if (lead < 0) throw new RangeError(`Cannot sum [${t.shape.join(', ')}] down to [${target.join(', ')}]`)
	const axes: number[] = []
	for (let axis = 0; axis < t.rank; axis += 1) {
		if (axis < lead) {
			axes.push(axis)
			continue
		}
		const wanted = target[axis - lead]!
		if (wanted === t.shape[axis]) continue
		if (wanted !== 1) throw new RangeError(`Cannot sum [${t.shape.join(', ')}] down to [${target.join(', ')}]`)
		axes.push(axis)
	}
	return sum(t, { axes, keepDims: true }).reshape(target)
}
