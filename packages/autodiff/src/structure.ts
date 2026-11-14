import * as T from '@scantling/tensor'
import type { Variable } from './graph.ts'
import { record } from './graph.ts'

/**
 * Looks up rows of a `[vocab, width]` table. The result is `[ids.length, width]`.
 * The gradient is a dense table of the same shape with each row's gradient added back.
 */
export function embedding(table: Variable, ids: ArrayLike<number>): Variable {
	const rows = Array.from(ids)
	return record(T.gatherRows(table.value, rows), [table], (g) => [T.scatterAddRowsInPlace(T.Tensor.zeros(table.shape), rows, g)], 'embedding')
}

/** Copies the range `[start, end)` along `axis`. */
export function sliceAxis(a: Variable, axis: number, start: number, end: number): Variable {
	const resolved = T.normalizeAxis(axis, a.shape.length)
	const length = a.shape[resolved]!
	const zeros = (size: number): T.Tensor => T.Tensor.zeros(a.shape.map((dimension, index) => (index === resolved ? size : dimension)))
	return record(T.sliceAxis(a.value, resolved, start, end), [a], (g) => {
		const pieces: T.Tensor[] = []
		if (start > 0) pieces.push(zeros(start))
		pieces.push(g)
		if (end < length) pieces.push(zeros(length - end))
		return [pieces.length === 1 ? g : T.concat(pieces, resolved)]
	}, 'sliceAxis')
}

/** Joins variables along `axis`. */
export function concat(parts: readonly Variable[], axis: number): Variable {
	const first = parts[0]
	if (first === undefined) throw new RangeError('concat needs at least one input')
	const resolved = T.normalizeAxis(axis, first.shape.length)
	return record(T.concat(parts.map((part) => part.value), resolved), parts, (g) => {
		let offset = 0
		return parts.map((part) => {
			const size = part.shape[resolved]!
			const piece = part.requiresGrad ? T.sliceAxis(g, resolved, offset, offset + size) : null
			offset += size
			return piece
		})
	}, 'concat')
}

/**
 * Replaces elements where `mask` is non-zero with `value`. `mask` must broadcast to the shape of `a`,
 * for example a `[T, T]` causal mask against `[batch, T, T]` scores. Masked positions get no gradient.
 */
export function maskedFill(a: Variable, mask: T.Tensor, value: number): Variable {
	if (!T.shapesEqual(T.broadcastShapes(a.shape, mask.shape), a.shape)) throw new RangeError(`A mask of shape [${mask.shape.join(', ')}] cannot be broadcast to [${a.shape.join(', ')}]`)
	const keep = T.map(mask, (flag) => (flag !== 0 ? 0 : 1))
	return record(T.zip(a.value, mask, (x, flag) => (flag !== 0 ? value : x)), [a], (g) => [T.mul(g, keep)], 'maskedFill')
}

/**
 * Inverted dropout: each element is zeroed with probability `p` and the rest are scaled by `1 / (1 - p)`.
 * Returns the input unchanged when `training` is false or `p` is 0.
 */
export function dropout(a: Variable, p: number, random: T.Random, training = true): Variable {
	if (!(p >= 0 && p < 1)) throw new RangeError(`Dropout probability ${p} must be in [0, 1)`)
	if (!training || p === 0) return a
	const scale = 1 / (1 - p)
	const mask = T.Tensor.zeros(a.shape)
	for (let index = 0; index < mask.size; index += 1) mask.data[index] = random.next() < p ? 0 : scale
	return record(T.mul(a.value, mask), [a], (g) => [T.mul(g, mask)], 'dropout')
}
