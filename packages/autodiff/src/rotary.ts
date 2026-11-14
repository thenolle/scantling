import * as T from '@scantling/tensor'
import type { Variable } from './graph.ts'
import { record } from './graph.ts'

export interface RopeTables {
	readonly cos: T.Tensor
	readonly sin: T.Tensor
}

export interface RopeOptions {
	/** Frequency base. Defaults to 10000. */
	readonly base?: number
	/** Position of the first row, used when decoding with a cache. Defaults to 0. */
	readonly offset?: number
}

/**
 * Cosine and sine tables of shape `[length, headDim / 2]` for angles `(t + offset) * base^(-2i / headDim)`.
 */
export function ropeTables(length: number, headDim: number, options: RopeOptions = {}): RopeTables {
	if (headDim % 2 !== 0 || headDim < 2) throw new RangeError(`Rotary embeddings need an even head size, received ${headDim}`)
	const base = options.base ?? 10000
	const offset = options.offset ?? 0
	const half = headDim / 2
	const cos = T.Tensor.zeros([length, half])
	const sin = T.Tensor.zeros([length, half])
	for (let position = 0; position < length; position += 1) {
		for (let index = 0; index < half; index += 1) {
			const angle = (position + offset) * Math.pow(base, -(2 * index) / headDim)
			cos.data[position * half + index] = Math.cos(angle)
			sin.data[position * half + index] = Math.sin(angle)
		}
	}
	return { cos, sin }
}

/**
 * Rotary position embedding. `x` is `[..., length, headDim]`; each pair `(x[2i], x[2i+1])` is rotated by
 * the angle for its position. A rotation is orthogonal, so the gradient is the rotation by the negative angle.
 */
export function rope(x: Variable, tables: RopeTables): Variable {
	const rank = x.shape.length
	const headDim = x.shape[rank - 1]
	const length = x.shape[rank - 2]
	if (headDim === undefined || length === undefined || headDim % 2 !== 0) throw new RangeError(`rope needs a tensor of shape [..., length, even head size], received [${x.shape.join(', ')}]`)
	const half = headDim / 2
	if (!T.shapesEqual(tables.cos.shape, [length, half]) || !T.shapesEqual(tables.sin.shape, [length, half])) throw new RangeError(`rope tables must have shape [${length}, ${half}]`)
	const rotate = (source: T.Tensor, inverse: boolean): T.Tensor => {
		const out = new Float32Array(source.size)
		const data = source.data
		const rows = source.size / headDim
		for (let row = 0; row < rows; row += 1) {
			const position = row % length
			const base = row * headDim
			for (let index = 0; index < half; index += 1) {
				const c = tables.cos.data[position * half + index]!
				const s = (inverse ? -1 : 1) * tables.sin.data[position * half + index]!
				const x0 = data[base + 2 * index]!
				const x1 = data[base + 2 * index + 1]!
				out[base + 2 * index] = x0 * c - x1 * s
				out[base + 2 * index + 1] = x0 * s + x1 * c
			}
		}
		return new T.Tensor(source.shape, out)
	}
	return record(rotate(x.value, false), [x], (g) => [rotate(g, true)], 'rope')
}
