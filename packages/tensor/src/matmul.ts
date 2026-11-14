import { broadcastShapes, broadcastStrides, sizeOf } from './shape.ts'
import type { Shape } from './shape.ts'
import { Tensor } from './tensor.ts'

export interface MatmulOptions {
	/** Treat `a` as stored transposed: its last two axes are `[k, m]`. */
	readonly transposeA?: boolean
	/** Treat `b` as stored transposed: its last two axes are `[n, k]`. */
	readonly transposeB?: boolean
}

function batchOffsets(batch: Shape, own: Shape, matrixSize: number): number[] {
	const count = sizeOf(batch)
	const strides = broadcastStrides(own, batch)
	const offsets = new Array<number>(count)
	for (let flat = 0; flat < count; flat += 1) {
		let rest = flat
		let offset = 0
		for (let axis = batch.length - 1; axis >= 0; axis -= 1) {
			const dimension = batch[axis]!
			offset += (rest % dimension) * strides[axis]!
			rest = Math.floor(rest / dimension)
		}
		offsets[flat] = offset * matrixSize
	}
	return offsets
}

type Kernel = (a: Float32Array, ao: number, b: Float32Array, bo: number, c: Float32Array, co: number, m: number, k: number, n: number, acc: Float64Array) => void

const nn: Kernel = (a, ao, b, bo, c, co, m, k, n, acc) => {
	for (let i = 0; i < m; i += 1) {
		acc.fill(0)
		for (let p = 0; p < k; p += 1) {
			const value = a[ao + i * k + p]!
			const row = bo + p * n
			for (let j = 0; j < n; j += 1) acc[j] = acc[j]! + value * b[row + j]!
		}
		const target = co + i * n
		for (let j = 0; j < n; j += 1) c[target + j] = acc[j]!
	}
}

const nt: Kernel = (a, ao, b, bo, c, co, m, k, n) => {
	for (let i = 0; i < m; i += 1) {
		const left = ao + i * k
		for (let j = 0; j < n; j += 1) {
			const right = bo + j * k
			let total = 0
			for (let p = 0; p < k; p += 1) total += a[left + p]! * b[right + p]!
			c[co + i * n + j] = total
		}
	}
}

const tn: Kernel = (a, ao, b, bo, c, co, m, k, n, acc) => {
	for (let i = 0; i < m; i += 1) {
		acc.fill(0)
		for (let p = 0; p < k; p += 1) {
			const value = a[ao + p * m + i]!
			const row = bo + p * n
			for (let j = 0; j < n; j += 1) acc[j] = acc[j]! + value * b[row + j]!
		}
		const target = co + i * n
		for (let j = 0; j < n; j += 1) c[target + j] = acc[j]!
	}
}

const tt: Kernel = (a, ao, b, bo, c, co, m, k, n) => {
	for (let i = 0; i < m; i += 1) {
		for (let j = 0; j < n; j += 1) {
			let total = 0
			for (let p = 0; p < k; p += 1) total += a[ao + p * m + i]! * b[bo + j * k + p]!
			c[co + i * n + j] = total
		}
	}
}

/**
 * Matrix product over the last two axes with batch broadcasting.
 *
 * `a` is `[...batchA, m, k]` and `b` is `[...batchB, k, n]`; the result is `[...batch, m, n]`
 * where `batch` is the broadcast of `batchA` and `batchB`. A rank-2 `b` is therefore applied
 * to every matrix of a batched `a`, which is what a linear layer needs.
 * Sums are accumulated in 64-bit floats and rounded once per output element.
 */
export function matmul(a: Tensor, b: Tensor, options: MatmulOptions = {}): Tensor {
	if (a.rank < 2 || b.rank < 2) throw new RangeError('matmul needs tensors of rank 2 or more; reshape vectors first')
	const transposeA = options.transposeA === true
	const transposeB = options.transposeB === true
	const aRows = a.shape[a.rank - 2]!
	const aCols = a.shape[a.rank - 1]!
	const bRows = b.shape[b.rank - 2]!
	const bCols = b.shape[b.rank - 1]!
	const m = transposeA ? aCols : aRows
	const k = transposeA ? aRows : aCols
	const innerB = transposeB ? bCols : bRows
	const n = transposeB ? bRows : bCols
	if (k !== innerB) throw new RangeError(`Cannot multiply [${a.shape.join(', ')}] by [${b.shape.join(', ')}]: inner dimensions ${k} and ${innerB} differ`)
	const batchA = a.shape.slice(0, -2)
	const batchB = b.shape.slice(0, -2)
	const batch = broadcastShapes(batchA, batchB)
	const count = sizeOf(batch)
	const offsetsA = batchOffsets(batch, batchA, aRows * aCols)
	const offsetsB = batchOffsets(batch, batchB, bRows * bCols)
	const kernel: Kernel = transposeA ? (transposeB ? tt : tn) : transposeB ? nt : nn
	const out = new Float32Array(count * m * n)
	const acc = new Float64Array(n)
	for (let index = 0; index < count; index += 1) kernel(a.data, offsetsA[index]!, b.data, offsetsB[index]!, out, index * m * n, m, k, n, acc)
	return new Tensor([...batch, m, n], out)
}
