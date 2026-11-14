import type { QuantizedMatrix } from './quantize.ts'

/** A matrix-vector product `out = W x` where `W` is stored `[rows, cols]` with one row per output. */
export interface Projection {
	readonly rows: number
	readonly cols: number
	apply(x: Float32Array, out: Float32Array): void
}

/**
 * Dense matrix-vector product. Four outputs are computed together with four independent accumulators, which
 * hides the latency of the floating-point add and loads each element of `x` once for all four.
 */
export function gemvDense(w: Float32Array, rows: number, cols: number, x: Float32Array, out: Float32Array): void {
	let r = 0
	for (; r + 4 <= rows; r += 4) {
		const o0 = r * cols
		const o1 = o0 + cols
		const o2 = o1 + cols
		const o3 = o2 + cols
		let s0 = 0
		let s1 = 0
		let s2 = 0
		let s3 = 0
		for (let c = 0; c < cols; c += 1) {
			const v = x[c]!
			s0 += v * w[o0 + c]!
			s1 += v * w[o1 + c]!
			s2 += v * w[o2 + c]!
			s3 += v * w[o3 + c]!
		}
		out[r] = s0
		out[r + 1] = s1
		out[r + 2] = s2
		out[r + 3] = s3
	}
	for (; r < rows; r += 1) {
		const o = r * cols
		let s = 0
		for (let c = 0; c < cols; c += 1) s += x[c]! * w[o + c]!
		out[r] = s
	}
}

/** Same product with int8 weights. Each output is `scale[row] * sum(x * q)`, so the scale is applied once per row. */
export function gemvInt8(matrix: QuantizedMatrix, x: Float32Array, out: Float32Array): void {
	const { rows, cols, data: w, scales } = matrix
	let r = 0
	for (; r + 4 <= rows; r += 4) {
		const o0 = r * cols
		const o1 = o0 + cols
		const o2 = o1 + cols
		const o3 = o2 + cols
		let s0 = 0
		let s1 = 0
		let s2 = 0
		let s3 = 0
		for (let c = 0; c < cols; c += 1) {
			const v = x[c]!
			s0 += v * w[o0 + c]!
			s1 += v * w[o1 + c]!
			s2 += v * w[o2 + c]!
			s3 += v * w[o3 + c]!
		}
		out[r] = s0 * scales[r]!
		out[r + 1] = s1 * scales[r + 1]!
		out[r + 2] = s2 * scales[r + 2]!
		out[r + 3] = s3 * scales[r + 3]!
	}
	for (; r < rows; r += 1) {
		const o = r * cols
		let s = 0
		for (let c = 0; c < cols; c += 1) s += x[c]! * w[o + c]!
		out[r] = s * scales[r]!
	}
}

export class DenseProjection implements Projection {
	constructor(readonly weights: Float32Array, readonly rows: number, readonly cols: number) { }

	apply(x: Float32Array, out: Float32Array): void {
		gemvDense(this.weights, this.rows, this.cols, x, out)
	}
}

export class Int8Projection implements Projection {
	readonly rows: number
	readonly cols: number

	constructor(readonly matrix: QuantizedMatrix) {
		this.rows = matrix.rows
		this.cols = matrix.cols
	}

	apply(x: Float32Array, out: Float32Array): void {
		gemvInt8(this.matrix, x, out)
	}
}
