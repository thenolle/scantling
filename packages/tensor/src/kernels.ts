export type KernelKind = 'nn' | 'nt' | 'tn' | 'tt'

/**
 * Computes rows `[i0, i1)` of `C = op(A) * op(B)` for one matrix pair. Sums run in 64-bit floats and are
 * rounded once per output element. `acc` is scratch space of at least `2 * n` doubles, see `accumulatorFor`.
 */
export type Kernel = (a: Float32Array, ao: number, b: Float32Array, bo: number, c: Float32Array, co: number, m: number, k: number, n: number, acc: Float64Array, i0: number, i1: number) => void

let scratch = new Float64Array(0)

/** A reusable accumulator for the kernels. Safe to share because kernels are synchronous and each thread has its own copy of this module. */
export function accumulatorFor(n: number): Float64Array {
	if (scratch.length < 2 * n) scratch = new Float64Array(2 * n)
	return scratch
}

/**
 * Rows of A (addressed as `a[ao + i*rowStride + p*colStride]`) times a row-major `[k, n]` matrix B.
 * Handles `nn` (rowStride k, colStride 1) and `tn` (rowStride 1, colStride m). Two rows are processed together
 * and `k` is unrolled by four, which cuts loads and stores of the accumulator by a factor of four.
 */
function rowsTimesMatrix(a: Float32Array, ao: number, rowStride: number, colStride: number, b: Float32Array, bo: number, c: Float32Array, co: number, k: number, n: number, acc: Float64Array, i0: number, i1: number): void {
	let i = i0
	for (; i + 2 <= i1; i += 2) {
		acc.fill(0, 0, 2 * n)
		const r0 = ao + i * rowStride
		const r1 = r0 + rowStride
		let p = 0
		for (; p + 4 <= k; p += 4) {
			const a00 = a[r0 + p * colStride]!
			const a01 = a[r0 + (p + 1) * colStride]!
			const a02 = a[r0 + (p + 2) * colStride]!
			const a03 = a[r0 + (p + 3) * colStride]!
			const a10 = a[r1 + p * colStride]!
			const a11 = a[r1 + (p + 1) * colStride]!
			const a12 = a[r1 + (p + 2) * colStride]!
			const a13 = a[r1 + (p + 3) * colStride]!
			const b0 = bo + p * n
			const b1 = b0 + n
			const b2 = b1 + n
			const b3 = b2 + n
			for (let j = 0; j < n; j += 1) {
				const x0 = b[b0 + j]!
				const x1 = b[b1 + j]!
				const x2 = b[b2 + j]!
				const x3 = b[b3 + j]!
				acc[j] = acc[j]! + (a00 * x0 + a01 * x1 + a02 * x2 + a03 * x3)
				acc[n + j] = acc[n + j]! + (a10 * x0 + a11 * x1 + a12 * x2 + a13 * x3)
			}
		}
		for (; p < k; p += 1) {
			const a0 = a[r0 + p * colStride]!
			const a1 = a[r1 + p * colStride]!
			const row = bo + p * n
			for (let j = 0; j < n; j += 1) {
				const x = b[row + j]!
				acc[j] = acc[j]! + a0 * x
				acc[n + j] = acc[n + j]! + a1 * x
			}
		}
		const o0 = co + i * n
		for (let j = 0; j < n; j += 1) {
			c[o0 + j] = acc[j]!
			c[o0 + n + j] = acc[n + j]!
		}
	}
	if (i < i1) {
		acc.fill(0, 0, n)
		const r0 = ao + i * rowStride
		let p = 0
		for (; p + 4 <= k; p += 4) {
			const a00 = a[r0 + p * colStride]!
			const a01 = a[r0 + (p + 1) * colStride]!
			const a02 = a[r0 + (p + 2) * colStride]!
			const a03 = a[r0 + (p + 3) * colStride]!
			const b0 = bo + p * n
			const b1 = b0 + n
			const b2 = b1 + n
			const b3 = b2 + n
			for (let j = 0; j < n; j += 1) acc[j] = acc[j]! + (a00 * b[b0 + j]! + a01 * b[b1 + j]! + a02 * b[b2 + j]! + a03 * b[b3 + j]!)
		}
		for (; p < k; p += 1) {
			const a0 = a[r0 + p * colStride]!
			const row = bo + p * n
			for (let j = 0; j < n; j += 1) acc[j] = acc[j]! + a0 * b[row + j]!
		}
		const o0 = co + i * n
		for (let j = 0; j < n; j += 1) c[o0 + j] = acc[j]!
	}
}

const nn: Kernel = (a, ao, b, bo, c, co, _m, k, n, acc, i0, i1) => rowsTimesMatrix(a, ao, k, 1, b, bo, c, co, k, n, acc, i0, i1)

const tn: Kernel = (a, ao, b, bo, c, co, m, k, n, acc, i0, i1) => rowsTimesMatrix(a, ao, 1, m, b, bo, c, co, k, n, acc, i0, i1)

/**
 * `B` is stored `[n, k]`, so every output is a dot product of two contiguous rows. Four outputs are
 * computed together with four independent accumulators, which hides the latency of the floating-point
 * add and loads each element of the A row once for all four.
 */
const nt: Kernel = (a, ao, b, bo, c, co, _m, k, n, _acc, i0, i1) => {
	for (let i = i0; i < i1; i += 1) {
		const left = ao + i * k
		const out = co + i * n
		let j = 0
		for (; j + 4 <= n; j += 4) {
			const r0 = bo + j * k
			const r1 = r0 + k
			const r2 = r1 + k
			const r3 = r2 + k
			let s0 = 0
			let s1 = 0
			let s2 = 0
			let s3 = 0
			for (let p = 0; p < k; p += 1) {
				const x = a[left + p]!
				s0 += x * b[r0 + p]!
				s1 += x * b[r1 + p]!
				s2 += x * b[r2 + p]!
				s3 += x * b[r3 + p]!
			}
			c[out + j] = s0
			c[out + j + 1] = s1
			c[out + j + 2] = s2
			c[out + j + 3] = s3
		}
		for (; j < n; j += 1) {
			const row = bo + j * k
			let t0 = 0
			let t1 = 0
			let t2 = 0
			let t3 = 0
			let p = 0
			for (; p + 4 <= k; p += 4) {
				t0 += a[left + p]! * b[row + p]!
				t1 += a[left + p + 1]! * b[row + p + 1]!
				t2 += a[left + p + 2]! * b[row + p + 2]!
				t3 += a[left + p + 3]! * b[row + p + 3]!
			}
			let tail = 0
			for (; p < k; p += 1) tail += a[left + p]! * b[row + p]!
			c[out + j] = t0 + t1 + (t2 + t3) + tail
		}
	}
}

const tt: Kernel = (a, ao, b, bo, c, co, m, k, n, _acc, i0, i1) => {
	for (let i = i0; i < i1; i += 1) {
		for (let j = 0; j < n; j += 1) {
			let total = 0
			for (let p = 0; p < k; p += 1) total += a[ao + p * m + i]! * b[bo + j * k + p]!
			c[co + i * n + j] = total
		}
	}
}

export const KERNELS: Readonly<Record<KernelKind, Kernel>> = { nn, nt, tn, tt }

export function kernelKind(transposeA: boolean, transposeB: boolean): KernelKind {
	if (transposeA) return transposeB ? 'tt' : 'tn'
	return transposeB ? 'nt' : 'nn'
}
