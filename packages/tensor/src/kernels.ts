export type KernelKind = 'nn' | 'nt' | 'tn' | 'tt'

/**
 * Computes rows `[i0, i1)` of `C = op(A) * op(B)` for one matrix pair. Sums run in 64-bit floats
 * and are rounded once per output element. `acc` is scratch space of at least `n` doubles.
 */
export type Kernel = (a: Float32Array, ao: number, b: Float32Array, bo: number, c: Float32Array, co: number, m: number, k: number, n: number, acc: Float64Array, i0: number, i1: number) => void

const nn: Kernel = (a, ao, b, bo, c, co, _m, k, n, acc, i0, i1) => {
	for (let i = i0; i < i1; i += 1) {
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

const nt: Kernel = (a, ao, b, bo, c, co, _m, k, n, _acc, i0, i1) => {
	for (let i = i0; i < i1; i += 1) {
		const left = ao + i * k
		for (let j = 0; j < n; j += 1) {
			const right = bo + j * k
			let total = 0
			for (let p = 0; p < k; p += 1) total += a[left + p]! * b[right + p]!
			c[co + i * n + j] = total
		}
	}
}

const tn: Kernel = (a, ao, b, bo, c, co, m, k, n, acc, i0, i1) => {
	for (let i = i0; i < i1; i += 1) {
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
