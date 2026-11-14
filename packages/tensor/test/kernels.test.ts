import { expect, test } from 'bun:test'
import { KERNELS, accumulatorFor } from '../src/kernels.ts'
import type { KernelKind } from '../src/kernels.ts'
import { Random } from '../src/random.ts'

const random = new Random(31)
const fill = (length: number): Float32Array => Float32Array.from({ length }, () => random.normal())

function naive(kind: KernelKind, a: Float32Array, b: Float32Array, m: number, k: number, n: number): Float64Array {
	const out = new Float64Array(m * n)
	for (let i = 0; i < m; i += 1) {
		for (let j = 0; j < n; j += 1) {
			let total = 0
			for (let p = 0; p < k; p += 1) {
				const left = kind === 'nn' || kind === 'nt' ? a[i * k + p]! : a[p * m + i]!
				const right = kind === 'nn' || kind === 'tn' ? b[p * n + j]! : b[j * k + p]!
				total += left * right
			}
			out[i * n + j] = total
		}
	}
	return out
}

test('every kernel matches a double-precision reference for awkward sizes', () => {
	const sizes = [1, 2, 3, 4, 5, 7, 8, 9]
	for (const kind of ['nn', 'nt', 'tn', 'tt'] as const) {
		for (const m of sizes) {
			for (const k of sizes) {
				for (const n of sizes) {
					const a = fill(m * k)
					const b = fill(k * n)
					const c = new Float32Array(m * n)
					KERNELS[kind](a, 0, b, 0, c, 0, m, k, n, accumulatorFor(n), 0, m)
					const expected = naive(kind, a, b, m, k, n)
					for (let index = 0; index < c.length; index += 1) expect(Math.abs(c[index]! - expected[index]!)).toBeLessThan(1e-5 * (1 + Math.abs(expected[index]!)))
				}
			}
		}
	}
})

test('splitting the rows at any boundary does not change a single bit', () => {
	const m = 9
	const k = 11
	const n = 7
	for (const kind of ['nn', 'nt', 'tn'] as const) {
		const a = fill(m * k)
		const b = fill(k * n)
		const whole = new Float32Array(m * n)
		KERNELS[kind](a, 0, b, 0, whole, 0, m, k, n, accumulatorFor(n), 0, m)
		for (let split = 1; split < m; split += 1) {
			const parts = new Float32Array(m * n)
			KERNELS[kind](a, 0, b, 0, parts, 0, m, k, n, accumulatorFor(n), 0, split)
			KERNELS[kind](a, 0, b, 0, parts, 0, m, k, n, accumulatorFor(n), split, m)
			expect(parts).toEqual(whole)
		}
	}
})
