import { expect, test } from 'bun:test'
import { matmul } from '../src/matmul.ts'
import { Random, randomNormal } from '../src/random.ts'
import { allClose } from '../src/compare.ts'
import { gemmSimd, gemmValid, setSimdEnabled, simdSupported } from '../src/simd.ts'

const random = new Random(41)
const fill = (length: number): Float32Array => Float32Array.from({ length }, () => random.normal())
const simd = simdSupported ? test : test.skip

test('a runtime that supports SIMD accepts the matrix-multiply module', () => {
	expect(gemmValid, 'SIMD is supported but the gemm module failed validation: the encoder has a bug').toBe(simdSupported)
})

simd('the SIMD kernel matches a reference for awkward sizes and every transpose mode', () => {
	const sizes = [1, 3, 4, 5, 8, 9, 17]
	for (const transposeA of [false, true]) {
		for (const transposeB of [false, true]) {
			for (const m of sizes) {
				for (const k of sizes) {
					for (const n of sizes) {
						const a = fill(m * k)
						const b = fill(k * n)
						const c = new Float32Array(m * n)
						gemmSimd(a, 0, transposeA, b, 0, transposeB, c, 0, m, k, n)
						for (let i = 0; i < m; i += 1) {
							for (let j = 0; j < n; j += 1) {
								let total = 0
								let magnitude = 0
								for (let p = 0; p < k; p += 1) {
									const left = transposeA ? a[p * m + i]! : a[i * k + p]!
									const right = transposeB ? b[j * k + p]! : b[p * n + j]!
									total += left * right
									magnitude += Math.abs(left * right)
								}
								expect(Math.abs(c[i * n + j]! - total)).toBeLessThanOrEqual(1e-6 * (1 + magnitude * k))
							}
						}
					}
				}
			}
		}
	}
})

simd('a row of the result does not depend on the other rows', () => {
	const k = 13
	const n = 11
	const a = fill(9 * k)
	const b = fill(k * n)
	const whole = new Float32Array(9 * n)
	gemmSimd(a, 0, false, b, 0, false, whole, 0, 9, k, n)
	const few = new Float32Array(5 * n)
	gemmSimd(a, 0, false, b, 0, false, few, 0, 5, k, n)
	expect(Array.from(few)).toEqual(Array.from(whole.subarray(0, 5 * n)))
})

simd('matmul agrees with the scalar path on large batched products', () => {
	const a = randomNormal([3, 24, 40], random)
	const b = randomNormal([40, 36], random)
	setSimdEnabled(true)
	const fast = matmul(a, b)
	setSimdEnabled(false)
	const scalar = matmul(a, b)
	setSimdEnabled(true)
	expect(allClose(fast, scalar, { rtol: 1e-5, atol: 1e-5 })).toBe(true)
})
