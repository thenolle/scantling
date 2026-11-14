import { expect, test } from 'bun:test'
import { allClose } from '../src/compare.ts'
import { transpose } from '../src/layout.ts'
import { matmul } from '../src/matmul.ts'
import { Random, randomNormal } from '../src/random.ts'
import { Tensor } from '../src/tensor.ts'

function reference(a: Tensor, b: Tensor): Tensor {
	const m = a.shape[a.rank - 2]!
	const k = a.shape[a.rank - 1]!
	const n = b.shape[b.rank - 1]!
	const batchA = a.size / (m * k)
	const batchB = b.size / (k * n)
	const count = Math.max(batchA, batchB)
	const out = new Float32Array(count * m * n)
	for (let q = 0; q < count; q += 1) {
		const ao = (batchA === 1 ? 0 : q) * m * k
		const bo = (batchB === 1 ? 0 : q) * k * n
		for (let i = 0; i < m; i += 1) {
			for (let j = 0; j < n; j += 1) {
				let total = 0
				for (let p = 0; p < k; p += 1) total += a.data[ao + i * k + p]! * b.data[bo + p * n + j]!
				out[q * m * n + i * n + j] = total
			}
		}
	}
	const batch = a.rank >= b.rank ? a.shape.slice(0, -2) : b.shape.slice(0, -2)
	return new Tensor([...batch, m, n], out)
}

test('multiplies a known pair of matrices', () => {
	const a = Tensor.fromNested([[1, 2, 3], [4, 5, 6]])
	const b = Tensor.fromNested([[7, 8], [9, 10], [11, 12]])
	expect(matmul(a, b).toArray()).toEqual([[58, 64], [139, 154]])
})

test('matches a double-precision reference on random shapes', () => {
	const random = new Random(7)
	const cases: [number[], number[]][] = [
		[[1, 1], [1, 1]],
		[[3, 5], [5, 4]],
		[[17, 9], [9, 13]],
		[[2, 4, 3], [2, 3, 5]],
		[[2, 3, 4, 6], [2, 3, 6, 2]],
		[[2, 4, 3], [3, 5]]
	]
	for (const [shapeA, shapeB] of cases) {
		const a = randomNormal(shapeA, random)
		const b = randomNormal(shapeB, random)
		expect(allClose(matmul(a, b), reference(a, b), { rtol: 1e-5, atol: 1e-5 })).toBe(true)
	}
})

test('transpose flags equal an explicit transpose', () => {
	const random = new Random(11)
	for (const batch of [[], [2]]) {
		const a = randomNormal([...batch, 4, 3], random)
		const b = randomNormal([...batch, 3, 5], random)
		const expected = matmul(a, b)
		expect(allClose(matmul(a, transpose(b), { transposeB: true }), expected, { atol: 1e-5 })).toBe(true)
		expect(allClose(matmul(transpose(a), b, { transposeA: true }), expected, { atol: 1e-5 })).toBe(true)
		expect(allClose(matmul(transpose(a), transpose(b), { transposeA: true, transposeB: true }), expected, { atol: 1e-5 })).toBe(true)
	}
})

test('rejects mismatched inner dimensions and vectors', () => {
	expect(() => matmul(Tensor.zeros([2, 3]), Tensor.zeros([4, 2]))).toThrow(RangeError)
	expect(() => matmul(Tensor.zeros([3]), Tensor.zeros([3, 2]))).toThrow(RangeError)
})
