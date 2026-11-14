import { afterAll, expect, test } from 'bun:test'
import { allClose, matmul, Random, randomNormal } from '@scantling/tensor'
import type { MatmulOptions } from '@scantling/tensor'
import { ParallelMatmul } from '../src/index.ts'

const parallel = new ParallelMatmul({ workers: 2, minimumFlops: 0 })
const random = new Random(5)

afterAll(async () => {
	await parallel.close()
})

const cases: [number[], number[], MatmulOptions][] = [
	[[1, 1], [1, 1], {}],
	[[7, 5], [5, 9], {}],
	[[33, 17], [17, 21], {}],
	[[3, 6, 4], [4, 5], {}],
	[[2, 3, 6, 4], [2, 3, 4, 5], {}],
	[[6, 4], [5, 4], { transposeB: true }],
	[[4, 6], [4, 5], { transposeA: true }],
	[[2, 4, 6], [2, 5, 4], { transposeA: true, transposeB: true }]
]

test('matches the single-threaded result exactly', async () => {
	for (const [shapeA, shapeB, options] of cases) {
		const a = randomNormal(shapeA, random)
		const b = randomNormal(shapeB, random)
		const expected = matmul(a, b, options)
		const actual = await parallel.matmul(a, b, options)
		expect(actual.shape).toEqual(expected.shape)
		expect(allClose(actual, expected, { rtol: 0, atol: 0 })).toBe(true)
	}
})

test('small products stay on the calling thread', async () => {
	const local = new ParallelMatmul({ workers: 1, minimumFlops: 1e12 })
	const a = randomNormal([4, 4], random)
	const result = await local.matmul(a, a)
	expect(allClose(result, matmul(a, a), { rtol: 0, atol: 0 })).toBe(true)
	await local.close()
})

test('concurrent calls are serialised and stay correct', async () => {
	const inputs = Array.from({ length: 4 }, () => [randomNormal([16, 8], random), randomNormal([8, 12], random)] as const)
	const results = await Promise.all(inputs.map(([a, b]) => parallel.matmul(a, b)))
	results.forEach((result, index) => {
		const [a, b] = inputs[index]!
		expect(allClose(result, matmul(a, b), { rtol: 0, atol: 0 })).toBe(true)
	})
})

test('rejects mismatched shapes', async () => {
	const a = randomNormal([2, 3], random)
	return expect(parallel.matmul(a, randomNormal([4, 2], random))).rejects.toThrow(RangeError)
})
