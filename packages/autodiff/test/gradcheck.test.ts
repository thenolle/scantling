import { expect, test } from 'bun:test'
import { Random, randomUniform } from '@scantling/tensor'
import type { Tensor } from '@scantling/tensor'
import { describeGradcheck, gradcheck, ops, record } from '../src/index.ts'
import type { Variable } from '../src/index.ts'

const random = new Random(2024)
const input = (shape: number[], low = -1, high = 1): Tensor => randomUniform(shape, random, low, high)

function check(fn: (...inputs: Variable[]) => Variable, inputs: Tensor[]): void {
	const result = gradcheck(fn, inputs)
	expect(result.checked).toBeGreaterThan(0)
	expect(result.ok, describeGradcheck(result)).toBe(true)
}

test('add, sub and mul with broadcasting', () => {
	check((a, b) => ops.add(a, b), [input([2, 3]), input([3])])
	check((a, b) => ops.sub(a, b), [input([2, 3]), input([2, 1])])
	check((a, b) => ops.mul(a, b), [input([2, 1, 3]), input([4, 1])])
})

test('div keeps its denominator away from zero', () => {
	check((a, b) => ops.div(a, b), [input([2, 3]), input([3], 0.5, 2)])
})

test('unary operations', () => {
	check((a) => ops.neg(a), [input([2, 3])])
	check((a) => ops.exp(a), [input([2, 3])])
	check((a) => ops.log(a), [input([2, 3], 0.5, 2)])
	check((a) => ops.sqrt(a), [input([2, 3], 0.5, 2)])
	check((a) => ops.mulScalar(ops.addScalar(a, 1), 3), [input([4])])
})

test('matmul in every transpose mode with batch broadcasting', () => {
	check((a, b) => ops.matmul(a, b), [input([2, 3, 4]), input([4, 5])])
	check((a, b) => ops.matmul(a, b, { transposeB: true }), [input([2, 3, 4]), input([5, 4])])
	check((a, b) => ops.matmul(a, b, { transposeA: true }), [input([4, 3]), input([2, 4, 5])])
	check((a, b) => ops.matmul(a, b, { transposeA: true, transposeB: true }), [input([2, 4, 3]), input([2, 5, 4])])
})

test('reductions', () => {
	check((a) => ops.sum(a), [input([2, 3])])
	check((a) => ops.sum(a, { axes: 1 }), [input([2, 3, 2])])
	check((a) => ops.sum(a, { axes: [0, 2], keepDims: true }), [input([2, 3, 2])])
	check((a) => ops.mean(a), [input([3, 4])])
	check((a) => ops.mean(a, { axes: -1 }), [input([3, 4])])
})

test('layout operations', () => {
	check((a) => ops.reshape(a, [3, -1]), [input([2, 3])])
	check((a) => ops.permute(a, [2, 0, 1]), [input([2, 3, 4])])
	check((a) => ops.transpose(a), [input([3, 4])])
})

test('a composite expression', () => {
	check(
		(x, w, b) => ops.mean(ops.exp(ops.mulScalar(ops.add(ops.matmul(x, w), b), 0.5)), { axes: -1 }),
		[input([3, 4]), input([4, 2]), input([2])]
	)
})

test('detects a deliberately wrong gradient', () => {
	const wrong = (a: Variable): Variable => record(a.value, [a], (g) => [g.clone().fill(2)], 'wrong')
	const result = gradcheck(wrong, [input([4])])
	expect(result.ok).toBe(false)
	expect(result.worst).toBeDefined()
})
