import { expect, test } from 'bun:test'
import { Random, Tensor, randomUniform } from '@scantling/tensor'
import { backward, describeGradcheck, gradcheck, noGrad, ops, parameter, variable } from '../src/index.ts'
import type { Variable } from '../src/index.ts'

const random = new Random(77)
const input = (shape: number[], low = -1, high = 1): Tensor => randomUniform(shape, random, low, high)
const awayFromZero = (shape: number[]): Tensor => {
	const tensor = Tensor.zeros(shape)
	for (let index = 0; index < tensor.size; index += 1) tensor.data[index] = (random.next() < 0.5 ? -1 : 1) * (0.2 + 0.8 * random.next())
	return tensor
}

function check(fn: (...inputs: Variable[]) => Variable, inputs: Tensor[]): void {
	const result = gradcheck(fn, inputs)
	expect(result.checked).toBeGreaterThan(0)
	expect(result.ok, describeGradcheck(result)).toBe(true)
}

test('activation gradients', () => {
	check((a) => ops.relu(a), [awayFromZero([3, 4])])
	check((a) => ops.sigmoid(a), [input([3, 4], -2, 2)])
	check((a) => ops.tanh(a), [input([3, 4], -2, 2)])
	check((a) => ops.silu(a), [input([3, 4], -2, 2)])
	check((a) => ops.gelu(a), [input([3, 4], -2, 2)])
})

test('softmax and log-softmax gradients along any axis', () => {
	check((a) => ops.softmax(a), [input([3, 4])])
	check((a) => ops.softmax(a, 0), [input([3, 4])])
	check((a) => ops.logSoftmax(a), [input([2, 3, 4])])
	check((a) => ops.logSoftmax(a, 1), [input([2, 3, 4])])
})

test('cross-entropy matches the closed form and its gradient', () => {
	const logits = parameter(Tensor.zeros([2, 4]))
	const loss = ops.crossEntropy(logits, [1, 3])
	expect(loss.value.item()).toBeCloseTo(Math.log(4), 5)
	backward(loss)
	expect(logits.grad?.at(0, 0)).toBeCloseTo(0.125, 6)
	expect(logits.grad?.at(0, 1)).toBeCloseTo(-0.375, 6)
	check((a) => ops.crossEntropy(a, [1, 0, 4]), [input([3, 5], -2, 2)])
	check((a) => ops.crossEntropy(a, [2, 1, 0, 3], {}), [input([2, 2, 4])])
})

test('cross-entropy ignores padding and rejects bad targets', () => {
	const logits = parameter(input([3, 4]))
	backward(ops.crossEntropy(logits, [1, -1, 2], { ignoreIndex: -1 }))
	expect(Array.from(logits.grad!.data.subarray(4, 8))).toEqual([0, 0, 0, 0])
	expect(() => ops.crossEntropy(logits, [1, 2, 9])).toThrow(RangeError)
	expect(() => ops.crossEntropy(logits, [1, 2])).toThrow(RangeError)
	expect(ops.crossEntropy(logits, [-1, -1, -1], { ignoreIndex: -1 }).value.item()).toBe(0)
})

test('embedding, slice and concat gradients', () => {
	check((table) => ops.reshape(ops.embedding(table, [2, 0, 2, 4]), [2, 2, 3]), [input([5, 3])])
	check((a) => ops.sliceAxis(a, 1, 1, 4), [input([2, 5, 3])])
	check((a) => ops.sliceAxis(a, -1, 0, 2), [input([2, 3])])
	check((a, b) => ops.concat([a, b], 1), [input([2, 3]), input([2, 2])])
	check((a, b, c) => ops.concat([a, b, c], 0), [input([1, 3]), input([2, 3]), input([1, 3])])
})

test('maskedFill passes no gradient through masked positions', () => {
	const mask = Tensor.from([3, 3], [0, 1, 1, 0, 0, 1, 0, 0, 0])
	check((a) => ops.maskedFill(a, mask, -5), [input([2, 3, 3])])
	const a = parameter(input([3, 3]))
	backward(ops.sum(ops.maskedFill(a, mask, -5)))
	expect(a.grad?.toArray()).toEqual([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
	expect(() => ops.maskedFill(variable(input([3])), mask, 0)).toThrow(RangeError)
})

test('dropout scales the kept elements and is the identity when disabled', () => {
	const a = parameter(Tensor.ones([2000]))
	const out = ops.dropout(a, 0.5, new Random(5))
	const kept = out.value.data.filter((value) => value !== 0).length
	expect(kept).toBeGreaterThan(900)
	expect(kept).toBeLessThan(1100)
	expect(out.value.data.reduce((total, value) => total + value, 0) / 2000).toBeCloseTo(1, 0)
	expect(ops.dropout(a, 0.5, new Random(5), false)).toBe(a)
	expect(() => ops.dropout(a, 1, new Random(1))).toThrow(RangeError)
})

test('rmsNorm matches the definition and its gradients', () => {
	const out = ops.rmsNorm(variable(Tensor.from([1, 2], [3, 4])), variable(Tensor.ones([2])), 0)
	expect(out.value.at(0, 0)).toBeCloseTo(0.848528, 5)
	expect(out.value.at(0, 1)).toBeCloseTo(1.131371, 5)
	check((x, w) => ops.rmsNorm(x, w), [input([3, 4]), input([4], 0.5, 1.5)])
	check((x, w) => ops.rmsNorm(x, w, 1e-3), [input([2, 3, 4]), input([4], 0.5, 1.5)])
})

test('rotary embeddings preserve norms, leave position zero alone and have correct gradients', () => {
	const tables = ops.ropeTables(3, 4)
	const x = variable(input([2, 3, 4]))
	const y = noGrad(() => ops.rope(x, tables))
	for (let row = 0; row < 6; row += 1) {
		const before = Array.from(x.value.data.subarray(row * 4, row * 4 + 4))
		const after = Array.from(y.value.data.subarray(row * 4, row * 4 + 4))
		expect(Math.hypot(...after)).toBeCloseTo(Math.hypot(...before), 5)
		if (row % 3 === 0) after.forEach((value, index) => expect(value).toBeCloseTo(before[index]!, 6))
	}
	check((a) => ops.rope(a, tables), [input([2, 3, 4])])
	expect(() => ops.rope(variable(input([3, 5])), tables)).toThrow(RangeError)
})

test('a causal attention pipeline has correct gradients', () => {
	const causal = Tensor.from([3, 3], [0, 1, 1, 0, 0, 1, 0, 0, 0])
	check((q, k, v) => {
		const scores = ops.mulScalar(ops.matmul(q, k, { transposeB: true }), 1 / Math.sqrt(4))
		return ops.matmul(ops.softmax(ops.maskedFill(scores, causal, -1e9)), v)
	}, [input([2, 3, 4]), input([2, 3, 4]), input([2, 3, 4])])
})
