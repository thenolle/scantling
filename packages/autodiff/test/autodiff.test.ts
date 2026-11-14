import { expect, test } from 'bun:test'
import { Tensor } from '@scantling/tensor'
import { backward, clipGradNorm, globalGradNorm, isGradEnabled, noGrad, ops, parameter, record, variable, zeroGrad } from '../src/index.ts'

const t = (nested: Parameters<typeof Tensor.fromNested>[0]): Tensor => Tensor.fromNested(nested)

test('differentiates a sum of squares', () => {
	const x = parameter(t([1, 2, 3]))
	backward(ops.sum(ops.mul(x, x)))
	expect(x.grad?.toArray()).toEqual([2, 4, 6])
})

test('differentiates z = a * b + a with respect to both inputs', () => {
	const a = parameter(Tensor.scalar(2))
	const b = parameter(Tensor.scalar(3))
	backward(ops.add(ops.mul(a, b), a))
	expect(a.grad?.item()).toBe(4)
	expect(b.grad?.item()).toBe(2)
})

test('sums gradients over a node used twice', () => {
	const a = parameter(Tensor.scalar(5))
	const doubled = ops.add(a, a)
	backward(ops.mul(doubled, doubled))
	expect(a.grad?.item()).toBe(40)
})

test('sums bias gradients over the broadcast batch', () => {
	const x = parameter(Tensor.ones([2, 3]))
	const bias = parameter(t([1, 2, 3]))
	backward(ops.sum(ops.add(x, bias)))
	expect(bias.grad?.toArray()).toEqual([2, 2, 2])
	expect(x.grad?.toArray()).toEqual([[1, 1, 1], [1, 1, 1]])
})

test('accumulates across backward calls until zeroGrad', () => {
	const x = parameter(t([1, 2]))
	backward(ops.sum(ops.mulScalar(x, 3)))
	backward(ops.sum(ops.mulScalar(x, 3)))
	expect(x.grad?.toArray()).toEqual([6, 6])
	zeroGrad([x])
	expect(x.grad).toBeNull()
})

test('does not alias a gradient shared by two parents', () => {
	const a = parameter(t([1, 2]))
	const b = parameter(t([3, 4]))
	backward(ops.sum(ops.add(a, b)))
	a.grad?.fill(9)
	expect(b.grad?.toArray()).toEqual([1, 1])
})

test('noGrad records no graph', () => {
	const x = parameter(t([1, 2]))
	const result = noGrad(() => ops.mulScalar(x, 2))
	expect(result.requiresGrad).toBe(false)
	expect(isGradEnabled()).toBe(true)
	expect(() => backward(result)).toThrow()
})

test('validates the root and the seed', () => {
	const x = parameter(t([1, 2]))
	expect(() => backward(ops.mulScalar(x, 2))).toThrow(RangeError)
	expect(() => backward(ops.mulScalar(x, 2), Tensor.ones([3]))).toThrow(RangeError)
	expect(() => backward(variable(t([1])))).toThrow()
})

test('reports a wrongly shaped gradient', () => {
	const x = parameter(t([1, 2]))
	const broken = record(x.value, [x], () => [Tensor.zeros([3])], 'broken')
	expect(() => backward(ops.sum(broken))).toThrow(/broken/)
})

test('clips the global gradient norm', () => {
	const a = parameter(t([0, 0]))
	const b = parameter(Tensor.scalar(0))
	a.grad = t([3, 0])
	b.grad = Tensor.scalar(4)
	expect(globalGradNorm([a, b])).toBe(5)
	expect(clipGradNorm([a, b], 10)).toBe(5)
	expect(a.grad.toArray()).toEqual([3, 0])
	expect(clipGradNorm([a, b], 1)).toBe(5)
	expect(a.grad.data[0]).toBeCloseTo(0.6, 6)
	expect(b.grad.item()).toBeCloseTo(0.8, 6)
})

test('leaves gradients untouched when the norm is not finite', () => {
	const a = parameter(t([1]))
	a.grad = t([Number.NaN])
	expect(Number.isNaN(clipGradNorm([a], 1))).toBe(true)
	expect(Number.isNaN(a.grad.data[0])).toBe(true)
})
