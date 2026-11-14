import { expect, test } from 'bun:test'
import { add, addInPlace, axpyInPlace, gelu, mul, relu, scaleInPlace, sigmoid, silu, sub } from '../src/elementwise.ts'
import { Tensor } from '../src/tensor.ts'

const t = (nested: Parameters<typeof Tensor.fromNested>[0]): Tensor => Tensor.fromNested(nested)

test('adds a row vector to a matrix', () => {
	const result = add(t([[1, 2, 3], [4, 5, 6]]), t([10, 20, 30]))
	expect(result.toArray()).toEqual([[11, 22, 33], [14, 25, 36]])
})

test('broadcasts a column against a row', () => {
	const result = add(t([[1], [2]]), t([[10, 20, 30]]))
	expect(result.toArray()).toEqual([[11, 21, 31], [12, 22, 32]])
})

test('broadcasts across three axes', () => {
	const result = add(t([[[1, 2]], [[3, 4]]]), t([[10], [20], [30]]))
	expect(result.shape).toEqual([2, 3, 2])
	expect(result.toArray()).toEqual([
		[[11, 12], [21, 22], [31, 32]],
		[[13, 14], [23, 24], [33, 34]]
	])
})

test('combines with scalars and rejects incompatible shapes', () => {
	expect(mul(t([1, 2, 3]), Tensor.scalar(2)).toArray()).toEqual([2, 4, 6])
	expect(sub(Tensor.scalar(10), t([1, 2])).toArray()).toEqual([9, 8])
	expect(() => add(t([1, 2, 3]), t([1, 2]))).toThrow(RangeError)
})

test('activation functions match known values', () => {
	expect(relu(t([-1, 0, 2])).toArray()).toEqual([0, 0, 2])
	expect(sigmoid(t([0, 1000, -1000])).toArray()).toEqual([0.5, 1, 0])
	expect(silu(t([0])).item()).toBe(0)
	expect(gelu(t([0])).item()).toBe(0)
	expect(gelu(t([1])).item()).toBeCloseTo(0.8412, 3)
})

test('in-place operations modify only the target', () => {
	const target = t([1, 2, 3])
	const other = t([1, 1, 1])
	addInPlace(target, other)
	axpyInPlace(target, 2, other)
	scaleInPlace(target, 10)
	expect(target.toArray()).toEqual([40, 50, 60])
	expect(other.toArray()).toEqual([1, 1, 1])
	expect(() => addInPlace(target, t([1, 2]))).toThrow(RangeError)
})
