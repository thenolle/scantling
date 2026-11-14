import { expect, test } from 'bun:test'
import { argmax, max, mean, min, sum, sumToShape } from '../src/reduce.ts'
import { Tensor } from '../src/tensor.ts'

const m = Tensor.fromNested([[1, 2, 3], [4, 5, 6]])

test('sums along axes', () => {
	expect(sum(m).item()).toBe(21)
	expect(sum(m, { axes: 0 }).toArray()).toEqual([5, 7, 9])
	expect(sum(m, { axes: 1 }).toArray()).toEqual([6, 15])
	expect(sum(m, { axes: -1, keepDims: true }).shape).toEqual([2, 1])
	expect(sum(m, { axes: [0, 1] }).item()).toBe(21)
})

test('computes mean, max and min', () => {
	expect(mean(m, { axes: 1 }).toArray()).toEqual([2, 5])
	expect(max(m, { axes: 0 }).toArray()).toEqual([4, 5, 6])
	expect(min(m).item()).toBe(1)
})

test('finds the index of the maximum', () => {
	expect(argmax(m, 1).toArray()).toEqual([2, 2])
	expect(argmax(Tensor.fromNested([3, 9, 9, 1])).item()).toBe(1)
})

test('propagates NaN through max and rejects empty axes', () => {
	expect(Number.isNaN(max(Tensor.fromNested([1, Number.NaN, 3])).item())).toBe(true)
	expect(() => max(Tensor.zeros([0]))).toThrow(RangeError)
	expect(sum(Tensor.zeros([0])).item()).toBe(0)
})

test('sums broadcast results back to the source shape', () => {
	expect(sumToShape(m, [3]).toArray()).toEqual([5, 7, 9])
	expect(sumToShape(m, [2, 1]).toArray()).toEqual([[6], [15]])
	expect(sumToShape(m, [1, 3]).toArray()).toEqual([[5, 7, 9]])
	expect(sumToShape(m, []).item()).toBe(21)
	expect(() => sumToShape(m, [4])).toThrow(RangeError)
})

test('accumulates in double precision', () => {
	const big = Tensor.full([100000], 0.1)
	expect(Math.abs(sum(big).item() - 10000)).toBeLessThan(0.01)
})
