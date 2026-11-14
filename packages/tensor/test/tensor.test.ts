import { expect, test } from 'bun:test'
import { Tensor } from '../src/tensor.ts'

test('validates the data length against the shape', () => {
	expect(() => new Tensor([2, 2], new Float32Array(3))).toThrow(RangeError)
	expect(() => Tensor.zeros([-1])).toThrow(RangeError)
})

test('round-trips nested arrays', () => {
	const t = Tensor.fromNested([[1, 2, 3], [4, 5, 6]])
	expect(t.shape).toEqual([2, 3])
	expect(t.toArray()).toEqual([[1, 2, 3], [4, 5, 6]])
	expect(t.at(1, 2)).toBe(6)
})

test('rejects ragged nested arrays', () => {
	expect(() => Tensor.fromNested([[1, 2], [3]])).toThrow(RangeError)
	expect(() => Tensor.fromNested([[1], [[2]]])).toThrow(RangeError)
})

test('reshape shares memory and infers one dimension', () => {
	const t = Tensor.arange(6)
	const view = t.reshape([2, -1])
	expect(view.shape).toEqual([2, 3])
	view.set([1, 0], 99)
	expect(t.data[3]).toBe(99)
	expect(t.clone().data).not.toBe(t.data)
})

test('bounds-checks element access', () => {
	const t = Tensor.zeros([2, 2])
	expect(() => t.at(2, 0)).toThrow(RangeError)
	expect(() => t.at(0)).toThrow(RangeError)
	expect(Tensor.scalar(7).item()).toBe(7)
	expect(() => t.item()).toThrow(RangeError)
})
