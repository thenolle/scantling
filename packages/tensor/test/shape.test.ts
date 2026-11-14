import { expect, test } from 'bun:test'
import { broadcastShapes, broadcastStrides, normalizeAxis, resolveReshape, sizeOf, stridesOf } from '../src/shape.ts'

test('computes sizes and row-major strides', () => {
	expect(sizeOf([])).toBe(1)
	expect(sizeOf([2, 3, 4])).toBe(24)
	expect(stridesOf([2, 3, 4])).toEqual([12, 4, 1])
})

test('broadcasts shapes with NumPy rules', () => {
	expect(broadcastShapes([2, 3], [3])).toEqual([2, 3])
	expect(broadcastShapes([2, 1], [1, 3])).toEqual([2, 3])
	expect(broadcastShapes([], [4])).toEqual([4])
	expect(() => broadcastShapes([2, 3], [2])).toThrow(RangeError)
})

test('gives broadcast dimensions a zero stride', () => {
	expect(broadcastStrides([3], [2, 3])).toEqual([0, 1])
	expect(broadcastStrides([2, 1], [2, 3])).toEqual([1, 0])
})

test('normalizes axes and resolves reshapes', () => {
	expect(normalizeAxis(-1, 3)).toBe(2)
	expect(() => normalizeAxis(3, 3)).toThrow(RangeError)
	expect(resolveReshape(12, [3, -1])).toEqual([3, 4])
	expect(() => resolveReshape(12, [5, -1])).toThrow(RangeError)
	expect(() => resolveReshape(12, [-1, -1])).toThrow(RangeError)
})
