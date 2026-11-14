import { expect, test } from 'bun:test'
import { concat, gatherRows, permute, scatterAddRowsInPlace, sliceAxis, transpose } from '../src/layout.ts'
import { Tensor } from '../src/tensor.ts'

const t = (nested: Parameters<typeof Tensor.fromNested>[0]): Tensor => Tensor.fromNested(nested)

test('transposes a matrix', () => {
	expect(transpose(t([[1, 2, 3], [4, 5, 6]])).toArray()).toEqual([[1, 4], [2, 5], [3, 6]])
})

test('permutes three axes', () => {
	const source = Tensor.arange(24).reshape([2, 3, 4])
	const result = permute(source, [2, 0, 1])
	expect(result.shape).toEqual([4, 2, 3])
	expect(result.at(1, 1, 2)).toBe(source.at(1, 2, 1))
	expect(() => permute(source, [0, 0, 1])).toThrow(RangeError)
})

test('slices along an axis', () => {
	const m = t([[1, 2, 3], [4, 5, 6]])
	expect(sliceAxis(m, 1, 1, 3).toArray()).toEqual([[2, 3], [5, 6]])
	expect(sliceAxis(m, 0, 1, 2).toArray()).toEqual([[4, 5, 6]])
	expect(() => sliceAxis(m, 1, 2, 5)).toThrow(RangeError)
})

test('concatenates along an axis', () => {
	expect(concat([t([[1, 2]]), t([[3, 4]])], 0).toArray()).toEqual([[1, 2], [3, 4]])
	expect(concat([t([[1], [2]]), t([[3], [4]])], 1).toArray()).toEqual([[1, 3], [2, 4]])
	expect(() => concat([t([[1, 2]]), t([[3]])], 0)).toThrow(RangeError)
})

test('gathers and scatters rows', () => {
	const table = t([[1, 2], [3, 4], [5, 6]])
	expect(gatherRows(table, [2, 0, 2]).toArray()).toEqual([[5, 6], [1, 2], [5, 6]])
	expect(() => gatherRows(table, [3])).toThrow(RangeError)
	const target = Tensor.zeros([3, 2])
	scatterAddRowsInPlace(target, [2, 0, 2], t([[1, 1], [2, 2], [3, 3]]))
	expect(target.toArray()).toEqual([[2, 2], [0, 0], [4, 4]])
})
