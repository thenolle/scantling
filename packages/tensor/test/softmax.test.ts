import { expect, test } from 'bun:test'
import { allClose } from '../src/compare.ts'
import { log } from '../src/elementwise.ts'
import { sum } from '../src/reduce.ts'
import { logSoftmax, logSumExp, softmax } from '../src/softmax.ts'
import { Tensor } from '../src/tensor.ts'

const t = (nested: Parameters<typeof Tensor.fromNested>[0]): Tensor => Tensor.fromNested(nested)

test('softmax matches known values and sums to one', () => {
	const result = softmax(t([[1, 2, 3], [0, 0, 0]]))
	expect(allClose(result, t([[0.09003057, 0.24472847, 0.66524096], [1 / 3, 1 / 3, 1 / 3]]), { atol: 1e-6 })).toBe(true)
	expect(allClose(sum(result, { axes: 1 }), t([1, 1]), { atol: 1e-6 })).toBe(true)
})

test('softmax along another axis', () => {
	const result = softmax(t([[1, 2], [3, 4]]), 0)
	expect(allClose(sum(result, { axes: 0 }), t([1, 1]), { atol: 1e-6 })).toBe(true)
})

test('softmax stays finite for very large inputs', () => {
	const result = softmax(t([1000, 1001]))
	expect(allClose(result, t([0.26894142, 0.73105858]), { atol: 1e-6 })).toBe(true)
})

test('log-softmax equals the log of softmax', () => {
	const x = t([[1, 2, 3], [-5, 0, 5]])
	expect(allClose(logSoftmax(x), log(softmax(x)), { atol: 1e-5 })).toBe(true)
	expect(allClose(logSoftmax(t([1, 2, 3])), t([-2.4076059, -1.4076059, -0.4076059]), { atol: 1e-6 })).toBe(true)
})

test('log-sum-exp is stable and honours keepDims', () => {
	expect(logSumExp(t([1, 2, 3])).item()).toBeCloseTo(3.4076059, 5)
	expect(logSumExp(t([1000, 1000])).item()).toBeCloseTo(1000.693147, 3)
	expect(logSumExp(t([[1, 2], [3, 4]]), 1, true).shape).toEqual([2, 1])
})
