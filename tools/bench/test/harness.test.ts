import { expect, test } from 'bun:test'
import { measure, median } from '../src/harness.ts'

test('computes the median of odd and even samples', () => {
	expect(median([3, 1, 2])).toBe(2)
	expect(median([4, 1, 2, 3])).toBe(2.5)
})

test('runs at least the requested number of iterations', async () => {
	let calls = 0
	const sample = await measure('count', () => calls++, { warmup: 1, minMs: 0, minRuns: 7 })
	expect(sample.runs).toBe(7)
	expect(calls).toBe(8)
	expect(sample.minMs).toBeLessThanOrEqual(sample.medianMs)
})
