import { expect, test } from 'bun:test'
import { Tensor } from '@scantling/tensor'
import { hasMismatchedBrackets, matchingPrefix, nextTokenAccuracy } from '../src/quality.ts'

test('bracket checks accept balanced and cut-off code and reject crossings', () => {
	expect(hasMismatchedBrackets('function f(a) { return [a, (1)] }')).toBe(false)
	expect(hasMismatchedBrackets('if (x) { call(')).toBe(false)
	expect(hasMismatchedBrackets('(]')).toBe(true)
	expect(hasMismatchedBrackets('a }')).toBe(true)
	expect(hasMismatchedBrackets("const s = ')' // ] not a bracket\n/* } */ f()")).toBe(false)
	expect(hasMismatchedBrackets('const t = `${a}` + (')).toBe(false)
})

test('matching prefix counts shared leading tokens', () => {
	expect(matchingPrefix([1, 2, 3], [1, 2, 9])).toBe(2)
	expect(matchingPrefix([1, 2], [1, 2, 3])).toBe(2)
	expect(matchingPrefix([5], [6])).toBe(0)
})

test('next-token accuracy ranks the true token among the logits', () => {
	const logits = Tensor.from([1, 3, 6], [0, 5, 1, 2, 3, 4, 9, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 7])
	const model = { config: { vocabSize: 6 }, forward: () => ({ value: logits }) } as unknown as Parameters<typeof nextTokenAccuracy>[0]
	const batch = { inputs: [0, 0, 0], targets: [1, 3, 5], batch: 1, length: 3 }
	const result = nextTokenAccuracy(model, [batch])
	expect(result.top1).toBeCloseTo(2 / 3, 9)
	expect(result.top5).toBeCloseTo(1, 9)
	expect(result.tokens).toBe(3)
})
