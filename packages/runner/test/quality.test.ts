import { expect, test } from 'bun:test'
import { Tensor } from '@scantling/tensor'
import type { TokenFile } from '@scantling/datasets'
import { spreadEvaluationBatches } from '../src/evaluation.ts'
import { START_STATE, endsInLoop, hasMismatchedBrackets, matchingPrefix, nextTokenAccuracy, scanBrackets } from '../src/quality.ts'

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

test('bracket state carries across the prompt boundary', () => {
	const prompt = scanBrackets('call(a, [1, ', START_STATE, true)
	expect(prompt.stack).toEqual(['(', '['])
	expect(scanBrackets('2]) }', { ...prompt, mismatch: false }, true).mismatch).toBe(false)
	expect(scanBrackets('2) }', { ...prompt, mismatch: false }, true).mismatch).toBe(true)
	const inString = scanBrackets("const a = 'x(", START_STATE, true)
	expect(inString.quote).toBe("'")
	expect(scanBrackets("y)' + (", inString, true).mismatch).toBe(false)
})

test('loops are detected by period', () => {
	expect(endsInLoop(Array.from({ length: 30 }, (_, index) => index))).toBe(false)
	expect(endsInLoop([9, 8, ...Array.from({ length: 30 }, (_, index) => [1, 2, 3][index % 3]!)])).toBe(true)
	expect(endsInLoop([1, 2, 3])).toBe(false)
})

test('evaluation windows are spread evenly over the file and are deterministic', () => {
	const file = { count: 100, read: (start: number, count: number) => Int32Array.from({ length: count }, (_, index) => start + index) } as unknown as TokenFile
	const first = spreadEvaluationBatches(file, 8, 2, 3)
	expect(first.length).toBe(3)
	expect(first[0]!.inputs[0]).toBe(0)
	expect(first[2]!.inputs[8]).toBe(91)
	expect(first[0]!.targets[0]).toBe(1)
	expect(spreadEvaluationBatches(file, 8, 2, 3)).toEqual(first)
})
