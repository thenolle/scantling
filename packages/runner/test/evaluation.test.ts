import { expect, test } from 'bun:test'
import { Tensor } from '@scantling/tensor'
import { DEFAULT_SPECIAL_TOKENS, Tokenizer } from '@scantling/tokenizer'
import { bitsPerByte, smokeChecks, tokenBytes, unigramLoss } from '../src/evaluation.ts'

test('a unigram model on uniform data scores ln(vocabulary)', () => {
	const train = Array.from({ length: 40 }, (_, index) => index % 4)
	expect(unigramLoss(train, [0, 1, 2, 3, 3, 2], 4)).toBeCloseTo(Math.log(4), 9)
})

test('a unigram model rewards frequent tokens and punishes unseen ones', () => {
	const train = Array.from({ length: 100 }, () => 0)
	expect(unigramLoss(train, [0], 2)).toBeCloseTo(-Math.log(101 / 102), 9)
	expect(unigramLoss(train, [1], 2)).toBeCloseTo(-Math.log(1 / 102), 9)
})

test('bits per byte follows from the loss, token count and byte count', () => {
	expect(bitsPerByte(Math.LN2, 1, 1)).toBeCloseTo(1, 9)
	expect(bitsPerByte(2 * Math.LN2, 4, 2)).toBeCloseTo(4, 9)
})

test('token byte counts add up', () => {
	const tokenizer = new Tokenizer(DEFAULT_SPECIAL_TOKENS)
	const ids = tokenizer.encode('héllo')
	expect(tokenBytes(tokenizer, ids)).toBe(new TextEncoder().encode('héllo').length)
})

const params = (data: number[]) => [{ name: 'w', variable: { value: Tensor.from([data.length], data) } }]

test('the smoke checks flag broken weights, non-finite loss and a model that does not beat the baseline', () => {
	expect(smokeChecks(params([1, 2]), 3, 5).every((check) => check.passed)).toBe(true)
	const broken = smokeChecks(params([1, Number.NaN]), 3, 5)
	expect(broken[0]).toMatchObject({ passed: false })
	expect(broken[0]!.detail).toContain('w')
	expect(smokeChecks(params([1]), Number.NaN, 5)[1]!.passed).toBe(false)
	expect(smokeChecks(params([1]), 6, 5)[2]!.passed).toBe(false)
})
