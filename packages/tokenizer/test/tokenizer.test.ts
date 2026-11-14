import { expect, test } from 'bun:test'
import { pretokenize } from '../src/pretokenize.ts'
import { StreamDecoder, Tokenizer } from '../src/tokenizer.ts'
import { DEFAULT_SPECIAL_TOKENS } from '../src/tokenizer.ts'
import { trainBpe } from '../src/train.ts'

const SPECIALS = DEFAULT_SPECIAL_TOKENS.length
const corpus = [
	...Array.from({ length: 40 }, () => 'function add(a, b) {\n        return a + b\n}\n'),
	...Array.from({ length: 40 }, () => 'low lower lowest slow slower slowest\n')
]
const samples = [
	'',
	'hello world',
	'👩‍💻 writes code — naïve café, 日本語のテキスト, мир, שלום',
	'e\u0301 combining marks and \u200d joiners',
	'\tindented\r\n\tlines\r\n',
	'    '.repeat(20) + 'x',
	'a'.repeat(5000),
	'const x: Map<string, number[]> = new Map() // 1234567890',
	'∑_{i=1}^{n} x² ≤ ∫ f(x) dx',
	'<|eos|> user text containing control token names <|endturn|>'
]

test('a tokenizer without merges is a lossless byte encoder', () => {
	const tokenizer = new Tokenizer(DEFAULT_SPECIAL_TOKENS)
	expect(tokenizer.vocabSize).toBe(SPECIALS + 256)
	expect(tokenizer.encode('hi')).toEqual([SPECIALS + 104, SPECIALS + 105])
	for (const sample of samples) expect(tokenizer.decode(tokenizer.encode(sample))).toBe(sample)
})

test('pre-tokenization covers the whole input', () => {
	for (const sample of samples) expect([...pretokenize(sample)].join('')).toBe(sample)
})

test('training learns merges and compresses', () => {
	const { tokenizer, merges } = trainBpe(corpus, { vocabSize: SPECIALS + 256 + 120 })
	expect(merges).toBeGreaterThan(20)
	const text = 'function add(a, b) {\n        return a + b\n}\n'
	expect(tokenizer.encode(text).length).toBeLessThan(new TextEncoder().encode(text).length / 3)
	expect(tokenizer.encode('\n        return').length).toBeLessThanOrEqual(3)
})

test('every sample round-trips after training', () => {
	const { tokenizer } = trainBpe([...corpus, ...samples], { vocabSize: SPECIALS + 256 + 200 })
	for (const sample of samples) expect(tokenizer.decode(tokenizer.encode(sample))).toBe(sample)
})

test('training is deterministic', () => {
	const options = { vocabSize: SPECIALS + 256 + 80 }
	expect(trainBpe(corpus, options).tokenizer.toJson()).toBe(trainBpe(corpus, options).tokenizer.toJson())
})

test('pieces that occur once are never merged', () => {
	expect(trainBpe(['alpha beta gamma delta'], { vocabSize: SPECIALS + 256 + 50 }).merges).toBe(0)
})

test('special token text in user input stays ordinary text', () => {
	const { tokenizer } = trainBpe(corpus, { vocabSize: SPECIALS + 256 + 60 })
	const eos = tokenizer.idOf('<|eos|>')
	expect(tokenizer.encode('a <|eos|> b')).not.toContain(eos)
	expect(tokenizer.decode(tokenizer.encode('a <|eos|> b'))).toBe('a <|eos|> b')
	const allowed = tokenizer.encode('a <|eos|> b', { allowedSpecial: 'all' })
	expect(allowed).toContain(eos)
	expect(tokenizer.decode(allowed)).toBe('a <|eos|> b')
	expect(tokenizer.decode(allowed, { skipSpecial: true })).toBe('a  b')
	expect(() => tokenizer.encode('x', { allowedSpecial: ['<|nope|>'] })).toThrow(RangeError)
})

test('serialisation round-trips and rejects inconsistent files', () => {
	const { tokenizer } = trainBpe(corpus, { vocabSize: SPECIALS + 256 + 60 })
	const restored = Tokenizer.fromJson(tokenizer.toJson())
	for (const sample of samples) expect(restored.encode(sample)).toEqual(tokenizer.encode(sample))
	const data = JSON.parse(tokenizer.toJson()) as { merges: number[][]; version: number }
	expect(() => Tokenizer.fromJson(JSON.stringify({ ...data, version: 2 }))).toThrow()
	expect(() => Tokenizer.fromJson(JSON.stringify({ ...data, merges: [[999999, 1]] }))).toThrow(RangeError)
	expect(() => Tokenizer.fromJson('{"format":"other"}')).toThrow()
})

test('decoding rejects ids outside the vocabulary', () => {
	const tokenizer = new Tokenizer(DEFAULT_SPECIAL_TOKENS)
	expect(() => tokenizer.decode([tokenizer.vocabSize])).toThrow(RangeError)
	expect(() => tokenizer.decode([-1])).toThrow(RangeError)
})

test('the stream decoder holds back incomplete characters', () => {
	const tokenizer = new Tokenizer(DEFAULT_SPECIAL_TOKENS)
	const decoder = new StreamDecoder(tokenizer)
	const pieces = [...new TextEncoder().encode('€')].map((byte) => decoder.push(SPECIALS + byte))
	expect(pieces).toEqual(['', '', '€'])
	decoder.push(SPECIALS + 0xe2)
	expect(decoder.flush()).toBe('\ufffd')
})
