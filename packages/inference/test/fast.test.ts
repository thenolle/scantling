import { beforeAll, expect, test } from 'bun:test'
import { Transformer } from '@scantling/nn'
import { Random } from '@scantling/tensor'
import { AdamW, RandomBatchSource, TokenWindows, Trainer, warmupCosine } from '@scantling/training'
import { FastDecoder, InferenceSession, gemvDense, gemvInt8, generate, quantizeRows } from '../src/index.ts'

const random = new Random(17)
const fill = (length: number): Float32Array => Float32Array.from({ length }, () => random.normal())

test('dense and int8 matrix-vector products match a reference for awkward sizes', () => {
	for (const rows of [1, 2, 3, 4, 5, 7, 8, 9]) {
		for (const cols of [1, 3, 4, 5, 8, 9]) {
			const w = fill(rows * cols)
			const x = fill(cols)
			const expected = Array.from({ length: rows }, (_, row) => x.reduce((total, value, column) => total + value * w[row * cols + column]!, 0))
			const dense = new Float32Array(rows)
			gemvDense(w, rows, cols, x, dense)
			dense.forEach((value, row) => expect(Math.abs(value - expected[row]!)).toBeLessThan(1e-5 * (1 + Math.abs(expected[row]!))))
			const quantized = quantizeRows(w, rows, cols)
			const approximate = new Float32Array(rows)
			gemvInt8(quantized, x, approximate)
			const magnitude = x.reduce((total, value) => total + Math.abs(value), 0)
			approximate.forEach((value, row) => expect(Math.abs(value - expected[row]!)).toBeLessThanOrEqual((quantized.scales[row]! / 2) * magnitude + 1e-5))
		}
	}
})

test('quantisation reproduces weights to within half a step and handles empty rows', () => {
	const values = Float32Array.from([0.5, -1, 0.25, 0, 0, 0])
	const q = quantizeRows(values, 2, 3)
	expect(q.scales[0]).toBeCloseTo(1 / 127, 9)
	expect(q.scales[1]).toBe(1)
	expect(Array.from(q.data.subarray(3))).toEqual([0, 0, 0])
	q.data.subarray(0, 3).forEach((level, column) => expect(Math.abs(level * q.scales[0]! - values[column]!)).toBeLessThanOrEqual(q.scales[0]! / 2 + 1e-9))
	expect(Math.max(...q.data)).toBeLessThanOrEqual(127)
	expect(Math.min(...q.data)).toBeGreaterThanOrEqual(-127)
})

const base = { vocabSize: 23, dim: 16, layers: 2, heads: 2, maxSeqLen: 16, initStd: 0.3 }
const ids = [3, 7, 1, 9, 22, 0, 4, 4, 11, 5]

test('the fast float32 decoder matches the tensor path', () => {
	for (const config of [base, { ...base, kvHeads: 1 }, { ...base, tieEmbeddings: false }]) {
		const model = new Transformer(config, 5)
		const reference = new InferenceSession(model)
		const decoder = new FastDecoder(model, 'float32')
		for (const id of ids) {
			const expected = reference.feed([id]).data
			const actual = decoder.step(id)
			for (let index = 0; index < expected.length; index += 1) expect(Math.abs(actual[index]! - expected[index]!)).toBeLessThan(1e-4)
		}
		expect(decoder.length).toBe(ids.length)
	}
})

test('int8 logits stay close to float32 logits', () => {
	const model = new Transformer({ ...base, kvHeads: 1 }, 5)
	const exact = new FastDecoder(model, 'float32')
	const approximate = new FastDecoder(model, 'int8')
	let worst = 0
	let total = 0
	let count = 0
	for (const id of ids) {
		const expected = Float32Array.from(exact.step(id))
		const actual = approximate.step(id)
		const range = Math.max(...expected.map(Math.abs))
		for (let index = 0; index < expected.length; index += 1) {
			const error = Math.abs(actual[index]! - expected[index]!) / range
			worst = Math.max(worst, error)
			total += error
			count += 1
		}
	}
	expect(worst).toBeLessThan(0.1)
	expect(total / count).toBeLessThan(0.02)
})

test('the decoder reuses its buffers, is repeatable after a reset, and validates input', () => {
	const decoder = new FastDecoder(new Transformer(base, 5), 'float32')
	const first = decoder.step(3)
	const copy = Float32Array.from(first)
	expect(decoder.step(7)).toBe(first)
	decoder.reset()
	expect(decoder.step(3)).toBe(first)
	expect(first).toEqual(copy)
	expect(() => decoder.step(99)).toThrow(RangeError)
	expect(() => decoder.step(-1)).toThrow(RangeError)
	decoder.reset()
	for (let index = 0; index < 16; index += 1) decoder.step(index % 23)
	expect(() => decoder.step(1)).toThrow(RangeError)
})

const cycle = Array.from({ length: 16 }, (_, index) => (index * 7 + 3) % 32)
let trained: Transformer

beforeAll(async () => {
	trained = new Transformer({ vocabSize: 32, dim: 32, layers: 2, heads: 2, maxSeqLen: 32 }, 1)
	const windows = new TokenWindows(Int32Array.from({ length: 320 }, (_, index) => cycle[index % 16]!), 16)
	const trainer = new Trainer({ model: trained, optimizer: new AdamW(), schedule: warmupCosine({ peak: 5e-3, warmupSteps: 10, totalSteps: 150 }), maxGradNorm: 1 })
	await trainer.run({ steps: 150, source: new RandomBatchSource(windows, 8, new Random(2)) })
})

test('every engine generates the learned sequence, including across a window slide', async () => {
	for (const engine of ['tensor', 'float32', 'int8'] as const) {
		const tokens: number[] = []
		for await (const event of generate(trained, cycle.slice(0, 3), { maxNewTokens: 60, sampling: { temperature: 0 }, engine })) if (event.type === 'token') tokens.push(event.token)
		expect(tokens, engine).toEqual(Array.from({ length: 60 }, (_, index) => cycle[(3 + index) % 16]!))
	}
})
