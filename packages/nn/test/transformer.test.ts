import { expect, test } from 'bun:test'
import { allClose, Tensor } from '@scantling/tensor'
import { noGrad } from '@scantling/autodiff'
import { checkModelGradients, parameterCount, presetConfig, resolveConfig, Transformer } from '../src/index.ts'

const tiny = { vocabSize: 11, dim: 8, layers: 2, heads: 2, kvHeads: 1, ffnHidden: 16, maxSeqLen: 8 }
const ids = [1, 4, 2, 9, 3, 3, 7, 0]
const targets = [4, 2, 9, 3, 3, 7, 0, 5]

test('the preset sizes match their arithmetic', () => {
	expect(parameterCount(presetConfig('tiny', 4096))).toBe(1_315_968)
	expect(parameterCount(presetConfig('micro', 4096))).toBeLessThan(parameterCount(presetConfig('tiny', 4096)))
	expect(parameterCount(presetConfig('small', 4096))).toBeGreaterThan(10_000_000)
})

test('the counted parameters equal the analytic formula', () => {
	for (const config of [tiny, { ...tiny, kvHeads: 2 }, { ...tiny, tieEmbeddings: false }]) {
		const model = new Transformer(config)
		expect(model.parameterCount()).toBe(parameterCount(model.config))
	}
	const names = new Transformer(tiny).namedParameters().map((entry) => entry.name)
	expect(new Set(names).size).toBe(names.length)
})

test('invalid configurations are rejected', () => {
	expect(() => resolveConfig({ ...tiny, dim: 9 })).toThrow(RangeError)
	expect(() => resolveConfig({ ...tiny, heads: 2, kvHeads: 3 })).toThrow(RangeError)
	expect(() => resolveConfig({ ...tiny, dim: 6, heads: 2 })).toThrow(RangeError)
	expect(() => new Transformer(tiny).forward(ids, 1, 9)).toThrow(RangeError)
	expect(() => new Transformer(tiny).forward(ids, 2, 5)).toThrow(RangeError)
})

test('initialisation is seeded and follows the scheme', () => {
	const a = new Transformer(tiny, 5)
	const b = new Transformer(tiny, 5)
	const c = new Transformer(tiny, 6)
	expect(allClose(a.embedding.table.value, b.embedding.table.value, { rtol: 0, atol: 0 })).toBe(true)
	expect(allClose(a.embedding.table.value, c.embedding.table.value)).toBe(false)
	expect(a.finalNorm.weight.value.toArray()).toEqual(Array.from({ length: 8 }, () => 1))
	const std = (t: Tensor): number => Math.sqrt(t.data.reduce((total, value) => total + value * value, 0) / t.size)
	const big = new Transformer({ vocabSize: 64, dim: 64, layers: 4, heads: 4, maxSeqLen: 16 }, 1)
	expect(std(big.embedding.table.value)).toBeCloseTo(0.02, 2)
	expect(std(big.blocks[0]!.attention.output.weight.value)).toBeCloseTo(0.02 / Math.sqrt(8), 3)
})

test('logits have the right shape and the initial loss is near ln(vocab)', () => {
	const model = new Transformer({ ...tiny, dim: 64, heads: 4, kvHeads: 4, ffnHidden: 176, vocabSize: 100, maxSeqLen: 16 })
	const logits = noGrad(() => model.forward(ids, 2, 4))
	expect(logits.shape).toEqual([2, 4, 100])
	const loss = noGrad(() => model.loss(ids, targets, 2, 4)).value.item()
	expect(Math.abs(loss - Math.log(100))).toBeLessThan(0.3)
})

test('a position never sees later tokens', () => {
	const model = new Transformer(tiny, 3)
	const changed = [...ids.slice(0, 7), 10]
	const a = noGrad(() => model.forward(ids, 1, 8)).value
	const b = noGrad(() => model.forward(changed, 1, 8)).value
	const row = (t: Tensor, position: number): Tensor => Tensor.from([11], t.data.subarray(position * 11, (position + 1) * 11))
	for (let position = 0; position < 7; position += 1) expect(allClose(row(a, position), row(b, position), { rtol: 1e-5, atol: 1e-6 })).toBe(true)
	expect(allClose(row(a, 7), row(b, 7))).toBe(false)
})

test('batches are independent of each other', () => {
	const model = new Transformer(tiny, 3)
	const together = noGrad(() => model.forward([...ids.slice(0, 4), ...ids.slice(4)], 2, 4)).value
	const alone = noGrad(() => model.forward(ids.slice(4), 1, 4)).value
	expect(allClose(Tensor.from([4, 11], together.data.subarray(4 * 11)), Tensor.from([4, 11], alone.data), { rtol: 1e-5, atol: 1e-6 })).toBe(true)
})

test('backpropagation through the whole model matches finite differences', () => {
	for (const config of [tiny, { ...tiny, kvHeads: 2 }, { ...tiny, tieEmbeddings: false }]) {
		const model = new Transformer({ ...config, initStd: 0.3 }, 11)
		const report = checkModelGradients(model, { inputs: ids, targets, batch: 2, length: 4 }, { samples: 60 })
		const worst = report.worst
		expect(report.ok, `max error ${report.maxAbsError}, worst ${worst?.parameter}[${worst?.index}] analytic ${worst?.analytic} numeric ${worst?.numeric}`).toBe(true)
	}
})

test('tied embeddings receive gradient from both uses', () => {
	const model = new Transformer({ ...tiny, initStd: 0.3 }, 2)
	checkModelGradients(model, { inputs: ids, targets, batch: 2, length: 4 }, { samples: 1 })
	const rowsWithoutInput = [5, 6, 8, 10]
	for (const row of rowsWithoutInput) {
		const gradient = model.embedding.table.grad!.data.subarray(row * 8, (row + 1) * 8)
		expect(gradient.some((value) => value !== 0)).toBe(true)
	}
})
