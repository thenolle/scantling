import { expect, test } from 'bun:test'
import { ops, parameter } from '@scantling/autodiff'
import { Transformer } from '@scantling/nn'
import { allClose, Random, Tensor } from '@scantling/tensor'
import { AdamW, RandomBatchSource, Sgd, TokenWindows, Trainer, constantRate, warmupCosine } from '../src/index.ts'
import type { Batch, BatchSource, TrainableModel } from '../src/index.ts'

const entry = (name: string, tensor: Tensor, gradient: Tensor) => {
	const variable = parameter(tensor)
	variable.grad = gradient
	return { name, variable }
}

test('AdamW takes a normalised step and decays only matrices', () => {
	const matrix = entry('matrix', Tensor.from([1, 1], [1]), Tensor.from([1, 1], [0.5]))
	const vector = entry('vector', Tensor.from([1], [1]), Tensor.from([1], [0.5]))
	new AdamW({ weightDecay: 0.1 }).step([matrix, vector], 0.1)
	expect(matrix.variable.value.data[0]).toBeCloseTo(0.89, 6)
	expect(vector.variable.value.data[0]).toBeCloseTo(0.9, 6)
})

test('AdamW bias correction keeps early steps full size', () => {
	const param = entry('p', Tensor.from([1], [1]), Tensor.from([1], [1]))
	const optimizer = new AdamW({ weightDecay: 0 })
	optimizer.step([param], 0.1)
	optimizer.step([param], 0.1)
	expect(param.variable.value.data[0]).toBeCloseTo(0.8, 5)
})

test('SGD with momentum accumulates velocity', () => {
	const param = entry('p', Tensor.from([1], [1]), Tensor.from([1], [1]))
	const optimizer = new Sgd({ momentum: 0.9 })
	optimizer.step([param], 0.1)
	expect(param.variable.value.data[0]).toBeCloseTo(0.9, 6)
	optimizer.step([param], 0.1)
	expect(param.variable.value.data[0]).toBeCloseTo(0.71, 6)
})

test('optimizer state can be saved and restored exactly', () => {
	const make = () => entry('p', Tensor.from([2, 1], [1, 2]), Tensor.from([2, 1], [0.3, -0.7]))
	const a = make()
	const first = new AdamW()
	first.step([a], 0.01)
	const saved = first.state()
	const b = entry('p', a.variable.value.clone(), Tensor.from([2, 1], [0.3, -0.7]))
	const restored = new AdamW()
	restored.loadState(saved)
	first.step([a], 0.01)
	restored.step([b], 0.01)
	expect(allClose(a.variable.value, b.variable.value, { rtol: 0, atol: 0 })).toBe(true)
	expect(saved.step).toBe(1)
})

test('the warmup-cosine schedule rises, decays and holds its floor', () => {
	const schedule = warmupCosine({ peak: 1, warmupSteps: 10, totalSteps: 110 })
	expect(schedule(0)).toBeCloseTo(0.1, 9)
	expect(schedule(9)).toBeCloseTo(1, 9)
	expect(schedule(10)).toBeCloseTo(1, 9)
	expect(schedule(60)).toBeCloseTo(0.55, 9)
	expect(schedule(110)).toBeCloseTo(0.1, 9)
	expect(schedule(500)).toBeCloseTo(0.1, 9)
	for (let step = 10; step < 110; step += 1) expect(schedule(step + 1)).toBeLessThanOrEqual(schedule(step))
	expect(() => warmupCosine({ peak: 1, warmupSteps: 10, totalSteps: 10 })).toThrow(RangeError)
})

test('token windows align targets one step ahead', () => {
	const windows = new TokenWindows(Int32Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 4)
	expect(windows.count).toBe(6)
	expect(windows.window(2)).toEqual({ inputs: Int32Array.from([2, 3, 4, 5]), targets: Int32Array.from([3, 4, 5, 6]) })
	const sampled = windows.sample(3, new Random(1))
	sampled.inputs.length === 12 && expect(Array.from(sampled.targets).every((value, index) => value === (sampled.inputs[index] as number) + 1)).toBe(true)
	const evaluation = windows.evaluationBatches(2, 10)
	expect(evaluation.map((batch) => batch.batch)).toEqual([2])
	expect(Array.from(evaluation[0]!.inputs)).toEqual([0, 1, 2, 3, 4, 5, 6, 7])
	expect(() => new TokenWindows(Int32Array.from([1, 2]), 4)).toThrow(RangeError)
})

const config = { vocabSize: 16, dim: 16, layers: 1, heads: 2, maxSeqLen: 8 }
const rows = (...starts: number[]): Batch => {
	const inputs = Int32Array.from(starts.flatMap((start) => [0, 1, 2, 3].map((offset) => (start + offset) % 16)))
	const targets = Int32Array.from(starts.flatMap((start) => [1, 2, 3, 4].map((offset) => (start + offset) % 16)))
	return { inputs, targets, batch: starts.length, length: 4 }
}
const fixed = (...batches: Batch[]): BatchSource => {
	let index = 0
	return { next: () => batches[index++ % batches.length]! }
}

test('accumulating two micro-batches equals one batch of both', () => {
	const whole = new Transformer(config, 4)
	const split = new Transformer(config, 4)
	new Trainer({ model: whole, optimizer: new Sgd(), schedule: constantRate(0.1) }).trainStep(fixed(rows(0, 5, 9, 12)))
	new Trainer({ model: split, optimizer: new Sgd(), schedule: constantRate(0.1), accumulationSteps: 2 }).trainStep(fixed(rows(0, 5), rows(9, 12)))
	const left = whole.parameters()
	const right = split.parameters()
	left.forEach((parameterValue, index) => expect(allClose(parameterValue.value, right[index]!.value, { rtol: 1e-4, atol: 1e-6 })).toBe(true))
})

function stub(loss: (p: ReturnType<typeof parameter>) => ReturnType<typeof ops.sum>, initial: number[]): TrainableModel & { p: ReturnType<typeof parameter> } {
	const p = parameter(Tensor.from([initial.length], initial))
	return { p, loss: () => loss(p), namedParameters: () => [{ name: 'p', variable: p }] }
}
const nothing: BatchSource = { next: () => ({ inputs: [], targets: [], batch: 1, length: 1 }) }

test('gradients are clipped and the unclipped norm is reported', () => {
	const model = stub((p) => ops.sum(ops.mul(p, p)), [3, 4])
	const metrics = new Trainer({ model, optimizer: new Sgd(), schedule: constantRate(1), maxGradNorm: 1 }).trainStep(nothing)
	expect(metrics.gradNorm).toBeCloseTo(10, 5)
	expect(Array.from(model.p.value.data)).toEqual([expect.closeTo(2.4, 5), expect.closeTo(3.2, 5)])
})

test('a non-finite loss skips the step and leaves the weights alone', () => {
	const model = stub((p) => ops.sum(ops.mulScalar(p, Number.NaN)), [1, 2])
	const trainer = new Trainer({ model, optimizer: new AdamW(), schedule: constantRate(0.1) })
	const metrics = trainer.trainStep(nothing)
	expect(metrics.skipped).toBe(true)
	expect(Array.from(model.p.value.data)).toEqual([1, 2])
	expect(trainer.skippedSteps).toBe(1)
})

test('a run can be cancelled', async () => {
	const model = stub((p) => ops.sum(ops.mul(p, p)), [1])
	const controller = new AbortController()
	const trainer = new Trainer({ model, optimizer: new Sgd(), schedule: constantRate(0.01) })
	const summary = await trainer.run({ steps: 100, source: nothing, signal: controller.signal, onStep: (metrics) => metrics.step === 2 && controller.abort() })
	expect(summary.stopped).toBe(true)
	expect(summary.history).toHaveLength(3)
})

test('a small Transformer learns a repeating sequence from random weights', async () => {
	const cycle = Array.from({ length: 16 }, (_, index) => (index * 7 + 3) % 32)
	const tokens = Int32Array.from({ length: 320 }, (_, index) => cycle[index % 16]!)
	const windows = new TokenWindows(tokens, 16)
	const model = new Transformer({ vocabSize: 32, dim: 32, layers: 2, heads: 2, maxSeqLen: 32 }, 1)
	const trainer = new Trainer({ model, optimizer: new AdamW(), schedule: warmupCosine({ peak: 5e-3, warmupSteps: 10, totalSteps: 150 }), maxGradNorm: 1 })
	const summary = await trainer.run({ steps: 150, source: new RandomBatchSource(windows, 8, new Random(2)) })
	const initial = summary.history.slice(0, 5).reduce((total, metrics) => total + metrics.loss, 0) / 5
	const final = summary.history.slice(-5).reduce((total, metrics) => total + metrics.loss, 0) / 5
	expect(Math.abs(initial - Math.log(32))).toBeLessThan(0.4)
	expect(final).toBeLessThan(initial * 0.25)
	expect(trainer.evaluate(windows.evaluationBatches(8, 4)).loss).toBeLessThan(initial * 0.25)
	expect(trainer.skippedSteps).toBe(0)
})
