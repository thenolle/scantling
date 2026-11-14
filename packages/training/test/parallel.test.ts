import { afterEach, expect, test } from 'bun:test'
import { Transformer } from '@scantling/nn'
import { allClose, Random } from '@scantling/tensor'
import { AdamW, ParallelTrainer, RandomBatchSource, Sgd, TokenWindows, Trainer, constantRate, estimateParallelMemory } from '../src/index.ts'

const config = { vocabSize: 16, dim: 16, layers: 1, heads: 2, maxSeqLen: 8 }
const windows = new TokenWindows(Int32Array.from({ length: 300 }, (_, index) => (index * 5 + 1) % 16), 8)
const source = (): RandomBatchSource => new RandomBatchSource(windows, 3, new Random(7))
const open: ParallelTrainer[] = []

afterEach(async () => {
	await Promise.all(open.splice(0).map((trainer) => trainer.close()))
})

function parallel(workers: number, rounds = 1, model = new Transformer(config, 4)): { model: Transformer; trainer: ParallelTrainer } {
	const trainer = new ParallelTrainer({ model, optimizer: new Sgd(), schedule: constantRate(0.1), workers, rounds })
	open.push(trainer)
	return { model, trainer }
}

const sameWeights = (a: Transformer, b: Transformer): void => {
	const left = a.parameters()
	const right = b.parameters()
	left.forEach((parameter, index) => expect(allClose(parameter.value, right[index]!.value, { rtol: 1e-4, atol: 1e-6 })).toBe(true))
}

test('two workers match single-threaded gradient accumulation', async () => {
	const reference = new Transformer(config, 4)
	const sequential = new Trainer({ model: reference, optimizer: new Sgd(), schedule: constantRate(0.1), accumulationSteps: 2 })
	const sequentialSource = source()
	const { model, trainer } = parallel(2)
	const parallelSource = source()
	for (let step = 0; step < 3; step += 1) {
		const expected = sequential.trainStep(sequentialSource)
		const actual = await trainer.trainStep(parallelSource)
		expect(actual.loss).toBeCloseTo(expected.loss, 5)
		expect(actual.tokens).toBe(expected.tokens)
	}
	sameWeights(reference, model)
	expect(trainer.step).toBe(3)
})

test('rounds are equivalent to more workers', async () => {
	const wide = parallel(2, 1)
	const narrow = parallel(1, 2)
	const wideSource = source()
	const narrowSource = source()
	for (let step = 0; step < 2; step += 1) {
		await wide.trainer.trainStep(wideSource)
		await narrow.trainer.trainStep(narrowSource)
	}
	sameWeights(wide.model, narrow.model)
})

test('the loss falls while training in parallel with AdamW', async () => {
	const model = new Transformer(config, 4)
	const trainer = new ParallelTrainer({ model, optimizer: new AdamW(), schedule: constantRate(3e-3), workers: 2, maxGradNorm: 1 })
	open.push(trainer)
	const summary = await trainer.run({ steps: 25, source: source() })
	expect(summary.history.at(-1)!.loss).toBeLessThan(summary.history[0]!.loss)
	expect(trainer.skippedSteps).toBe(0)
})

test('a failing job is reported and the pool keeps working', async () => {
	const { trainer } = parallel(2)
	const bad = { next: () => ({ inputs: [99, 0, 0, 0, 0, 0, 0, 0], targets: [0, 0, 0, 0, 0, 0, 0, 0], batch: 1, length: 8 }) }
	expect(trainer.trainStep(bad)).rejects.toThrow(/training worker failed/)
	const metrics = await trainer.trainStep(source())
	expect(Number.isFinite(metrics.loss)).toBe(true)
})

test('memory limits and a closed trainer are enforced', async () => {
	expect(estimateParallelMemory(1000, 4)).toBe(4 * 1000 * 17)
	expect(() => new ParallelTrainer({ model: new Transformer(config, 1), optimizer: new Sgd(), schedule: constantRate(0.1), workers: 2, maxMemoryBytes: 1000 })).toThrow(RangeError)
	const { trainer } = parallel(1)
	await trainer.close()
	expect(trainer.trainStep(source())).rejects.toThrow('closed')
})
