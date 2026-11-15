import { afterEach, expect, test } from 'bun:test'
import { mkdtemp, readdir, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Transformer } from '@scantling/nn'
import type { ModelConfigInput } from '@scantling/nn'
import { allClose, Random, Tensor } from '@scantling/tensor'
import { AdamW, RandomBatchSource, TokenWindows, Trainer, warmupCosine } from '@scantling/training'
import { CheckpointStore, CorruptCheckpointError, applyWeights, weightsOf } from '../src/index.ts'
import type { CheckpointState } from '../src/index.ts'

const roots: string[] = []
afterEach(async () => {
	await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })))
})
const temporary = async (): Promise<string> => {
	const root = await mkdtemp(join(tmpdir(), 'scantling-ckpt-'))
	roots.push(root)
	return root
}

const dummy = (step: number): CheckpointState => ({
	step,
	skippedSteps: 0,
	config: { a: 1 } as unknown as ModelConfigInput,
	weights: new Map([['w', Tensor.from([2, 2], [1, 2, 3, 4])]]),
	optimizer: { kind: 'adamw', step, tensors: new Map([['m:w', Tensor.from([4], [0.5, 0.5, 0.5, 0.5])]]) },
	random: { batches: new Random(3).state() },
	metrics: { evalLoss: 2 }
})

test('a checkpoint round-trips exactly', async () => {
	const store = new CheckpointStore(await temporary())
	const id = store.save(dummy(7))
	expect(id).toBe('step-00000007')
	const loaded = store.load(id)
	expect(loaded.step).toBe(7)
	expect(loaded.weights.get('w')!.toArray()).toEqual([[1, 2], [3, 4]])
	expect(loaded.optimizer.tensors.get('m:w')!.toArray()).toEqual([0.5, 0.5, 0.5, 0.5])
	expect(loaded.random['batches']).toEqual(new Random(3).state())
	expect(loaded.metrics).toEqual({ evalLoss: 2 })
	expect(() => store.save(dummy(7))).toThrow()
	expect((await readdir(store.root)).some((name) => name.startsWith('.tmp-'))).toBe(false)
})

test('corruption is detected', async () => {
	const store = new CheckpointStore(await temporary())
	const id = store.save(dummy(1))
	const path = join(store.root, id, 'tensors.bin')
	const bytes = new Uint8Array(await Bun.file(path).arrayBuffer())
	bytes[3] = bytes[3]! ^ 0xff
	await Bun.write(path, bytes)
	expect(() => store.load(id)).toThrow(CorruptCheckpointError)
	await Bun.write(join(store.root, id, 'meta.json'), '{ broken')
	expect(() => store.load(id)).toThrow(CorruptCheckpointError)
	expect(() => store.load('step-00000099')).toThrow()
	expect(() => store.load('../escape')).toThrow(RangeError)
})

const config = { vocabSize: 16, dim: 16, layers: 1, heads: 2, maxSeqLen: 8 }

function makeRun() {
	const tokens = Int32Array.from({ length: 200 }, (_, index) => (index * 5 + 1) % 16)
	const model = new Transformer(config, 4)
	const optimizer = new AdamW()
	const source = new RandomBatchSource(new TokenWindows(tokens, 8), 4, new Random(9))
	const trainer = new Trainer({ model, optimizer, schedule: warmupCosine({ peak: 3e-3, warmupSteps: 3, totalSteps: 12 }), maxGradNorm: 1 })
	return { model, optimizer, source, trainer }
}

test('resuming from disk reproduces an uninterrupted run bit for bit', async () => {
	const straight = makeRun()
	for (let step = 0; step < 12; step += 1) straight.trainer.trainStep(straight.source)
	const first = makeRun()
	for (let step = 0; step < 6; step += 1) first.trainer.trainStep(first.source)
	const store = new CheckpointStore(await temporary())
	const optimizerState = first.optimizer.state()
	const id = store.save({
		step: first.trainer.step,
		skippedSteps: first.trainer.skippedSteps,
		config: first.model.config,
		weights: weightsOf(first.model.namedParameters()),
		optimizer: { kind: first.optimizer.kind, step: optimizerState.step, tensors: optimizerState.tensors },
		random: { batches: first.source.random.state() },
		metrics: {}
	})
	const resumed = makeRun()
	const saved = store.load(id)
	applyWeights(resumed.model.namedParameters(), saved.weights)
	resumed.optimizer.loadState({ step: saved.optimizer.step, tensors: saved.optimizer.tensors })
	resumed.source.random.restore(saved.random['batches']!)
	resumed.trainer.resume(saved.step, saved.skippedSteps)
	for (let step = 6; step < 12; step += 1) resumed.trainer.trainStep(resumed.source)
	const left = straight.model.parameters()
	const right = resumed.model.parameters()
	left.forEach((parameter, index) => expect(allClose(parameter.value, right[index]!.value, { rtol: 0, atol: 0 })).toBe(true))
})

test('loading rejects a checkpoint of a different model', () => {
	const model = new Transformer(config, 1)
	const other = new Transformer({ ...config, layers: 2 }, 1)
	expect(() => applyWeights(model.namedParameters(), weightsOf(other.namedParameters()))).toThrow(RangeError)
	const wider = new Transformer({ ...config, dim: 32 }, 1)
	expect(() => applyWeights(model.namedParameters(), weightsOf(wider.namedParameters()))).toThrow(RangeError)
})

const pass = [{ name: 'validation loss is finite', passed: true }]

test('the gate promotes real improvements and rejects everything else', async () => {
	const store = new CheckpointStore(await temporary())
	for (const step of [1, 2, 3, 4, 5]) store.save(dummy(step))
	expect(store.promote('step-00000001', { evalLoss: 3, regression: pass }).accepted).toBe(true)
	expect(store.stable()?.id).toBe('step-00000001')
	const worse = store.promote('step-00000002', { evalLoss: 3.5, regression: pass })
	expect(worse.accepted).toBe(false)
	expect(worse.reasons[0]).toContain('worse')
	const regression = store.promote('step-00000003', { evalLoss: 1, regression: [{ name: 'codegen suite', passed: false, detail: '3 of 40 failed' }] })
	expect(regression.accepted).toBe(false)
	expect(regression.reasons[0]).toContain('codegen suite')
	expect(store.promote('step-00000003', { evalLoss: Number.NaN, regression: pass }).accepted).toBe(false)
	expect(store.promote('step-00000003', { evalLoss: 1, regression: [] }).accepted).toBe(false)
	expect(store.stable()?.id).toBe('step-00000001')
	expect(store.promote('step-00000004', { evalLoss: 2.5, regression: pass }).accepted).toBe(true)
	expect(store.stable()?.id).toBe('step-00000004')
	expect(store.history().map((entry) => entry.action)).toEqual(['promote', 'reject', 'reject', 'reject', 'reject', 'promote'])
})

test('rollback restores the previous stable checkpoint and prune protects it', async () => {
	const store = new CheckpointStore(await temporary())
	for (const step of [1, 2, 3, 4, 5, 6]) store.save(dummy(step))
	store.promote('step-00000002', { evalLoss: 3, regression: pass })
	store.promote('step-00000005', { evalLoss: 2, regression: pass })
	expect(store.rollback()).toBe('step-00000002')
	expect(store.stable()?.id).toBe('step-00000002')
	expect(store.stable()?.evalLoss).toBe(3)
	expect(store.prune(2).sort()).toEqual(['step-00000001', 'step-00000003', 'step-00000004'])
	expect(store.list().map((meta) => meta.id)).toEqual(['step-00000002', 'step-00000005', 'step-00000006'])
	expect(store.latest()).toBe('step-00000006')
})
