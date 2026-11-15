import { expect, test } from 'bun:test'
import { Transformer } from '@scantling/nn'
import { Random } from '@scantling/tensor'
import { AdamW, Trainer } from '@scantling/training'
import type { BatchSource } from '@scantling/training'
import { CudaDevice, GpuTrainer, selectDevice } from '../src/index.ts'

function source(seed: number, vocabulary: number, batch: number, length: number): BatchSource {
	const random = new Random(seed)
	return {
		next: () => {
			const inputs = new Int32Array(batch * length)
			const targets = new Int32Array(batch * length)
			for (let index = 0; index < inputs.length; index += 1) {
				inputs[index] = random.nextInt(vocabulary)
				targets[index] = random.nextInt(vocabulary)
			}
			return { inputs, targets, batch, length }
		}
	}
}

test('the resident GPU optimizer follows the CPU optimizer over several steps', async () => {
	const selection = selectDevice('auto')
	if (!(selection.device instanceof CudaDevice)) return
	const config = { vocabSize: 32, dim: 16, layers: 1, heads: 2, maxSeqLen: 16 }
	const cpuModel = new Transformer(config, 3)
	const gpuModel = new Transformer(config, 3)
	const schedule = (): number => 5e-3
	const cpu = new Trainer({ model: cpuModel, optimizer: new AdamW(), schedule, accumulationSteps: 2, maxGradNorm: 0.5 })
	const gpu = new GpuTrainer({ model: gpuModel, optimizer: new AdamW(), schedule, device: selection.device, microBatch: 2, length: 16, accumulation: 2, maxGradNorm: 0.5 })
	expect(gpu.resident).toBe(true)
	const cpuSource = source(5, 32, 2, 16)
	const gpuSource = source(5, 32, 2, 16)
	for (let step = 0; step < 4; step += 1) {
		const expected = cpu.trainStep(cpuSource)
		const actual = gpu.trainStep(gpuSource)
		expect(Math.abs(expected.loss - actual.loss)).toBeLessThan(5e-3)
		expect(Math.abs(expected.gradNorm - actual.gradNorm)).toBeLessThan(0.03 * Math.max(1, expected.gradNorm))
	}
	gpu.syncToCpu()
	expect(gpu.step).toBe(4)
	const reference = new Map(cpuModel.namedParameters().map((entry) => [entry.name, entry.variable.value.data]))
	for (const entry of gpuModel.namedParameters()) {
		const expected = reference.get(entry.name)!
		const actual = entry.variable.value.data
		let sum = 0
		let worst = 0
		for (let index = 0; index < actual.length; index += 1) {
			const difference = Math.abs(actual[index]! - expected[index]!)
			sum += difference
			worst = Math.max(worst, difference)
		}
		expect(sum / actual.length).toBeLessThan(1e-3)
		expect(worst).toBeLessThan(0.05)
	}
	await gpu.close()
})
