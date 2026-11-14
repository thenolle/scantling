import { expect } from 'bun:test'
import { backward, noGrad, zeroGrad } from '@scantling/autodiff'
import { Transformer } from '@scantling/nn'
import { AdamW } from '@scantling/training'
import { Random } from '@scantling/tensor'
import { CudaDevice, GpuModel, GpuTrainer } from '../src/index.ts'
import { gpu } from './available.ts'

const config = { vocabSize: 50, dim: 32, layers: 2, heads: 4, maxSeqLen: 16 }

function randomised(seed: number): Transformer {
	const model = new Transformer(config, seed)
	const random = new Random(seed + 100)
	for (const { name, variable } of model.namedParameters()) {
		const data = variable.value.data
		for (let index = 0; index < data.length; index += 1) data[index] = (name.endsWith('norm.weight') ? 1 : 0) + (name.endsWith('norm.weight') ? 0.1 : 0.3) * random.normal()
	}
	return model
}

function batchOf(rows: number, length: number, seed: number): { inputs: number[]; targets: number[] } {
	const random = new Random(seed)
	return { inputs: Array.from({ length: rows * length }, () => random.nextInt(50)), targets: Array.from({ length: rows * length }, () => random.nextInt(50)) }
}

gpu('the GPU loss and every gradient match the CPU autograd model', () => {
	const model = randomised(3)
	const [rows, length] = [3, 12]
	const { inputs, targets } = batchOf(rows, length, 9)
	const variables = model.namedParameters()
	zeroGrad(variables.map((entry) => entry.variable))
	const cpuLoss = model.loss(inputs, targets, rows, length)
	backward(cpuLoss)
	const expected = new Map(variables.map(({ name, variable }) => [name, Float32Array.from(variable.grad!.data)] as const))
	const device = CudaDevice.create()
	const gpuModel = new GpuModel(device, model, { batch: rows, length })
	gpuModel.uploadWeights()
	gpuModel.zeroGrads()
	const loss = gpuModel.forward(inputs, targets, rows)
	gpuModel.backward(rows)
	gpuModel.downloadGrads(1)
	expect(Math.abs(loss - cpuLoss.value.item())).toBeLessThanOrEqual(1e-4 * Math.abs(cpuLoss.value.item()))
	for (const { name, variable } of variables) {
		const want = expected.get(name)!
		const got = variable.grad!.data
		const peak = want.reduce((largest, value) => Math.max(largest, Math.abs(value)), 0)
		for (let index = 0; index < want.length; index += 1) expect(Math.abs(got[index]! - want[index]!), `${name}[${index}] is ${got[index]} but should be ${want[index]}`).toBeLessThanOrEqual(2e-3 * peak + 1e-6)
	}
	gpuModel.dispose()
	device.dispose()
})

gpu('gradients accumulate across micro-batches and a smaller batch uses part of the capacity', () => {
	const model = randomised(4)
	const device = CudaDevice.create()
	const gpuModel = new GpuModel(device, model, { batch: 4, length: 10 })
	const { inputs, targets } = batchOf(2, 10, 5)
	gpuModel.uploadWeights()
	const cpuLoss = noGrad(() => model.loss(inputs, targets, 2, 10)).value.item()
	expect(Math.abs(gpuModel.forward(inputs, targets, 2) - cpuLoss)).toBeLessThanOrEqual(1e-4 * cpuLoss)
	gpuModel.zeroGrads()
	gpuModel.forward(inputs, targets, 2)
	gpuModel.backward(2)
	gpuModel.downloadGrads(1)
	const once = model.namedParameters().map(({ variable }) => Float32Array.from(variable.grad!.data))
	gpuModel.zeroGrads()
	for (let repeat = 0; repeat < 2; repeat += 1) {
		gpuModel.forward(inputs, targets, 2)
		gpuModel.backward(2)
	}
	gpuModel.downloadGrads(0.5)
	model.namedParameters().forEach(({ name, variable }, position) => {
		const peak = once[position]!.reduce((largest, value) => Math.max(largest, Math.abs(value)), 0)
		for (let index = 0; index < once[position]!.length; index += 1) expect(Math.abs(variable.grad!.data[index]! - once[position]![index]!), `${name}[${index}]`).toBeLessThanOrEqual(1e-3 * peak + 1e-6)
	})
	expect(() => gpuModel.forward(inputs, targets, 5)).toThrow(RangeError)
	gpuModel.dispose()
	device.dispose()
})

gpu('the GPU trainer lowers the loss through the shared trainer logic', async () => {
	const model = randomised(6)
	const { inputs, targets } = batchOf(4, 12, 7)
	const device = CudaDevice.create()
	const trainer = new GpuTrainer({ model, device, optimizer: new AdamW(), schedule: () => 0.01, microBatch: 4, length: 12, accumulation: 2, maxGradNorm: 1 })
	const source = { next: () => ({ inputs, targets, batch: 4, length: 12 }) }
	const summary = await trainer.run({ steps: 25, source })
	expect(summary.history[24]!.loss).toBeLessThan(summary.history[0]!.loss * 0.7)
	expect(trainer.step).toBe(25)
	const evaluated = trainer.evaluate([{ inputs, targets, batch: 4, length: 12 }])
	expect(Math.abs(evaluated.loss - noGrad(() => model.loss(inputs, targets, 4, 12)).value.item())).toBeLessThanOrEqual(1e-3)
	await trainer.close()
	device.dispose()
})
