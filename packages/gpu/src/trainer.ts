import type { Transformer } from '@scantling/nn'
import { Trainer } from '@scantling/training'
import type { Batch, BatchSource, EvalResult, RunOptions, RunSummary, StepMetrics, TrainerOptions } from '@scantling/training'
import type { CudaDevice } from './cuda/device.ts'
import { GpuModel } from './model.ts'

export interface GpuTrainerOptions extends Omit<TrainerOptions, 'model' | 'accumulationSteps'> {
	readonly model: Transformer
	readonly device: CudaDevice
	/** Sequences per micro-batch, which also sets the activation memory. */
	readonly microBatch: number
	readonly length: number
	/** Micro-batches averaged into one optimizer step. Defaults to 1. */
	readonly accumulation?: number
}

/**
 * Trains on the GPU: gradients come from `GpuModel`, then the shared `Trainer` clips them, skips non-finite steps,
 * applies the optimizer and advances the step counter. Weights, optimizer state and checkpoints stay on the CPU side,
 * so everything built around `Trainer` keeps working. Evaluation also runs on the GPU.
 */
export class GpuTrainer {
	readonly trainer: Trainer
	readonly gpu: GpuModel
	readonly accumulation: number

	constructor(options: GpuTrainerOptions) {
		this.accumulation = options.accumulation ?? 1
		if (!Number.isInteger(this.accumulation) || this.accumulation < 1) throw new RangeError('accumulation must be a positive integer')
		this.trainer = new Trainer({ model: options.model, optimizer: options.optimizer, schedule: options.schedule, ...(options.maxGradNorm === undefined ? {} : { maxGradNorm: options.maxGradNorm }) })
		this.gpu = new GpuModel(options.device, options.model, { batch: options.microBatch, length: options.length })
	}

	get step(): number {
		return this.trainer.step
	}

	get skippedSteps(): number {
		return this.trainer.skippedSteps
	}

	resume(step: number, skippedSteps: number): void {
		this.trainer.resume(step, skippedSteps)
	}

	trainStep(source: BatchSource): StepMetrics {
		const started = performance.now()
		this.gpu.uploadWeights()
		this.gpu.zeroGrads()
		let loss = 0
		let tokens = 0
		for (let micro = 0; micro < this.accumulation; micro += 1) {
			const batch = source.next()
			this.#check(batch)
			loss += this.gpu.forward(batch.inputs, batch.targets, batch.batch) / this.accumulation
			this.gpu.backward(batch.batch)
			tokens += batch.batch * batch.length
		}
		this.gpu.downloadGrads(1 / this.accumulation)
		return this.trainer.applyAccumulated(loss, tokens, started)
	}

	/** Mean loss over `batches`, computed on the GPU with the current CPU weights. */
	evaluate(batches: Iterable<Batch>): EvalResult {
		this.gpu.uploadWeights()
		let total = 0
		let tokens = 0
		for (const batch of batches) {
			this.#check(batch)
			const count = batch.batch * batch.length
			total += this.gpu.forward(batch.inputs, batch.targets, batch.batch) * count
			tokens += count
		}
		const loss = tokens === 0 ? Number.NaN : total / tokens
		return { loss, perplexity: Math.exp(loss), tokens }
	}

	run(options: Omit<RunOptions, 'stepper' | 'evaluator'>): Promise<RunSummary> {
		return this.trainer.run({ ...options, stepper: (source) => this.trainStep(source), evaluator: (batches) => this.evaluate(batches) })
	}

	async close(): Promise<void> {
		this.gpu.dispose()
	}

	#check(batch: Batch): void {
		if (batch.length !== this.gpu.length || batch.batch > this.gpu.capacity) throw new RangeError(`The GPU model was built for up to ${this.gpu.capacity} sequences of ${this.gpu.length} tokens, received ${batch.batch} of ${batch.length}`)
	}
}
