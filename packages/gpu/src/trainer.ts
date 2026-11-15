import type { Transformer } from '@scantling/nn'
import { Tensor } from '@scantling/tensor'
import { Trainer } from '@scantling/training'
import type { Batch, BatchSource, EvalResult, RunOptions, RunSummary, StepMetrics, TrainerOptions } from '@scantling/training'
import type { CudaDevice } from './cuda/device.ts'
import type { DeviceBuffer } from './device.ts'
import { GpuModel } from './model.ts'

export interface AdamWSettings {
	readonly beta1?: number
	readonly beta2?: number
	readonly epsilon?: number
	readonly weightDecay?: number
}

export interface GpuTrainerOptions extends Omit<TrainerOptions, 'model' | 'accumulationSteps'> {
	readonly model: Transformer
	readonly device: CudaDevice
	/** Sequences per micro-batch, which also sets the activation memory. */
	readonly microBatch: number
	readonly length: number
	/** Micro-batches averaged into one optimizer step. Defaults to 1. */
	readonly accumulation?: number
	/** Keep weights and AdamW moments on the device and update them there. Defaults to true for AdamW. */
	readonly resident?: boolean
	/** Must match the CPU optimizer's settings when `resident` is on. Defaults to its defaults. */
	readonly adamw?: AdamWSettings
}

/** Where the time of one GPU step went, in milliseconds. */
export interface GpuStepTimings {
	readonly uploadMs: number
	readonly computeMs: number
	readonly downloadMs: number
	readonly optimizerMs: number
}

interface Slot {
	readonly name: string
	readonly weights: DeviceBuffer
	readonly grads: DeviceBuffer
	readonly length: number
	readonly rank: number
	readonly m: DeviceBuffer
	readonly v: DeviceBuffer
}

/**
 * Trains on the GPU. In resident mode the whole step stays on the device: gradient norm, clipping and AdamW run
 * there, and the CPU weights and optimizer state are refreshed only by `syncToCpu` (checkpoints, end of a run).
 * Otherwise gradients are copied back and the shared `Trainer` applies the optimizer on the CPU, as before.
 */
export class GpuTrainer {
	readonly trainer: Trainer
	readonly gpu: GpuModel
	readonly accumulation: number
	readonly resident: boolean
	readonly #optimizer: TrainerOptions['optimizer']
	readonly #maxGradNorm: number | undefined
	readonly #adamw: Required<AdamWSettings>
	readonly #slots: Slot[] = []
	#norms: DeviceBuffer | undefined
	#count = 0
	#timings: GpuStepTimings | undefined

	constructor(options: GpuTrainerOptions) {
		this.accumulation = options.accumulation ?? 1
		if (!Number.isInteger(this.accumulation) || this.accumulation < 1) throw new RangeError('accumulation must be a positive integer')
		this.#optimizer = options.optimizer
		this.#maxGradNorm = options.maxGradNorm
		this.resident = options.resident ?? options.optimizer.kind === 'adamw'
		this.#adamw = { beta1: options.adamw?.beta1 ?? 0.9, beta2: options.adamw?.beta2 ?? 0.95, epsilon: options.adamw?.epsilon ?? 1e-8, weightDecay: options.adamw?.weightDecay ?? 0.1 }
		this.trainer = new Trainer({ model: options.model, optimizer: options.optimizer, schedule: options.schedule, ...(options.maxGradNorm === undefined ? {} : { maxGradNorm: options.maxGradNorm }) })
		this.gpu = new GpuModel(options.device, options.model, { batch: options.microBatch, length: options.length })
		if (this.resident) {
			const { device } = this.gpu
			for (const buffers of this.gpu.parameterBuffers()) {
				const m = device.alloc(buffers.length)
				const v = device.alloc(buffers.length)
				device.fill(m, 0)
				device.fill(v, 0)
				this.#slots.push({ ...buffers, m, v })
			}
			this.#norms = device.alloc(this.#slots.length)
		}
		this.gpu.uploadWeights()
	}

	get step(): number {
		return this.trainer.step
	}

	get skippedSteps(): number {
		return this.trainer.skippedSteps
	}

	get lastTimings(): GpuStepTimings | undefined {
		return this.#timings
	}

	resume(step: number, skippedSteps: number): void {
		this.trainer.resume(step, skippedSteps)
	}

	/** Copies the CPU weights and the CPU optimizer's state onto the device. Call after loading a checkpoint. */
	uploadFromCpu(): void {
		this.gpu.uploadWeights()
		if (!this.resident) return
		const state = this.#optimizer.state()
		this.#count = state.step
		const { device } = this.gpu
		for (const slot of this.#slots) {
			for (const [prefix, buffer] of [['m', slot.m], ['v', slot.v]] as const) {
				const saved = state.tensors.get(`${prefix}:${slot.name}`)
				if (saved === undefined) device.fill(buffer, 0)
				else device.upload(buffer, saved.data)
			}
		}
	}

	/** Copies the device weights and AdamW moments back to the CPU model and optimizer. Call before saving a checkpoint. */
	syncToCpu(): void {
		if (!this.resident) return
		this.gpu.downloadWeights()
		const tensors = new Map<string, Tensor>()
		for (const slot of this.#slots) {
			for (const [prefix, buffer] of [['m', slot.m], ['v', slot.v]] as const) {
				const data = new Float32Array(slot.length)
				this.gpu.device.download(buffer, data)
				tensors.set(`${prefix}:${slot.name}`, Tensor.from([slot.length], data))
			}
		}
		this.#optimizer.loadState({ step: this.#count, tensors })
	}

	trainStep(source: BatchSource): StepMetrics {
		return this.resident ? this.#residentStep(source) : this.#hostStep(source)
	}

	/** Mean loss over `batches`, computed on the GPU with the device weights. */
	evaluate(batches: Iterable<Batch>): EvalResult {
		if (!this.resident) this.gpu.uploadWeights()
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

	async run(options: Omit<RunOptions, 'stepper' | 'evaluator'>): Promise<RunSummary> {
		const summary = await this.trainer.run({ ...options, stepper: (source) => this.trainStep(source), evaluator: (batches) => this.evaluate(batches) })
		this.syncToCpu()
		return summary
	}

	async close(): Promise<void> {
		const { device } = this.gpu
		for (const slot of this.#slots) {
			device.free(slot.m)
			device.free(slot.v)
		}
		if (this.#norms !== undefined) device.free(this.#norms)
		this.#slots.length = 0
		this.gpu.dispose()
	}

	#residentStep(source: BatchSource): StepMetrics {
		const started = performance.now()
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
		const norms = this.#norms!
		this.#slots.forEach((slot, index) => this.gpu.neural.sumSquares(slot.grads, norms, slot.length, index))
		const sums = new Float32Array(this.#slots.length)
		this.gpu.device.download(norms, sums)
		const computed = performance.now()
		let total = 0
		for (const value of sums) total += value
		const gradNorm = Math.sqrt(total) / this.accumulation
		const skipped = !Number.isFinite(loss) || !Number.isFinite(gradNorm)
		const learningRate = this.trainer.scheduledRate()
		if (!skipped) {
			this.#count += 1
			const clip = this.#maxGradNorm !== undefined && gradNorm > this.#maxGradNorm ? this.#maxGradNorm / (gradNorm + 1e-6) : 1
			const gradScale = clip / this.accumulation
			for (const slot of this.#slots) {
				this.gpu.neural.adamw(slot.weights, slot.grads, slot.m, slot.v, slot.length, { step: this.#count, learningRate, decayed: slot.rank >= 2, gradScale, ...this.#adamw })
			}
		}
		const metrics = this.trainer.recordExternal({ loss, gradNorm, learningRate, tokens, started, skipped })
		this.#timings = { uploadMs: 0, computeMs: computed - started, downloadMs: 0, optimizerMs: performance.now() - computed }
		return metrics
	}

	#hostStep(source: BatchSource): StepMetrics {
		const started = performance.now()
		this.gpu.uploadWeights()
		this.gpu.zeroGrads()
		const uploaded = performance.now()
		let loss = 0
		let tokens = 0
		for (let micro = 0; micro < this.accumulation; micro += 1) {
			const batch = source.next()
			this.#check(batch)
			loss += this.gpu.forward(batch.inputs, batch.targets, batch.batch) / this.accumulation
			this.gpu.backward(batch.batch)
			tokens += batch.batch * batch.length
		}
		this.gpu.device.synchronize()
		const computed = performance.now()
		this.gpu.downloadGrads(1 / this.accumulation)
		const downloaded = performance.now()
		const metrics = this.trainer.applyAccumulated(loss, tokens, started)
		this.#timings = { uploadMs: uploaded - started, computeMs: computed - uploaded, downloadMs: downloaded - computed, optimizerMs: performance.now() - downloaded }
		return metrics
	}

	#check(batch: Batch): void {
		if (batch.length !== this.gpu.length || batch.batch > this.gpu.capacity) throw new RangeError(`The GPU model was built for up to ${this.gpu.capacity} sequences of ${this.gpu.length} tokens, received ${batch.batch} of ${batch.length}`)
	}
}
