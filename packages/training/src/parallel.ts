import type { ModelConfigInput } from '@scantling/nn'
import { Tensor } from '@scantling/tensor'
import { WorkerPool } from '@scantling/workers'
import type { Batch, BatchSource } from './batches.ts'
import type { GradientJob, GradientReply } from './parallel-protocol.ts'
import type { ParamEntry } from './optimizers.ts'
import type { EvalResult, RunOptions, RunSummary, StepMetrics, TrainableModel, TrainerOptions } from './trainer.ts'
import { Trainer } from './trainer.ts'

export interface ParallelModel extends TrainableModel {
	/** The configuration used to build identical copies in the workers. */
	readonly config: ModelConfigInput
}

export interface ParallelTrainerOptions extends Omit<TrainerOptions, 'model' | 'accumulationSteps'> {
	readonly model: ParallelModel
	/** Worker threads, each processing one micro-batch per round. */
	readonly workers: number
	/** Rounds per step. One step averages `workers * rounds` micro-batches. Defaults to 1. */
	readonly rounds?: number
	/** Refuse to start when the estimate (excluding activations) exceeds this many bytes. */
	readonly maxMemoryBytes?: number
}

/**
 * A lower bound on the memory of data-parallel training in bytes, excluding activations: master weights,
 * shared weights, gradient accumulator and two optimizer moments (five copies of the parameters), plus per
 * worker one gradient slot and a local copy of weights and gradients.
 */
export function estimateParallelMemory(parameterCount: number, workers: number): number {
	return 4 * parameterCount * (5 + 3 * workers)
}

/**
 * Data-parallel training. Every step publishes the master weights to shared memory; each worker runs forward
 * and backward on its own micro-batch and writes its gradient into its own slot; the slots are summed in
 * slot order and the shared `Trainer` logic clips, skips on non-finite values and applies the optimizer.
 * Micro-batches are drawn in a fixed order, so a run is reproducible for a given worker count and matches
 * single-threaded gradient accumulation up to float rounding.
 */
export class ParallelTrainer {
	readonly trainer: Trainer
	readonly workers: number
	readonly rounds: number
	readonly #pool: WorkerPool
	readonly #model: ParallelModel
	readonly #params: ParamEntry[]
	readonly #count: number
	readonly #weights: SharedArrayBuffer
	readonly #slots: SharedArrayBuffer
	readonly #accumulated: Float32Array
	#closed = false

	constructor(options: ParallelTrainerOptions) {
		this.workers = options.workers
		this.rounds = options.rounds ?? 1
		if (!Number.isInteger(this.workers) || this.workers < 1) throw new RangeError('workers must be a positive integer')
		if (!Number.isInteger(this.rounds) || this.rounds < 1) throw new RangeError('rounds must be a positive integer')
		this.#model = options.model
		this.#params = options.model.namedParameters()
		this.#count = this.#params.reduce((total, entry) => total + entry.variable.value.size, 0)
		const estimate = estimateParallelMemory(this.#count, this.workers)
		if (options.maxMemoryBytes !== undefined && estimate > options.maxMemoryBytes) throw new RangeError(`Parallel training needs at least ${estimate} bytes but only ${options.maxMemoryBytes} are allowed`)
		this.trainer = new Trainer({ model: options.model, optimizer: options.optimizer, schedule: options.schedule, ...(options.maxGradNorm === undefined ? {} : { maxGradNorm: options.maxGradNorm }) })
		this.#weights = new SharedArrayBuffer(this.#count * 4)
		this.#slots = new SharedArrayBuffer(this.#count * 4 * this.workers)
		this.#accumulated = new Float32Array(this.#count)
		this.#pool = new WorkerPool(this.workers, new URL('./parallel-worker.ts', import.meta.url))
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

	evaluate(batches: Iterable<Batch>): EvalResult {
		return this.trainer.evaluate(batches)
	}

	async trainStep(source: BatchSource): Promise<StepMetrics> {
		if (this.#closed) throw new Error('The parallel trainer is closed')
		const started = performance.now()
		this.#publish()
		this.#accumulated.fill(0)
		const slots = new Float32Array(this.#slots)
		const total = this.workers * this.rounds
		let loss = 0
		let tokens = 0
		for (let round = 0; round < this.rounds; round += 1) {
			const replies = await Promise.all(
				Array.from({ length: this.workers }, (_, slot) => {
					const batch = source.next()
					tokens += batch.batch * batch.length
					const job: GradientJob = {
						config: this.#model.config,
						weights: this.#weights,
						grads: this.#slots,
						slot,
						parameterCount: this.#count,
						inputs: Int32Array.from(batch.inputs),
						targets: Int32Array.from(batch.targets),
						batch: batch.batch,
						length: batch.length,
						scale: 1 / total
					}
					return this.#pool.run<GradientReply>(job)
				})
			)
			for (const reply of replies) {
				if (!reply.ok) throw new Error(`A training worker failed: ${reply.error}`)
				loss += reply.loss / total
			}
			for (let slot = 0; slot < this.workers; slot += 1) {
				const base = slot * this.#count
				for (let index = 0; index < this.#count; index += 1) this.#accumulated[index] = this.#accumulated[index]! + slots[base + index]!
			}
		}
		this.#scatter()
		return this.trainer.applyAccumulated(loss, tokens, started)
	}

	run(options: Omit<RunOptions, 'stepper'>): Promise<RunSummary> {
		return this.trainer.run({ ...options, stepper: (source) => this.trainStep(source) })
	}

	async close(): Promise<void> {
		this.#closed = true
		await this.#pool.close()
	}

	#publish(): void {
		const weights = new Float32Array(this.#weights)
		let offset = 0
		for (const { variable } of this.#params) {
			weights.set(variable.value.data, offset)
			offset += variable.value.size
		}
	}

	#scatter(): void {
		let offset = 0
		for (const { variable } of this.#params) {
			variable.grad = new Tensor(variable.value.shape, this.#accumulated.slice(offset, offset + variable.value.size))
			offset += variable.value.size
		}
	}
}
