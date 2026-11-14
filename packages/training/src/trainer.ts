import type { Variable } from '@scantling/autodiff'
import { backward, clipGradNorm, globalGradNorm, noGrad, zeroGrad } from '@scantling/autodiff'
import { Tensor } from '@scantling/tensor'
import type { Batch, BatchSource } from './batches.ts'
import type { Optimizer, ParamEntry } from './optimizers.ts'
import type { LearningRateSchedule } from './schedule.ts'

export interface TrainableModel {
	loss(inputs: ArrayLike<number>, targets: ArrayLike<number>, batch: number, length: number): Variable
	namedParameters(): ParamEntry[]
}

export interface TrainerOptions {
	readonly model: TrainableModel
	readonly optimizer: Optimizer
	readonly schedule: LearningRateSchedule
	/** Micro-batches whose gradients are averaged into one optimizer step. Defaults to 1. */
	readonly accumulationSteps?: number
	/** Gradients are scaled down to this global norm. Leave unset to disable clipping. */
	readonly maxGradNorm?: number
}

export interface StepMetrics {
	readonly step: number
	readonly loss: number
	/** The global gradient norm before clipping. */
	readonly gradNorm: number
	readonly learningRate: number
	readonly tokens: number
	readonly milliseconds: number
	readonly tokensPerSecond: number
	/** True when the loss or gradient was not finite, so the weights were left untouched. */
	readonly skipped: boolean
}

export interface EvalResult {
	readonly loss: number
	readonly perplexity: number
	readonly tokens: number
}

export interface RunOptions {
	readonly steps: number
	readonly source: BatchSource
	readonly evalEvery?: number
	readonly evalBatches?: () => Iterable<Batch>
	readonly onStep?: (metrics: StepMetrics) => void
	readonly onEval?: (step: number, result: EvalResult) => void
	/** Aborting stops the run cleanly after the current step. */
	readonly signal?: AbortSignal
	/** Replaces the default single-threaded step, for example with a parallel one. */
	readonly stepper?: (source: BatchSource) => StepMetrics | Promise<StepMetrics>
	/** Replaces `evaluate`, for example with an evaluation on the GPU. */
	readonly evaluator?: (batches: Iterable<Batch>) => EvalResult
}

export interface RunSummary {
	readonly history: StepMetrics[]
	readonly evaluations: { readonly step: number; readonly result: EvalResult }[]
	readonly stopped: boolean
}

/**
 * Runs optimizer steps. Each step averages the gradients of `accumulationSteps` micro-batches
 * (by seeding every backward pass with `1 / accumulationSteps`), clips the global gradient norm, and
 * applies the optimizer with the scheduled learning rate. A non-finite loss or gradient skips the step.
 */
export class Trainer {
	readonly #model: TrainableModel
	readonly #optimizer: Optimizer
	readonly #schedule: LearningRateSchedule
	readonly #accumulation: number
	readonly #maxGradNorm: number | undefined
	readonly #params: ParamEntry[]
	#step = 0
	#skipped = 0

	constructor(options: TrainerOptions) {
		this.#model = options.model
		this.#optimizer = options.optimizer
		this.#schedule = options.schedule
		this.#accumulation = options.accumulationSteps ?? 1
		this.#maxGradNorm = options.maxGradNorm
		if (!Number.isInteger(this.#accumulation) || this.#accumulation < 1) throw new RangeError('accumulationSteps must be a positive integer')
		this.#params = this.#model.namedParameters()
	}

	/** Optimizer steps taken so far, including skipped ones. */
	get step(): number {
		return this.#step
	}

	get skippedSteps(): number {
		return this.#skipped
	}

	/** Restores the step counters when resuming from a checkpoint. */
	resume(step: number, skippedSteps: number): void {
		this.#step = step
		this.#skipped = skippedSteps
	}

	trainStep(source: BatchSource): StepMetrics {
		const started = performance.now()
		zeroGrad(this.#variables())
		const seed = Tensor.scalar(1 / this.#accumulation)
		let loss = 0
		let tokens = 0
		for (let micro = 0; micro < this.#accumulation; micro += 1) {
			const batch = source.next()
			const value = this.#model.loss(batch.inputs, batch.targets, batch.batch, batch.length)
			loss += value.value.item() / this.#accumulation
			tokens += batch.batch * batch.length
			backward(value, seed)
		}
		return this.applyAccumulated(loss, tokens, started)
	}

	/**
	 * Finishes a step whose averaged gradients are already stored on the parameters: clips the global
	 * norm, skips the update when the loss or gradient is not finite, otherwise applies the optimizer with
	 * the scheduled learning rate, then clears the gradients and advances the step counter.
	 */
	applyAccumulated(loss: number, tokens: number, started: number): StepMetrics {
		const variables = this.#variables()
		const gradNorm = this.#maxGradNorm === undefined ? globalGradNorm(variables) : clipGradNorm(variables, this.#maxGradNorm)
		const learningRate = this.#schedule(this.#step)
		const skipped = !Number.isFinite(loss) || !Number.isFinite(gradNorm)
		if (skipped) this.#skipped += 1
		else this.#optimizer.step(this.#params, learningRate)
		zeroGrad(variables)
		const step = this.#step
		this.#step += 1
		const milliseconds = performance.now() - started
		return { step, loss, gradNorm, learningRate, tokens, milliseconds, tokensPerSecond: tokens / (milliseconds / 1000), skipped }
	}

	/** Mean loss over `batches` without recording a graph. */
	evaluate(batches: Iterable<Batch>): EvalResult {
		let total = 0
		let tokens = 0
		for (const batch of batches) {
			const count = batch.batch * batch.length
			total += noGrad(() => this.#model.loss(batch.inputs, batch.targets, batch.batch, batch.length)).value.item() * count
			tokens += count
		}
		const loss = tokens === 0 ? Number.NaN : total / tokens
		return { loss, perplexity: Math.exp(loss), tokens }
	}

	async run(options: RunOptions): Promise<RunSummary> {
		const history: StepMetrics[] = []
		const evaluations: { step: number; result: EvalResult }[] = []
		const stepper = options.stepper ?? ((source: BatchSource): StepMetrics => this.trainStep(source))
		let stopped = false
		for (let index = 0; index < options.steps; index += 1) {
			if (options.signal?.aborted === true) {
				stopped = true
				break
			}
			const metrics = await stepper(options.source)
			history.push(metrics)
			options.onStep?.(metrics)
			if (options.evalEvery !== undefined && options.evalBatches !== undefined && this.#step % options.evalEvery === 0) {
				const result = (options.evaluator ?? ((batches: Iterable<Batch>) => this.evaluate(batches)))(options.evalBatches())
				evaluations.push({ step: this.#step, result })
				options.onEval?.(this.#step, result)
			}
			await new Promise<void>((resolve) => setTimeout(resolve, 0))
		}
		return { history, evaluations, stopped }
	}

	#variables(): Variable[] {
		return this.#params.map((entry) => entry.variable)
	}
}
