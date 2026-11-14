import type { Variable } from '@scantling/autodiff'
import { Tensor } from '@scantling/tensor'

export interface ParamEntry {
	readonly name: string
	readonly variable: Variable
}

export interface OptimizerState {
	readonly step: number
	readonly tensors: ReadonlyMap<string, Tensor>
}

export interface Optimizer {
	readonly kind: string
	/** Applies one update using the gradients currently stored on the parameters. Parameters without a gradient are skipped. */
	step(params: readonly ParamEntry[], learningRate: number): void
	/** A copy of everything needed to resume training exactly. */
	state(): OptimizerState
	loadState(state: OptimizerState): void
}

function slot(buffers: Map<string, Float32Array>, name: string, length: number): Float32Array {
	let buffer = buffers.get(name)
	if (buffer === undefined) {
		buffer = new Float32Array(length)
		buffers.set(name, buffer)
	}
	if (buffer.length !== length) throw new RangeError(`Optimizer state for "${name}" has ${buffer.length} values but the parameter has ${length}`)
	return buffer
}

function exportBuffers(prefix: string, buffers: ReadonlyMap<string, Float32Array>, into: Map<string, Tensor>): void {
	for (const [name, buffer] of buffers) into.set(`${prefix}:${name}`, Tensor.from([buffer.length], buffer))
}

function importBuffers(prefix: string, state: OptimizerState): Map<string, Float32Array> {
	const buffers = new Map<string, Float32Array>()
	for (const [key, tensor] of state.tensors) {
		if (key.startsWith(`${prefix}:`)) buffers.set(key.slice(prefix.length + 1), tensor.data.slice())
	}
	return buffers
}

export interface AdamWOptions {
	readonly beta1?: number
	readonly beta2?: number
	readonly epsilon?: number
	/** Decoupled weight decay, applied only to parameters of rank 2 or more. */
	readonly weightDecay?: number
}

/**
 * AdamW. With bias-corrected moments `m^ = m / (1 - b1^t)` and `v^ = v / (1 - b2^t)`, each update is
 * `p <- p - lr * (m^ / (sqrt(v^) + eps) + weightDecay * p)`, where the decay term is skipped for vectors such as norm scales.
 */
export class AdamW implements Optimizer {
	readonly kind = 'adamw'
	readonly #beta1: number
	readonly #beta2: number
	readonly #epsilon: number
	readonly #weightDecay: number
	#first = new Map<string, Float32Array>()
	#second = new Map<string, Float32Array>()
	#count = 0

	constructor(options: AdamWOptions = {}) {
		this.#beta1 = options.beta1 ?? 0.9
		this.#beta2 = options.beta2 ?? 0.95
		this.#epsilon = options.epsilon ?? 1e-8
		this.#weightDecay = options.weightDecay ?? 0.1
	}

	step(params: readonly ParamEntry[], learningRate: number): void {
		this.#count += 1
		const correction1 = 1 - this.#beta1 ** this.#count
		const correction2 = 1 - this.#beta2 ** this.#count
		for (const { name, variable } of params) {
			const grad = variable.grad
			if (grad === null) continue
			const value = variable.value.data
			const m = slot(this.#first, name, value.length)
			const v = slot(this.#second, name, value.length)
			const decay = variable.value.rank >= 2 ? learningRate * this.#weightDecay : 0
			const g = grad.data
			for (let index = 0; index < value.length; index += 1) {
				const gradient = g[index]!
				const first = this.#beta1 * m[index]! + (1 - this.#beta1) * gradient
				const second = this.#beta2 * v[index]! + (1 - this.#beta2) * gradient * gradient
				m[index] = first
				v[index] = second
				const update = first / correction1 / (Math.sqrt(second / correction2) + this.#epsilon)
				value[index] = value[index]! - learningRate * update - decay * value[index]!
			}
		}
	}

	state(): OptimizerState {
		const tensors = new Map<string, Tensor>()
		exportBuffers('m', this.#first, tensors)
		exportBuffers('v', this.#second, tensors)
		return { step: this.#count, tensors }
	}

	loadState(state: OptimizerState): void {
		this.#count = state.step
		this.#first = importBuffers('m', state)
		this.#second = importBuffers('v', state)
	}
}

export interface SgdOptions {
	readonly momentum?: number
	/** Decoupled weight decay, applied only to parameters of rank 2 or more. */
	readonly weightDecay?: number
}

/** Stochastic gradient descent with optional momentum: `v <- mu * v + g`, `p <- p - lr * v`. */
export class Sgd implements Optimizer {
	readonly kind = 'sgd'
	readonly #momentum: number
	readonly #weightDecay: number
	#velocity = new Map<string, Float32Array>()
	#count = 0

	constructor(options: SgdOptions = {}) {
		this.#momentum = options.momentum ?? 0
		this.#weightDecay = options.weightDecay ?? 0
	}

	step(params: readonly ParamEntry[], learningRate: number): void {
		this.#count += 1
		for (const { name, variable } of params) {
			const grad = variable.grad
			if (grad === null) continue
			const value = variable.value.data
			const velocity = slot(this.#velocity, name, value.length)
			const decay = variable.value.rank >= 2 ? learningRate * this.#weightDecay : 0
			const g = grad.data
			for (let index = 0; index < value.length; index += 1) {
				const next = this.#momentum * velocity[index]! + g[index]!
				velocity[index] = next
				value[index] = value[index]! - learningRate * next - decay * value[index]!
			}
		}
	}

	state(): OptimizerState {
		const tensors = new Map<string, Tensor>()
		exportBuffers('velocity', this.#velocity, tensors)
		return { step: this.#count, tensors }
	}

	loadState(state: OptimizerState): void {
		this.#count = state.step
		this.#velocity = importBuffers('velocity', state)
	}
}
