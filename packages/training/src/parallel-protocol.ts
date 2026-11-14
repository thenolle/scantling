import type { ModelConfigInput } from '@scantling/nn'

/** One micro-batch of work for a worker. The weights and gradient buffers are shared, not copied. */
export interface GradientJob {
	readonly config: ModelConfigInput
	readonly weights: SharedArrayBuffer
	/** `workers` slots of `parameterCount` floats; this job writes slot `slot`. */
	readonly grads: SharedArrayBuffer
	readonly slot: number
	readonly parameterCount: number
	readonly inputs: Int32Array
	readonly targets: Int32Array
	readonly batch: number
	readonly length: number
	/** Seed for the backward pass: 1 divided by the number of micro-batches averaged in the step. */
	readonly scale: number
}

export type GradientReply = { readonly ok: true; readonly loss: number } | { readonly ok: false; readonly error: string }
