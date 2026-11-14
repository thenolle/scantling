import type { RandomState, Tensor } from '@scantling/tensor'

/** Everything needed to resume a training run exactly. */
export interface CheckpointState {
	readonly step: number
	readonly skippedSteps: number
	/** The model configuration, as plain JSON. */
	readonly config: unknown
	readonly weights: ReadonlyMap<string, Tensor>
	readonly optimizer: { readonly kind: string; readonly step: number; readonly tensors: ReadonlyMap<string, Tensor> }
	/** Named random generators, such as the batch sampler. */
	readonly random: Readonly<Record<string, RandomState>>
	readonly metrics: Readonly<Record<string, number>>
	readonly notes?: string
}

export interface TensorEntry {
	readonly group: 'weights' | 'optimizer'
	readonly name: string
	readonly shape: readonly number[]
	readonly offset: number
	readonly bytes: number
	readonly sha256: string
}

export interface CheckpointMeta {
	readonly format: 'scantling-checkpoint'
	readonly version: 1
	readonly createdAt: string
	readonly step: number
	readonly skippedSteps: number
	readonly config: unknown
	readonly tensors: readonly TensorEntry[]
	readonly optimizer: { readonly kind: string; readonly step: number }
	readonly random: Readonly<Record<string, RandomState>>
	readonly metrics: Readonly<Record<string, number>>
	readonly notes: string
	/** SHA-256 of the whole `tensors.bin`. */
	readonly sha256: string
}

export class CorruptCheckpointError extends Error {
	constructor(message: string) {
		super(message)
		this.name = 'CorruptCheckpointError'
	}
}
