import type { KernelKind } from '@scantling/tensor'

/** A contiguous block of output rows of one matrix in the batch. */
export interface MatmulPiece {
	readonly ao: number
	readonly bo: number
	readonly co: number
	readonly i0: number
	readonly i1: number
}

/** A unit of work sent to a matmul worker. The buffers are shared, not copied. */
export interface MatmulJob {
	readonly kind: KernelKind
	readonly a: SharedArrayBuffer
	readonly b: SharedArrayBuffer
	readonly c: SharedArrayBuffer
	readonly m: number
	readonly k: number
	readonly n: number
	readonly pieces: readonly MatmulPiece[]
}

export type MatmulReply = { readonly ok: true } | { readonly ok: false; readonly error: string }
