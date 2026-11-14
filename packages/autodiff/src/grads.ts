import type { Variable } from './graph.ts'

/** Clears the gradient of every parameter. */
export function zeroGrad(params: readonly Variable[]): void {
	for (const param of params) param.zeroGrad()
}

/**
 * The Euclidean norm of all gradients taken together, accumulated in 64-bit floats.
 * Parameters without a gradient contribute nothing.
 */
export function globalGradNorm(params: readonly Variable[]): number {
	let total = 0
	for (const param of params) {
		const data = param.grad?.data
		if (data === undefined) continue
		for (let index = 0; index < data.length; index += 1) total += data[index]! * data[index]!
	}
	return Math.sqrt(total)
}

/**
 * Scales all gradients so their global norm is at most `maxNorm`, and returns the norm measured
 * before clipping. A non-finite norm is returned unchanged and nothing is scaled, so the caller
 * can skip the optimizer step instead of spreading NaN into the weights.
 */
export function clipGradNorm(params: readonly Variable[], maxNorm: number): number {
	const norm = globalGradNorm(params)
	if (!Number.isFinite(norm) || norm <= maxNorm) return norm
	const scale = maxNorm / norm
	for (const param of params) {
		const data = param.grad?.data
		if (data === undefined) continue
		for (let index = 0; index < data.length; index += 1) data[index] = data[index]! * scale
	}
	return norm
}
