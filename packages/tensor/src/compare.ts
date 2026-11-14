import { shapesEqual } from './shape.ts'
import type { Tensor } from './tensor.ts'

export interface CloseOptions {
	readonly rtol?: number
	readonly atol?: number
}

/** Largest absolute element-wise difference. Throws when shapes differ. */
export function maxAbsDiff(a: Tensor, b: Tensor): number {
	if (!shapesEqual(a.shape, b.shape)) throw new RangeError(`Shapes [${a.shape.join(', ')}] and [${b.shape.join(', ')}] differ`)
	let worst = 0
	for (let index = 0; index < a.size; index += 1) {
		const difference = Math.abs(a.data[index]! - b.data[index]!)
		if (difference > worst || difference !== difference) worst = difference
	}
	return worst
}

/**
 * True when every pair satisfies `|a - b| <= atol + rtol * |b|` (NumPy semantics).
 * Different shapes and NaNs are never close.
 */
export function allClose(a: Tensor, b: Tensor, options: CloseOptions = {}): boolean {
	if (!shapesEqual(a.shape, b.shape)) return false
	const rtol = options.rtol ?? 1e-5
	const atol = options.atol ?? 1e-8
	for (let index = 0; index < a.size; index += 1) {
		const left = a.data[index]!
		const right = b.data[index]!
		if (!(Math.abs(left - right) <= atol + rtol * Math.abs(right))) return false
	}
	return true
}

/** Index of the first NaN or infinity, or -1 when every element is finite. */
export function firstNonFinite(t: Tensor): number {
	for (let index = 0; index < t.size; index += 1) {
		if (!Number.isFinite(t.data[index]!)) return index
	}
	return -1
}

/** Throws when a tensor contains NaN or infinity, naming the first offending position. */
export function assertFinite(t: Tensor, label: string): void {
	const index = firstNonFinite(t)
	if (index !== -1) throw new RangeError(`${label} contains a non-finite value (${t.data[index]}) at flat index ${index}`)
}
