import * as T from '@scantling/tensor'
import type { Variable } from './graph.ts'
import { record } from './graph.ts'

const when = (input: Variable, compute: () => T.Tensor): T.Tensor | null => (input.requiresGrad ? compute() : null)

function expand(t: T.Tensor, shape: T.Shape): T.Tensor {
	return T.zip(t, T.Tensor.zeros(shape), (x) => x)
}

function keptShape(shape: T.Shape, axes: number | readonly number[] | undefined): number[] {
	if (axes === undefined) return shape.map(() => 1)
	const resolved = (typeof axes === 'number' ? [axes] : axes).map((axis) => T.normalizeAxis(axis, shape.length))
	return shape.map((dimension, index) => (resolved.includes(index) ? 1 : dimension))
}

export function add(a: Variable, b: Variable): Variable {
	return record(T.add(a.value, b.value), [a, b], (g) => [when(a, () => T.sumToShape(g, a.shape)), when(b, () => T.sumToShape(g, b.shape))], 'add')
}

export function sub(a: Variable, b: Variable): Variable {
	return record(T.sub(a.value, b.value), [a, b], (g) => [when(a, () => T.sumToShape(g, a.shape)), when(b, () => T.sumToShape(T.neg(g), b.shape))], 'sub')
}

export function mul(a: Variable, b: Variable): Variable {
	return record(T.mul(a.value, b.value), [a, b], (g) => [when(a, () => T.sumToShape(T.mul(g, b.value), a.shape)), when(b, () => T.sumToShape(T.mul(g, a.value), b.shape))], 'mul')
}

export function div(a: Variable, b: Variable): Variable {
	return record(T.div(a.value, b.value), [a, b], (g) => [when(a, () => T.sumToShape(T.div(g, b.value), a.shape)), when(b, () => T.sumToShape(T.neg(T.div(T.mul(g, a.value), T.square(b.value))), b.shape))], 'div')
}

export function neg(a: Variable): Variable {
	return record(T.neg(a.value), [a], (g) => [T.neg(g)], 'neg')
}

export function addScalar(a: Variable, scalar: number): Variable {
	return record(T.addScalar(a.value, scalar), [a], (g) => [g], 'addScalar')
}

export function mulScalar(a: Variable, scalar: number): Variable {
	return record(T.mulScalar(a.value, scalar), [a], (g) => [T.mulScalar(g, scalar)], 'mulScalar')
}

export function exp(a: Variable): Variable {
	const out = T.exp(a.value)
	return record(out, [a], (g) => [T.mul(g, out)], 'exp')
}

export function log(a: Variable): Variable {
	return record(T.log(a.value), [a], (g) => [T.div(g, a.value)], 'log')
}

export function sqrt(a: Variable): Variable {
	const out = T.sqrt(a.value)
	return record(out, [a], (g) => [T.div(g, T.mulScalar(out, 2))], 'sqrt')
}

/**
 * Batched matrix product; see `matmul` in the tensor package for shapes and the transpose flags.
 * Gradients for operands that were broadcast across a batch are summed back to their own shape.
 */
export function matmul(a: Variable, b: Variable, options: T.MatmulOptions = {}): Variable {
	const transposeA = options.transposeA === true
	const transposeB = options.transposeB === true
	const gradients = (g: T.Tensor): [T.Tensor | null, T.Tensor | null] => {
		let ga: () => T.Tensor
		let gb: () => T.Tensor
		if (!transposeA && !transposeB) {
			ga = () => T.matmul(g, b.value, { transposeB: true })
			gb = () => T.matmul(a.value, g, { transposeA: true })
		} else if (!transposeA && transposeB) {
			ga = () => T.matmul(g, b.value)
			gb = () => T.matmul(g, a.value, { transposeA: true })
		} else if (transposeA && !transposeB) {
			ga = () => T.matmul(b.value, g, { transposeB: true })
			gb = () => T.matmul(a.value, g)
		} else {
			ga = () => T.matmul(b.value, g, { transposeA: true, transposeB: true })
			gb = () => T.matmul(g, a.value, { transposeA: true, transposeB: true })
		}
		return [when(a, () => T.sumToShape(ga(), a.shape)), when(b, () => T.sumToShape(gb(), b.shape))]
	}
	return record(T.matmul(a.value, b.value, options), [a, b], gradients, 'matmul')
}

export interface SumOptions {
	readonly axes?: number | readonly number[]
	readonly keepDims?: boolean
}

/** Sum over the given axes, or every axis when none are given. */
export function sum(a: Variable, options: SumOptions = {}): Variable {
	const kept = keptShape(a.shape, options.axes)
	return record(T.sum(a.value, options), [a], (g) => [expand(g.reshape(kept), a.shape)], 'sum')
}

/** Mean over the given axes, or every axis when none are given. */
export function mean(a: Variable, options: SumOptions = {}): Variable {
	const kept = keptShape(a.shape, options.axes)
	const count = a.value.size / kept.reduce((product, dimension) => product * dimension, 1)
	return record(T.mean(a.value, options), [a], (g) => [T.mulScalar(expand(g.reshape(kept), a.shape), 1 / count)], 'mean')
}

/** Reshapes to `shape` (one dimension may be -1). The result shares memory with the input. */
export function reshape(a: Variable, shape: T.Shape): Variable {
	return record(a.value.reshape(shape), [a], (g) => [g.reshape(a.shape)], 'reshape')
}

/** Reorders axes; `axes[i]` is the input axis that becomes output axis `i`. */
export function permute(a: Variable, axes: readonly number[]): Variable {
	const resolved = axes.map((axis) => T.normalizeAxis(axis, a.shape.length))
	const inverse = new Array<number>(resolved.length).fill(0)
	resolved.forEach((axis, position) => { inverse[axis] = position })
	return record(T.permute(a.value, resolved), [a], (g) => [T.permute(g, inverse)], 'permute')
}

/** Swaps two axes (the last two by default). */
export function transpose(a: Variable, axis0 = -2, axis1 = -1): Variable {
	return record(T.transpose(a.value, axis0, axis1), [a], (g) => [T.transpose(g, axis0, axis1)], 'transpose')
}
