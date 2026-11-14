import * as T from '@scantling/tensor'
import type { Variable } from './graph.ts'
import { record } from './graph.ts'

const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI)

function pointwise(a: Variable, forward: (t: T.Tensor) => T.Tensor, derivative: (x: number, y: number) => number, label: string): Variable {
	const out = forward(a.value)
	return record(out, [a], (g) => {
		const grad = new Float32Array(g.size)
		const x = a.value.data
		const y = out.data
		const upstream = g.data
		for (let index = 0; index < grad.length; index += 1) grad[index] = upstream[index]! * derivative(x[index]!, y[index]!)
		return [new T.Tensor(g.shape, grad)]
	}, label)
}

export const relu = (a: Variable): Variable => pointwise(a, T.relu, (x) => (x > 0 ? 1 : 0), 'relu')

export const sigmoid = (a: Variable): Variable => pointwise(a, T.sigmoid, (_x, y) => y * (1 - y), 'sigmoid')

export const tanh = (a: Variable): Variable => pointwise(a, T.tanh, (_x, y) => 1 - y * y, 'tanh')

/** `x * sigmoid(x)`. */
export const silu = (a: Variable): Variable => pointwise(a, T.silu, (x) => {
	const s = 1 / (1 + Math.exp(-x))
	return s * (1 + x * (1 - s))
}, 'silu')

/** Tanh approximation of the Gaussian error linear unit. */
export const gelu = (a: Variable): Variable => pointwise(a, T.gelu, (x) => {
	const t = Math.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x))
	return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * SQRT_2_OVER_PI * (1 + 3 * 0.044715 * x * x)
}, 'gelu')
