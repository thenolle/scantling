import * as T from '@scantling/tensor'
import type { Variable } from './graph.ts'
import { record } from './graph.ts'

/**
 * Root-mean-square normalisation over the last axis: `y = x / sqrt(mean(x^2) + eps) * weight`.
 * Statistics are accumulated in 64-bit floats and only one scalar per row is kept for the backward pass.
 */
export function rmsNorm(x: Variable, weight: Variable, eps = 1e-5): Variable {
	const width = x.shape[x.shape.length - 1]
	if (width === undefined || weight.shape.length !== 1 || weight.shape[0] !== width) throw new RangeError(`rmsNorm needs a weight of shape [${width}], received [${weight.shape.join(', ')}]`)
	const rows = x.value.size / width
	const inverse = new Float64Array(rows)
	const out = new Float32Array(x.value.size)
	const input = x.value.data
	const scale = weight.value.data
	for (let row = 0; row < rows; row += 1) {
		const base = row * width
		let total = 0
		for (let column = 0; column < width; column += 1) total += input[base + column]! * input[base + column]!
		const r = 1 / Math.sqrt(total / width + eps)
		inverse[row] = r
		for (let column = 0; column < width; column += 1) out[base + column] = input[base + column]! * r * scale[column]!
	}
	return record(new T.Tensor(x.shape, out), [x, weight], (g) => {
		const upstream = g.data
		const dx = new Float32Array(x.value.size)
		const dw = new Float64Array(width)
		for (let row = 0; row < rows; row += 1) {
			const base = row * width
			const r = inverse[row]!
			let dot = 0
			for (let column = 0; column < width; column += 1) {
				const normalised = input[base + column]! * r
				dw[column] = dw[column]! + upstream[base + column]! * normalised
				dot += upstream[base + column]! * scale[column]! * normalised
			}
			dot /= width
			for (let column = 0; column < width; column += 1) {
				const normalised = input[base + column]! * r
				dx[base + column] = r * (upstream[base + column]! * scale[column]! - normalised * dot)
			}
		}
		return [x.requiresGrad ? new T.Tensor(x.shape, dx) : null, weight.requiresGrad ? T.Tensor.from(weight.shape, dw) : null]
	}, 'rmsNorm')
}
