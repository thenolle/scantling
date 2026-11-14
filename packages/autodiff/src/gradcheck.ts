import * as T from '@scantling/tensor'
import type { Variable } from './graph.ts'
import { backward, noGrad, parameter, variable } from './graph.ts'

export interface GradcheckOptions {
	/** Finite-difference step. The default 1e-2 balances truncation against Float32 rounding. */
	readonly epsilon?: number
	readonly rtol?: number
	readonly atol?: number
	readonly seed?: number
}

export interface GradcheckResult {
	readonly ok: boolean
	readonly maxAbsError: number
	readonly maxRelError: number
	readonly checked: number
	/** The element with the largest error beyond tolerance, or the largest error overall. */
	readonly worst: { readonly input: number; readonly index: number; readonly analytic: number; readonly numeric: number } | undefined
}

/**
 * Compares the analytic gradient of `fn` with central finite differences for every element of
 * every input. A non-scalar output is reduced with fixed random weights, and the reduction is
 * summed in 64-bit floats, so a single check exercises every output element.
 */
export function gradcheck(fn: (...inputs: Variable[]) => Variable, inputs: readonly T.Tensor[], options: GradcheckOptions = {}): GradcheckResult {
	const epsilon = options.epsilon ?? 1e-2
	const rtol = options.rtol ?? 1e-2
	const atol = options.atol ?? 1e-3
	const params = inputs.map((input) => parameter(input.clone()))
	const output = fn(...params)
	const weights = T.randomNormal(output.shape, new T.Random(options.seed ?? 1))
	if (output.requiresGrad) backward(output, weights)
	const work = inputs.map((input) => input.clone())
	const objective = (): number => noGrad(() => {
		const result = fn(...work.map((tensor) => variable(tensor))).value.data
		let total = 0
		for (let index = 0; index < result.length; index += 1) total += result[index]! * weights.data[index]!
		return total
	})
	let maxAbsError = 0
	let maxRelError = 0
	let checked = 0
	let worst: GradcheckResult['worst']
	let worstScore = -1
	let failed = false
	work.forEach((tensor, input) => {
		const analytic = params[input]!.grad
		for (let index = 0; index < tensor.size; index += 1) {
			const original = tensor.data[index]!
			const plus = Math.fround(original + epsilon)
			const minus = Math.fround(original - epsilon)
			tensor.data[index] = plus
			const high = objective()
			tensor.data[index] = minus
			const low = objective()
			tensor.data[index] = original
			const numeric = (high - low) / (plus - minus)
			const exact = analytic?.data[index] ?? 0
			const error = Math.abs(exact - numeric)
			const scale = Math.max(Math.abs(exact), Math.abs(numeric))
			const within = error <= atol + rtol * scale
			maxAbsError = Math.max(maxAbsError, error)
			maxRelError = Math.max(maxRelError, scale > 0 ? error / scale : 0)
			checked += 1
			if (!within) failed = true
			const score = within ? error / (atol + rtol * scale) : 1e9 + error
			if (score > worstScore) {
				worstScore = score
				worst = { input, index, analytic: exact, numeric }
			}
		}
	})
	return { ok: !failed, maxAbsError, maxRelError, checked, worst }
}

/** A readable summary for test failure messages. */
export function describeGradcheck(result: GradcheckResult): string {
	const worst = result.worst
	return worst === undefined ? 'no elements were checked' : `max abs error ${result.maxAbsError.toExponential(2)}, worst at input ${worst.input}[${worst.index}]: analytic ${worst.analytic} vs numeric ${worst.numeric}`
}
