import { backward, noGrad, zeroGrad } from '@scantling/autodiff'
import { Random } from '@scantling/tensor'
import type { Transformer } from './transformer.ts'

export interface ModelBatch {
	readonly inputs: ArrayLike<number>
	readonly targets: ArrayLike<number>
	readonly batch: number
	readonly length: number
}

export interface ModelGradientOptions {
	readonly samples?: number
	readonly epsilon?: number
	readonly rtol?: number
	readonly atol?: number
	readonly seed?: number
}

export interface ModelGradientReport {
	readonly ok: boolean
	readonly samples: number
	readonly maxAbsError: number
	readonly worst: { readonly parameter: string; readonly index: number; readonly analytic: number; readonly numeric: number } | undefined
}

/**
 * Validates backpropagation through the whole model. It runs one backward pass, then perturbs
 * randomly chosen elements across all parameters and compares central finite differences of the
 * loss with the analytic gradients. Gradients are left populated on the parameters.
 *
 * Float32 rounding limits accuracy, so run it on a small model whose weights are not tiny
 * (for example `initStd` 0.3), where gradients are much larger than the rounding noise.
 */
export function checkModelGradients(model: Transformer, batch: ModelBatch, options: ModelGradientOptions = {}): ModelGradientReport {
	const samples = options.samples ?? 40
	const epsilon = options.epsilon ?? 5e-3
	const rtol = options.rtol ?? 2e-2
	const atol = options.atol ?? 1e-3
	const named = model.namedParameters()
	zeroGrad(named.map((entry) => entry.variable))
	backward(model.loss(batch.inputs, batch.targets, batch.batch, batch.length))
	const total = named.reduce((sum, entry) => sum + entry.variable.value.size, 0)
	const random = new Random(options.seed ?? 1)
	const objective = (): number => noGrad(() => model.loss(batch.inputs, batch.targets, batch.batch, batch.length).value.item())
	let maxAbsError = 0
	let failed = false
	let worst: ModelGradientReport['worst']
	let worstScore = -1
	for (let sample = 0; sample < samples; sample += 1) {
		let flat = random.nextInt(total)
		const entry = named.find((candidate) => {
			if (flat < candidate.variable.value.size) return true
			flat -= candidate.variable.value.size
			return false
		})!
		const data = entry.variable.value.data
		const original = data[flat]!
		const plus = Math.fround(original + epsilon)
		const minus = Math.fround(original - epsilon)
		data[flat] = plus
		const high = objective()
		data[flat] = minus
		const low = objective()
		data[flat] = original
		const numeric = (high - low) / (plus - minus)
		const analytic = entry.variable.grad?.data[flat] ?? 0
		const error = Math.abs(analytic - numeric)
		const tolerance = atol + rtol * Math.max(Math.abs(analytic), Math.abs(numeric))
		maxAbsError = Math.max(maxAbsError, error)
		if (error > tolerance) failed = true
		const score = error / tolerance
		if (score > worstScore) {
			worstScore = score
			worst = { parameter: entry.name, index: flat, analytic, numeric }
		}
	}
	return { ok: !failed, samples, maxAbsError, worst }
}
