import type { Tensor } from '@scantling/tensor'

export interface NamedVariable {
	readonly name: string
	readonly variable: { readonly value: Tensor }
}

/** The tensors of a model by parameter name. They are referenced, not copied. */
export function weightsOf(entries: readonly NamedVariable[]): Map<string, Tensor> {
	return new Map(entries.map((entry) => [entry.name, entry.variable.value]))
}

/**
 * Copies saved weights into a model's existing tensors, in place. Throws when the names or shapes of
 * the checkpoint and the model do not match exactly.
 */
export function applyWeights(entries: readonly NamedVariable[], weights: ReadonlyMap<string, Tensor>): void {
	if (entries.length !== weights.size) throw new RangeError(`The model has ${entries.length} parameters but the checkpoint holds ${weights.size}`)
	for (const { name, variable } of entries) {
		const saved = weights.get(name)
		if (saved === undefined) throw new RangeError(`The checkpoint has no parameter "${name}"`)
		if (saved.shape.length !== variable.value.shape.length || saved.shape.some((dimension, index) => dimension !== variable.value.shape[index])) throw new RangeError(`Parameter "${name}" has shape [${variable.value.shape.join(', ')}] but the checkpoint holds [${saved.shape.join(', ')}]`)
		variable.value.data.set(saved.data)
	}
}
