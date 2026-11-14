import type { RegressionResult } from '@scantling/checkpoint'
import type { Tensor } from '@scantling/tensor'

/**
 * Cross-entropy, in nats per token, of `validation` under a unigram model fitted to `train` with
 * add-one smoothing. It is the loss of a model that knows how common each token is and nothing else.
 */
export function unigramLoss(train: ArrayLike<number>, validation: ArrayLike<number>, vocabSize: number): number {
	const counts = new Float64Array(vocabSize)
	for (let index = 0; index < train.length; index += 1) counts[train[index]!] = counts[train[index]!]! + 1
	const total = train.length + vocabSize
	let loss = 0
	for (let index = 0; index < validation.length; index += 1) loss -= Math.log((counts[validation[index]!]! + 1) / total)
	return loss / validation.length
}

/**
 * Bits per byte of text: the loss in nats per token, converted to bits, times the number of tokens,
 * divided by the number of bytes they stand for. Unlike loss per token it can be compared across tokenizers.
 */
export function bitsPerByte(lossPerToken: number, tokens: number, bytes: number): number {
	return (lossPerToken * tokens) / Math.LN2 / bytes
}

/** The number of bytes a sequence of tokens stands for. */
export function tokenBytes(tokenizer: { tokenBytes(id: number): Uint8Array }, tokens: ArrayLike<number>): number {
	let total = 0
	for (let index = 0; index < tokens.length; index += 1) total += tokenizer.tokenBytes(tokens[index]!).length
	return total
}

/**
 * Minimum regression checks for promoting a checkpoint: every weight is finite, the validation loss is
 * finite, and the model beats the unigram baseline, so a model that only memorised token frequencies is
 * never promoted.
 */
export function smokeChecks(params: readonly { readonly name: string; readonly variable: { readonly value: Tensor } }[], validationLoss: number, baseline: number): RegressionResult[] {
	const broken = params.find((entry) => entry.variable.value.data.some((value) => !Number.isFinite(value)))
	return [
		{ name: 'weights are finite', passed: broken === undefined, ...(broken === undefined ? {} : { detail: `parameter ${broken.name} has a non-finite value` }) },
		{ name: 'validation loss is finite', passed: Number.isFinite(validationLoss) },
		{ name: 'beats the unigram baseline', passed: validationLoss < baseline, detail: `validation ${validationLoss.toFixed(4)} against baseline ${baseline.toFixed(4)}` }
	]
}
