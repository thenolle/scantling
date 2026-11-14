import type { Random } from '@scantling/tensor'

export interface SamplingOptions {
	/** 0 selects the most likely token deterministically. Defaults to 0.8. */
	readonly temperature?: number
	/** Keep only the k most likely tokens. 0 disables. */
	readonly topK?: number
	/** Keep the smallest set of tokens whose probability reaches p. 1 disables. */
	readonly topP?: number
	/** Divides positive logits and multiplies negative ones for recent tokens. 1 disables. */
	readonly repetitionPenalty?: number
	/** Subtracted from a token's logit for every earlier occurrence. */
	readonly frequencyPenalty?: number
	/** Subtracted once from the logit of every token that occurred before. */
	readonly presencePenalty?: number
	/** How many recent tokens the penalties look at. Defaults to 64. */
	readonly penaltyWindow?: number
	/** Tokens that can never be produced. */
	readonly banned?: readonly number[]
}

export interface Candidate {
	readonly token: number
	readonly probability: number
}

export interface SampledToken {
	readonly token: number
	readonly probability: number
	/** The most likely tokens at this step, best first. */
	readonly candidates: readonly Candidate[]
}

interface Resolved {
	readonly temperature: number
	readonly topK: number
	readonly topP: number
	readonly repetitionPenalty: number
	readonly frequencyPenalty: number
	readonly presencePenalty: number
	readonly penaltyWindow: number
	readonly banned: readonly number[]
}

export function resolveSampling(options: SamplingOptions = {}): Resolved {
	const resolved: Resolved = {
		temperature: options.temperature ?? 0.8,
		topK: options.topK ?? 0,
		topP: options.topP ?? 1,
		repetitionPenalty: options.repetitionPenalty ?? 1,
		frequencyPenalty: options.frequencyPenalty ?? 0,
		presencePenalty: options.presencePenalty ?? 0,
		penaltyWindow: options.penaltyWindow ?? 64,
		banned: options.banned ?? []
	}
	if (!(resolved.temperature >= 0) || !Number.isFinite(resolved.temperature)) throw new RangeError('temperature must be a finite number of at least 0')
	if (!Number.isInteger(resolved.topK) || resolved.topK < 0) throw new RangeError('topK must be a non-negative integer')
	if (!(resolved.topP > 0 && resolved.topP <= 1)) throw new RangeError('topP must be in (0, 1]')
	if (!(resolved.repetitionPenalty > 0) || !Number.isFinite(resolved.repetitionPenalty)) throw new RangeError('repetitionPenalty must be a positive number')
	if (!Number.isFinite(resolved.frequencyPenalty) || !Number.isFinite(resolved.presencePenalty)) throw new RangeError('The penalties must be finite')
	if (!Number.isInteger(resolved.penaltyWindow) || resolved.penaltyWindow < 0) throw new RangeError('penaltyWindow must be a non-negative integer')
	return resolved
}

function softmax(values: Float64Array, scale: number): Float64Array {
	let top = Number.NEGATIVE_INFINITY
	for (const value of values) if (value > top) top = value
	const out = new Float64Array(values.length)
	let total = 0
	for (let index = 0; index < values.length; index += 1) {
		out[index] = Math.exp((values[index]! - top) * scale)
		total += out[index]!
	}
	for (let index = 0; index < out.length; index += 1) out[index] = out[index]! / total
	return out
}

/**
 * Chooses the next token from `logits`. Order: bans and penalties on the raw logits, then temperature,
 * softmax, top-k, top-p, renormalisation and a draw from `random`. With temperature 0 the most likely
 * token (lowest id on ties) is returned and `random` is not used.
 */
export function sampleToken(logits: ArrayLike<number>, history: readonly number[], options: SamplingOptions, random: Random, traceSize = 5): SampledToken {
	const config = resolveSampling(options)
	const adjusted = Float64Array.from(logits)
	const recent = history.slice(config.penaltyWindow === 0 ? history.length : -config.penaltyWindow)
	const counts = new Map<number, number>()
	for (const token of recent) counts.set(token, (counts.get(token) ?? 0) + 1)
	for (const [token, count] of counts) {
		if (token < 0 || token >= adjusted.length) continue
		let value = adjusted[token]!
		if (config.repetitionPenalty !== 1) value = value > 0 ? value / config.repetitionPenalty : value * config.repetitionPenalty
		adjusted[token] = value - config.frequencyPenalty * count - config.presencePenalty
	}
	for (const token of config.banned) if (token >= 0 && token < adjusted.length) adjusted[token] = Number.NEGATIVE_INFINITY
	const rank = (probabilities: Float64Array): number[] => Array.from(probabilities.keys()).filter((token) => probabilities[token]! > 0).sort((a, b) => probabilities[b]! - probabilities[a]! || a - b)
	if (config.temperature === 0) {
		const probabilities = softmax(adjusted, 1)
		const order = rank(probabilities)
		if (order.length === 0) throw new RangeError('Every token is banned')
		return { token: order[0]!, probability: probabilities[order[0]!]!, candidates: order.slice(0, traceSize).map((token) => ({ token, probability: probabilities[token]! })) }
	}
	const probabilities = softmax(adjusted, 1 / config.temperature)
	let order = rank(probabilities)
	if (order.length === 0) throw new RangeError('Every token is banned')
	if (config.topK > 0) order = order.slice(0, config.topK)
	if (config.topP < 1) {
		let cumulative = 0
		let keep = 0
		while (keep < order.length) {
			cumulative += probabilities[order[keep]!]!
			keep += 1
			if (cumulative >= config.topP) break
		}
		order = order.slice(0, keep)
	}
	const mass = order.reduce((total, token) => total + probabilities[token]!, 0)
	const candidates = order.map((token) => ({ token, probability: probabilities[token]! / mass }))
	let draw = random.next()
	let chosen = candidates[candidates.length - 1]!
	for (const candidate of candidates) {
		draw -= candidate.probability
		if (draw < 0) {
			chosen = candidate
			break
		}
	}
	return { token: chosen.token, probability: chosen.probability, candidates: candidates.slice(0, traceSize) }
}
