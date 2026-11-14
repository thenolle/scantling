import type { Transformer } from '@scantling/nn'
import { Random } from '@scantling/tensor'
import { sampleToken } from './sampler.ts'
import type { Candidate, SamplingOptions } from './sampler.ts'
import { InferenceSession } from './session.ts'
import { StopMatcher } from './stop.ts'

/** Turns token ids into text incrementally. `StreamDecoder` from the tokenizer package fits. */
export interface TokenText {
	push(id: number): string
	flush(): string
}

export interface GenerateOptions {
	readonly maxNewTokens: number
	readonly sampling?: SamplingOptions
	/** Seed of the sampling generator. Defaults to 1. */
	readonly seed?: number
	readonly stopTokens?: readonly number[]
	readonly stopSequences?: readonly string[]
	readonly text?: TokenText
	readonly signal?: AbortSignal
	/** What to do when the context window is full: re-feed the recent half (`slide`) or finish (`stop`). Defaults to `slide`. */
	readonly contextPolicy?: 'slide' | 'stop'
	/** Candidates to report per token. Defaults to 0. */
	readonly trace?: number
}

export type FinishReason = 'length' | 'stop-token' | 'stop-sequence' | 'cancelled' | 'context'

export type GenerationEvent =
	| { readonly type: 'token'; readonly token: number; readonly text: string; readonly probability: number; readonly candidates: readonly Candidate[]; readonly position: number }
	| { readonly type: 'done'; readonly reason: FinishReason; readonly tokens: number; readonly tail: string }

/**
 * Generates tokens for `prompt`, yielding each as it is produced. A prompt longer than the window is cut to
 * its last `maxSeqLen - 1` tokens. The final `done` event carries any text still held back by the decoder.
 */
export async function* generate(model: Transformer, prompt: readonly number[], options: GenerateOptions): AsyncGenerator<GenerationEvent> {
	if (prompt.length === 0) throw new RangeError('The prompt must contain at least one token')
	const { maxSeqLen } = model.config
	if (maxSeqLen < 2) throw new RangeError('The context window must hold at least two tokens')
	const session = new InferenceSession(model)
	const random = new Random(options.seed ?? 1)
	const matcher = options.stopSequences !== undefined && options.stopSequences.length > 0 ? new StopMatcher(options.stopSequences) : undefined
	const history = [...prompt]
	let logits = session.feed(prompt.slice(-(maxSeqLen - 1))).data
	let generated = 0
	const finish = (reason: FinishReason, hold: boolean): GenerationEvent => ({
		type: 'done',
		reason,
		tokens: generated,
		tail: hold ? (matcher?.flush() ?? '') + (options.text?.flush() ?? '') : ''
	})
	while (generated < options.maxNewTokens) {
		if (options.signal?.aborted === true) {
			yield finish('cancelled', true)
			return
		}
		const sampled = sampleToken(logits, history, options.sampling ?? {}, random, options.trace ?? 0)
		if (options.stopTokens?.includes(sampled.token) === true) {
			yield finish('stop-token', true)
			return
		}
		history.push(sampled.token)
		generated += 1
		const piece = options.text?.push(sampled.token) ?? ''
		const matched = matcher?.push(piece) ?? { emit: piece, stopped: false }
		yield { type: 'token', token: sampled.token, text: matched.emit, probability: sampled.probability, candidates: sampled.candidates, position: history.length - 1 }
		if (matched.stopped) {
			yield finish('stop-sequence', false)
			return
		}
		if (generated === options.maxNewTokens) break
		if (session.length + 1 > maxSeqLen) {
			if ((options.contextPolicy ?? 'slide') === 'stop') {
				yield finish('context', true)
				return
			}
			session.reset()
			logits = session.feed(history.slice(-Math.floor(maxSeqLen / 2))).data
		} else {
			logits = session.feed([sampled.token]).data
		}
		await new Promise<void>((resolve) => setTimeout(resolve, 0))
	}
	yield finish('length', true)
}
