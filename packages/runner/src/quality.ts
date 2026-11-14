import { noGrad } from '@scantling/autodiff'
import type { TokenFile } from '@scantling/datasets'
import { FastDecoder } from '@scantling/inference'
import type { Transformer } from '@scantling/nn'
import { Random } from '@scantling/tensor'
import type { Tokenizer } from '@scantling/tokenizer'
import type { Batch } from '@scantling/training'

export interface Accuracy {
	readonly top1: number
	readonly top5: number
	readonly tokens: number
}

/**
 * How often the true next token is the model's top choice, and among its top five, over held-out batches.
 * Ties are not counted against the model.
 */
export function nextTokenAccuracy(model: Pick<Transformer, 'config' | 'forward'>, batches: Iterable<Batch>): Accuracy {
	const vocab = model.config.vocabSize
	let top1 = 0
	let top5 = 0
	let tokens = 0
	for (const batch of batches) {
		const logits = noGrad(() => model.forward(batch.inputs, batch.batch, batch.length)).value.data
		const rows = batch.batch * batch.length
		for (let row = 0; row < rows; row += 1) {
			const base = row * vocab
			const wanted = logits[base + batch.targets[row]!]!
			let better = 0
			for (let token = 0; token < vocab; token += 1) if (logits[base + token]! > wanted) better += 1
			if (better === 0) top1 += 1
			if (better < 5) top5 += 1
		}
		tokens += rows
	}
	return { top1: top1 / tokens, top5: top5 / tokens, tokens }
}

/** Where a bracket scan stands: open brackets, an open string or comment, and whether nesting has broken. */
export interface BracketState {
	readonly stack: readonly string[]
	readonly quote: string | undefined
	readonly comment: 'line' | 'block' | undefined
	/** True once a closing bracket disagreed with the bracket it closed. */
	readonly mismatch: boolean
}

export const START_STATE: BracketState = { stack: [], quote: undefined, comment: undefined, mismatch: false }

const CLOSERS: Readonly<Record<string, string>> = { ')': '(', ']': '[', '}': '{' }

/**
 * Scans text for bracket nesting, skipping strings, template literals and comments, and returns the state so a
 * scan can continue from where another stopped. With `tolerateUnderflow`, a closer with nothing open is not an
 * error, which suits text that starts in the middle of a file. The check is approximate: regular expression
 * literals and JSX text can fool it.
 */
export function scanBrackets(text: string, from: BracketState = START_STATE, tolerateUnderflow = false): BracketState {
	const stack = [...from.stack]
	let quote = from.quote
	let comment = from.comment
	let mismatch = from.mismatch
	for (let index = 0; index < text.length; index += 1) {
		const char = text[index]!
		if (comment === 'line') {
			if (char === '\n') comment = undefined
			continue
		}
		if (comment === 'block') {
			if (char === '*' && text[index + 1] === '/') {
				comment = undefined
				index += 1
			}
			continue
		}
		if (quote !== undefined) {
			if (char === '\\') index += 1
			else if (char === quote || (char === '\n' && quote !== '`')) quote = undefined
			continue
		}
		if (char === '/' && text[index + 1] === '/') {
			comment = 'line'
			index += 1
		} else if (char === '/' && text[index + 1] === '*') {
			comment = 'block'
			index += 1
		} else if (char === '"' || char === "'" || char === '`') {
			quote = char
		} else if (char === '(' || char === '[' || char === '{') {
			stack.push(char)
		} else if (CLOSERS[char] !== undefined) {
			const top = stack.pop()
			if (top === undefined ? !tolerateUnderflow : top !== CLOSERS[char]) mismatch = true
		}
	}
	return { stack, quote, comment, mismatch }
}

/** True when a closing bracket does not match the bracket it closes. Unclosed brackets are not a mismatch. */
export function hasMismatchedBrackets(text: string): boolean {
	return scanBrackets(text).mismatch
}

/** True when the end of a sequence repeats with a period of at most `maxPeriod` tokens across `window` tokens. */
export function endsInLoop(tokens: readonly number[], window = 24, maxPeriod = 8): boolean {
	if (tokens.length < window) return false
	const tail = tokens.slice(-window)
	for (let period = 1; period <= maxPeriod; period += 1) {
		let repeats = true
		for (let index = period; index < tail.length; index += 1) {
			if (tail[index] !== tail[index - period]) {
				repeats = false
				break
			}
		}
		if (repeats) return true
	}
	return false
}

/** The number of leading tokens two sequences share. */
export function matchingPrefix(expected: readonly number[], actual: readonly number[]): number {
	let length = 0
	while (length < expected.length && length < actual.length && expected[length] === actual[length]) length += 1
	return length
}

export interface CompletionOptions {
	readonly prompts: number
	readonly promptTokens: number
	readonly newTokens: number
	readonly seed: number
}

export interface CompletionReport {
	/** Mean number of leading generated tokens that equal the real continuation. */
	readonly meanMatchingPrefix: number
	/** Share of prompts whose first eight generated tokens equal the real ones. */
	readonly firstEightExact: number
	/** Share of prompts whose generated text breaks bracket nesting, counted from the state the prompt left. */
	readonly generatedMismatch: number
	/** The same share for the real continuation, as a baseline. */
	readonly realMismatch: number
	/** Share of generated sequences that end in a repeating loop. */
	readonly generatedLoops: number
	/** The same share for the real continuation, as a baseline. */
	readonly realLoops: number
	readonly samples: readonly { readonly prompt: string; readonly expected: string; readonly generated: string }[]
}

function argmax(values: Float32Array): number {
	let best = 0
	for (let index = 1; index < values.length; index += 1) if (values[index]! > values[best]!) best = index
	return best
}

/**
 * Greedy completion test on held-out code: gives the model the start of a random stretch and compares what it
 * writes with what the file really contains.
 */
export function evaluateCompletions(model: Transformer, tokenizer: Tokenizer, file: TokenFile, options: CompletionOptions): CompletionReport {
	const total = options.promptTokens + options.newTokens
	const random = new Random(options.seed)
	const decoder = new FastDecoder(model, 'float32')
	const samples: { prompt: string; expected: string; generated: string }[] = []
	let prefixes = 0
	let exact = 0
	let generatedBad = 0
	let realBad = 0
	let generatedLooping = 0
	let realLooping = 0
	for (let prompt = 0; prompt < options.prompts; prompt += 1) {
		const chunk = file.read(random.nextInt(file.count - total), total)
		const promptIds = Array.from(chunk.subarray(0, options.promptTokens))
		const expected = Array.from(chunk.subarray(options.promptTokens))
		decoder.reset()
		let logits = decoder.feed(promptIds)
		const generated: number[] = []
		for (let index = 0; index < options.newTokens; index += 1) {
			const next = argmax(logits)
			generated.push(next)
			if (index + 1 < options.newTokens) logits = decoder.step(next)
		}
		const matched = matchingPrefix(expected, generated)
		prefixes += matched
		if (matched >= 8) exact += 1
		const promptText = tokenizer.decode(promptIds)
		const generatedText = tokenizer.decode(generated)
		const expectedText = tokenizer.decode(expected)
		const afterPrompt = { ...scanBrackets(promptText, START_STATE, true), mismatch: false }
		if (scanBrackets(generatedText, afterPrompt, true).mismatch) generatedBad += 1
		if (scanBrackets(expectedText, afterPrompt, true).mismatch) realBad += 1
		if (endsInLoop(generated)) generatedLooping += 1
		if (endsInLoop(expected)) realLooping += 1
		if (samples.length < 3) samples.push({ prompt: promptText.slice(-160), expected: expectedText, generated: generatedText })
	}
	return {
		meanMatchingPrefix: prefixes / options.prompts,
		firstEightExact: exact / options.prompts,
		generatedMismatch: generatedBad / options.prompts,
		realMismatch: realBad / options.prompts,
		generatedLoops: generatedLooping / options.prompts,
		realLoops: realLooping / options.prompts,
		samples
	}
}
