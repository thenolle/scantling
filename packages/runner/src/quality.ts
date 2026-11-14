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

/**
 * True when a closing bracket does not match the bracket it closes. Strings, template literals and comments are
 * skipped. The check is approximate: regular expression literals and JSX text can fool it. Unclosed brackets are
 * not a mismatch, so a cut-off completion is not penalised.
 */
export function hasMismatchedBrackets(text: string): boolean {
	const opens: Readonly<Record<string, string>> = { ')': '(', ']': '[', '}': '{' }
	const stack: string[] = []
	let quote: string | undefined
	for (let index = 0; index < text.length; index += 1) {
		const char = text[index]!
		if (quote !== undefined) {
			if (char === '\\') index += 1
			else if (char === quote || (char === '\n' && quote !== '`')) quote = undefined
			continue
		}
		if (char === '/' && text[index + 1] === '/') {
			while (index < text.length && text[index] !== '\n') index += 1
		} else if (char === '/' && text[index + 1] === '*') {
			const end = text.indexOf('*/', index + 2)
			if (end === -1) return false
			index = end + 1
		} else if (char === '"' || char === "'" || char === '`') {
			quote = char
		} else if (char === '(' || char === '[' || char === '{') {
			stack.push(char)
		} else if (opens[char] !== undefined && stack.pop() !== opens[char]) {
			return true
		}
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
	/** Share of prompts whose prompt plus generated text has mismatched brackets. */
	readonly generatedMismatch: number
	/** The same share for the prompt plus the real continuation, as a baseline. */
	readonly realMismatch: number
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
		if (hasMismatchedBrackets(promptText + generatedText)) generatedBad += 1
		if (hasMismatchedBrackets(promptText + expectedText)) realBad += 1
		if (samples.length < 3) samples.push({ prompt: promptText.slice(-160), expected: expectedText, generated: generatedText })
	}
	return {
		meanMatchingPrefix: prefixes / options.prompts,
		firstEightExact: exact / options.prompts,
		generatedMismatch: generatedBad / options.prompts,
		realMismatch: realBad / options.prompts,
		samples
	}
}
