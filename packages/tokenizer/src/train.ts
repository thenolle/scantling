import { applyMerge, PAIR_BASE, pairKey } from './merge.ts'
import { byteChunks, pretokenize } from './pretokenize.ts'
import { DEFAULT_SPECIAL_TOKENS, Tokenizer } from './tokenizer.ts'
import type { Merge } from './tokenizer.ts'

export interface TrainOptions {
	/** Total vocabulary size including special tokens and the 256 bytes. */
	readonly vocabSize: number
	readonly specialTokens?: readonly string[]
	/** Pairs seen fewer times than this are never merged. Defaults to 2. */
	readonly minFrequency?: number
	readonly onProgress?: (merges: number, target: number) => void
}

export interface TrainResult {
	readonly tokenizer: Tokenizer
	readonly corpusBytes: number
	readonly distinctPieces: number
	readonly merges: number
}

interface Word {
	ids: number[]
	readonly count: number
}

class PairHeap {
	readonly #counts: number[] = []
	readonly #keys: number[] = []

	push(count: number, key: number): void {
		let index = this.#counts.length
		this.#counts.push(count)
		this.#keys.push(key)
		while (index > 0) {
			const parent = (index - 1) >> 1
			if (!this.#before(index, parent)) break
			this.#swap(index, parent)
			index = parent
		}
	}

	pop(): [count: number, key: number] | undefined {
		if (this.#counts.length === 0) return undefined
		const top: [number, number] = [this.#counts[0]!, this.#keys[0]!]
		const lastCount = this.#counts.pop()!
		const lastKey = this.#keys.pop()!
		if (this.#counts.length > 0) {
			this.#counts[0] = lastCount
			this.#keys[0] = lastKey
			let index = 0
			for (; ;) {
				const left = index * 2 + 1
				const right = left + 1
				let best = index
				if (left < this.#counts.length && this.#before(left, best)) best = left
				if (right < this.#counts.length && this.#before(right, best)) best = right
				if (best === index) break
				this.#swap(index, best)
				index = best
			}
		}
		return top
	}

	#before(a: number, b: number): boolean {
		return this.#counts[a]! > this.#counts[b]! || (this.#counts[a] === this.#counts[b] && this.#keys[a]! < this.#keys[b]!)
	}

	#swap(a: number, b: number): void {
		const [countA, countB] = [this.#counts[b]!, this.#counts[a]!]
		this.#counts[a] = countA
		this.#counts[b] = countB
		const [keyA, keyB] = [this.#keys[b]!, this.#keys[a]!]
		this.#keys[a] = keyA
		this.#keys[b] = keyB
	}
}

/**
 * Learns BPE merges from `texts`. Texts are pre-tokenized, identical pieces are counted once,
 * and the most frequent adjacent pair is merged repeatedly. Pair counts and the word index are
 * updated incrementally, and ties break toward the smaller pair, so training is deterministic.
 */
export function trainBpe(texts: Iterable<string>, options: TrainOptions): TrainResult {
	const specials = options.specialTokens ?? DEFAULT_SPECIAL_TOKENS
	const base = specials.length + 256
	if (!Number.isInteger(options.vocabSize) || options.vocabSize < base || options.vocabSize >= PAIR_BASE) throw new RangeError(`vocabSize must be an integer between ${base} and ${PAIR_BASE - 1}`)
	const minFrequency = options.minFrequency ?? 2
	const encoder = new TextEncoder()
	const counts = new Map<string, number>()
	let corpusBytes = 0
	for (const text of texts) {
		for (const piece of pretokenize(text)) {
			const bytes = encoder.encode(piece)
			corpusBytes += bytes.length
			for (const chunk of byteChunks(bytes)) {
				const key = Buffer.from(chunk).toString('latin1')
				counts.set(key, (counts.get(key) ?? 0) + 1)
			}
		}
	}
	const words: Word[] = [...counts].map(([key, count]) => ({ ids: Array.from(Buffer.from(key, 'latin1'), (byte) => specials.length + byte), count }))
	const pairCounts = new Map<number, number>()
	const pairWords = new Map<number, Set<number>>()
	const heap = new PairHeap()
	const update = (index: number, sign: 1 | -1, touched: Set<number>): void => {
		const word = words[index]!
		for (let position = 0; position < word.ids.length - 1; position += 1) {
			const key = pairKey(word.ids[position]!, word.ids[position + 1]!)
			const next = (pairCounts.get(key) ?? 0) + sign * word.count
			if (next > 0) pairCounts.set(key, next)
			else pairCounts.delete(key)
			let owners = pairWords.get(key)
			if (sign === 1) {
				if (owners === undefined) {
					owners = new Set()
					pairWords.set(key, owners)
				}
				owners.add(index)
			} else {
				owners?.delete(index)
			}
			touched.add(key)
		}
	}
	const publish = (touched: Set<number>): void => {
		for (const key of touched) {
			const count = pairCounts.get(key)
			if (count !== undefined) heap.push(count, key)
		}
	}
	const initial = new Set<number>()
	words.forEach((_word, index) => update(index, 1, initial))
	publish(initial)
	const merges: Merge[] = []
	const target = options.vocabSize - base
	while (merges.length < target) {
		let best = -1
		let bestCount = 0
		for (let top = heap.pop(); top !== undefined; top = heap.pop()) {
			if (pairCounts.get(top[1]) === top[0]) {
				best = top[1]
				bestCount = top[0]
				break
			}
		}
		if (best === -1 || bestCount < minFrequency) break
		const left = Math.floor(best / PAIR_BASE)
		const right = best % PAIR_BASE
		const merged = base + merges.length
		merges.push([left, right])
		const touched = new Set<number>()
		for (const index of [...(pairWords.get(best) ?? [])]) {
			update(index, -1, touched)
			const word = words[index]!
			word.ids = applyMerge(word.ids, left, right, merged)
			update(index, 1, touched)
		}
		pairCounts.delete(best)
		pairWords.delete(best)
		publish(touched)
		options.onProgress?.(merges.length, target)
	}
	return { tokenizer: new Tokenizer(specials, merges), corpusBytes, distinctPieces: counts.size, merges: merges.length }
}
