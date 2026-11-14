import { applyMerge, PAIR_BASE, pairKey } from './merge.ts'
import { byteChunks, pretokenize, PRETOKENIZER_ID } from './pretokenize.ts'

export type Merge = readonly [number, number]

export const DEFAULT_SPECIAL_TOKENS = [
	'<|pad|>',
	'<|bos|>',
	'<|eos|>',
	'<|file|>',
	'<|endfile|>',
	'<|system|>',
	'<|user|>',
	'<|assistant|>',
	'<|endturn|>'
] as const

export interface EncodeOptions {
	/** Which special tokens are recognised in the text. Defaults to `none`, so control tokens cannot be injected. */
	readonly allowedSpecial?: 'none' | 'all' | readonly string[]
}

export interface DecodeOptions {
	readonly skipSpecial?: boolean
}

const FORMAT = 'scantling-bpe'
const VERSION = 1
const CACHE_LIMIT = 50_000
const SPECIAL_FORMAT = /^<\|[a-z][a-z0-9_]*\|>$/
const encoder = new TextEncoder()

const escapeRegExp = (value: string): string => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

function append(target: number[], source: readonly number[]): void {
	for (const value of source) target.push(value)
}

/**
 * A byte-level BPE tokenizer. Ids `0 .. specials-1` are special tokens, the next 256 are the byte
 * values, and every learned merge adds one more id in learning order.
 */
export class Tokenizer {
	readonly specialTokens: readonly string[]
	readonly merges: readonly Merge[]
	readonly vocabSize: number
	readonly #base: number
	readonly #bytes: Uint8Array[] = []
	readonly #ranks = new Map<number, number>()
	readonly #specialIds = new Map<string, number>()
	readonly #cache = new Map<string, readonly number[]>()

	constructor(specialTokens: readonly string[], merges: readonly Merge[] = []) {
		this.specialTokens = [...specialTokens]
		this.merges = merges.map(([left, right]) => [left, right] as const)
		specialTokens.forEach((token, index) => {
			if (!SPECIAL_FORMAT.test(token)) throw new RangeError(`Special token "${token}" must look like <|name|>`)
			if (this.#specialIds.has(token)) throw new RangeError(`Duplicate special token "${token}"`)
			this.#specialIds.set(token, index)
			this.#bytes.push(encoder.encode(token))
		})
		this.#base = specialTokens.length
		for (let byte = 0; byte < 256; byte += 1) this.#bytes.push(Uint8Array.of(byte))
		merges.forEach(([left, right], index) => {
			const limit = this.#bytes.length
			const valid = [left, right].every((id) => Number.isInteger(id) && id >= this.#base && id < limit)
			if (!valid) throw new RangeError(`Merge ${index} refers to an unknown token`)
			const key = pairKey(left, right)
			if (this.#ranks.has(key)) throw new RangeError(`Merge ${index} repeats an earlier merge`)
			this.#ranks.set(key, index)
			const joined = new Uint8Array(this.#bytes[left]!.length + this.#bytes[right]!.length)
			joined.set(this.#bytes[left]!)
			joined.set(this.#bytes[right]!, this.#bytes[left]!.length)
			this.#bytes.push(joined)
		})
		this.vocabSize = this.#bytes.length
		if (this.vocabSize >= PAIR_BASE) throw new RangeError('The vocabulary is too large')
	}

	/** The id of a special token. */
	idOf(token: string): number {
		const id = this.#specialIds.get(token)
		if (id === undefined) throw new RangeError(`Unknown special token "${token}"`)
		return id
	}

	isSpecial(id: number): boolean {
		return id >= 0 && id < this.#base
	}

	/** The bytes a token stands for. Special tokens stand for their literal text. */
	tokenBytes(id: number): Uint8Array {
		const bytes = this.#bytes[id]
		if (bytes === undefined) throw new RangeError(`Token id ${id} is outside the vocabulary of ${this.vocabSize}`)
		return bytes
	}

	encode(text: string, options: EncodeOptions = {}): number[] {
		const allowed = this.#allowed(options.allowedSpecial ?? 'none')
		if (allowed.length === 0) return this.#encodeOrdinary(text)
		const out: number[] = []
		let cursor = 0
		const pattern = new RegExp(allowed.map(escapeRegExp).join('|'), 'g')
		for (const match of text.matchAll(pattern)) {
			const start = match.index ?? 0
			append(out, this.#encodeOrdinary(text.slice(cursor, start)))
			out.push(this.idOf(match[0]))
			cursor = start + match[0].length
		}
		append(out, this.#encodeOrdinary(text.slice(cursor)))
		return out
	}

	decodeBytes(ids: ArrayLike<number>, options: DecodeOptions = {}): Uint8Array {
		const parts: Uint8Array[] = []
		let length = 0
		for (let index = 0; index < ids.length; index += 1) {
			const id = ids[index]!
			const bytes = this.tokenBytes(id)
			if (options.skipSpecial === true && this.isSpecial(id)) continue
			parts.push(bytes)
			length += bytes.length
		}
		const out = new Uint8Array(length)
		let offset = 0
		for (const part of parts) {
			out.set(part, offset)
			offset += part.length
		}
		return out
	}

	/** Decodes ids to text. Byte sequences that are not valid UTF-8 become U+FFFD. */
	decode(ids: ArrayLike<number>, options: DecodeOptions = {}): string {
		return new TextDecoder().decode(this.decodeBytes(ids, options))
	}

	toJson(): string {
		return JSON.stringify({ format: FORMAT, version: VERSION, pretokenizer: PRETOKENIZER_ID, specialTokens: this.specialTokens, merges: this.merges })
	}

	/** Parses and validates a serialised tokenizer. Throws on any inconsistency. */
	static fromJson(text: string): Tokenizer {
		const data: unknown = JSON.parse(text)
		if (typeof data !== 'object' || data === null) throw new Error('Tokenizer file must contain an object')
		const record = data as Record<string, unknown>
		if (record['format'] !== FORMAT) throw new Error(`Not a ${FORMAT} file`)
		if (record['version'] !== VERSION) throw new Error(`Unsupported tokenizer version ${String(record['version'])}`)
		if (record['pretokenizer'] !== PRETOKENIZER_ID) throw new Error(`Unsupported pre-tokenizer ${String(record['pretokenizer'])}`)
		const specials = record['specialTokens']
		const merges = record['merges']
		if (!Array.isArray(specials) || !specials.every((token): token is string => typeof token === 'string')) throw new Error('specialTokens must be a list of strings')
		const pairs = Array.isArray(merges) ? merges : undefined
		if (pairs === undefined || !pairs.every((pair) => Array.isArray(pair) && pair.length === 2 && pair.every((id) => typeof id === 'number'))) throw new Error('merges must be a list of id pairs')
		return new Tokenizer(specials, pairs as Merge[])
	}

	#allowed(option: NonNullable<EncodeOptions['allowedSpecial']>): readonly string[] {
		if (option === 'none') return []
		if (option === 'all') return this.specialTokens
		for (const token of option) this.idOf(token)
		return option
	}

	#encodeOrdinary(text: string): number[] {
		const out: number[] = []
		for (const piece of pretokenize(text)) {
			let ids = this.#cache.get(piece)
			if (ids === undefined) {
				const computed: number[] = []
				for (const chunk of byteChunks(encoder.encode(piece))) append(computed, this.#encodeChunk(chunk))
				if (this.#cache.size >= CACHE_LIMIT) this.#cache.clear()
				this.#cache.set(piece, computed)
				ids = computed
			}
			append(out, ids)
		}
		return out
	}

	#encodeChunk(bytes: Uint8Array): number[] {
		let ids = Array.from(bytes, (byte) => this.#base + byte)
		for (; ;) {
			let bestRank = Number.POSITIVE_INFINITY
			let bestKey = -1
			for (let index = 0; index < ids.length - 1; index += 1) {
				const key = pairKey(ids[index]!, ids[index + 1]!)
				const rank = this.#ranks.get(key)
				if (rank !== undefined && rank < bestRank) {
					bestRank = rank
					bestKey = key
				}
			}
			if (bestKey === -1) return ids
			ids = applyMerge(ids, Math.floor(bestKey / PAIR_BASE), bestKey % PAIR_BASE, this.#base + 256 + bestRank)
		}
	}
}

/**
 * Decodes a stream of ids one at a time. Bytes of a character that is split across several tokens
 * are held back until the character is complete.
 */
export class StreamDecoder {
	readonly #tokenizer: Tokenizer
	readonly #skipSpecial: boolean
	readonly #decoder = new TextDecoder()

	constructor(tokenizer: Tokenizer, options: DecodeOptions = {}) {
		this.#tokenizer = tokenizer
		this.#skipSpecial = options.skipSpecial === true
	}

	push(id: number): string {
		const bytes = this.#tokenizer.tokenBytes(id)
		if (this.#skipSpecial && this.#tokenizer.isSpecial(id)) return ''
		return this.#decoder.decode(bytes, { stream: true })
	}

	/** Emits anything still buffered, replacing an incomplete character with U+FFFD. */
	flush(): string {
		return this.#decoder.decode()
	}
}
