import { closeSync, fstatSync, openSync, readSync, writeSync } from 'node:fs'
import type { Random } from '@scantling/tensor'
import type { Batch, BatchSource } from '@scantling/training'

if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) throw new Error('Token files require a little-endian platform')

const MAGIC = 0x4b544353
const VERSION = 1
const HEADER_BYTES = 32
const CHUNK_TOKENS = 32768

/**
 * Appends token ids to a flat file of unsigned 16-bit values behind a 32-byte header holding the
 * magic number, version, element width, vocabulary size and token count. Call `close` to finish.
 */
export class TokenWriter {
	readonly #fd: number
	readonly #vocabSize: number
	readonly #buffer = new Uint16Array(CHUNK_TOKENS)
	#fill = 0
	#count = 0
	#closed = false

	constructor(path: string, vocabSize: number) {
		if (!Number.isInteger(vocabSize) || vocabSize < 1 || vocabSize > 65536) throw new RangeError('Token files support vocabularies of at most 65536 entries')
		this.#vocabSize = vocabSize
		this.#fd = openSync(path, 'w')
		writeSync(this.#fd, new Uint8Array(HEADER_BYTES))
	}

	append(ids: ArrayLike<number>): void {
		if (this.#closed) throw new Error('The token file is already closed')
		for (let index = 0; index < ids.length; index += 1) {
			const id = ids[index]!
			if (!Number.isInteger(id) || id < 0 || id >= this.#vocabSize) throw new RangeError(`Token id ${id} is outside the vocabulary of ${this.#vocabSize}`)
			this.#buffer[this.#fill++] = id
			if (this.#fill === CHUNK_TOKENS) this.#flush()
		}
	}

	/** Writes the header and closes the file. Returns the number of tokens written. */
	close(): number {
		if (this.#closed) return this.#count
		this.#flush()
		const header = new DataView(new ArrayBuffer(HEADER_BYTES))
		header.setUint32(0, MAGIC, true)
		header.setUint32(4, VERSION, true)
		header.setUint32(8, 16, true)
		header.setUint32(12, this.#vocabSize, true)
		header.setBigUint64(16, BigInt(this.#count), true)
		writeSync(this.#fd, new Uint8Array(header.buffer), 0, HEADER_BYTES, 0)
		closeSync(this.#fd)
		this.#closed = true
		return this.#count
	}

	#flush(): void {
		if (this.#fill === 0) return
		writeSync(this.#fd, new Uint8Array(this.#buffer.buffer, 0, this.#fill * 2))
		this.#count += this.#fill
		this.#fill = 0
	}
}

/** Random-access reader for a token file. Reads go to disk on demand, so the corpus never has to fit in memory. */
export class TokenFile {
	readonly count: number
	readonly vocabSize: number
	readonly #fd: number

	private constructor(fd: number, count: number, vocabSize: number) {
		this.#fd = fd
		this.count = count
		this.vocabSize = vocabSize
	}

	static open(path: string): TokenFile {
		const fd = openSync(path, 'r')
		try {
			const raw = new Uint8Array(HEADER_BYTES)
			if (readSync(fd, raw, 0, HEADER_BYTES, 0) !== HEADER_BYTES) throw new Error('The token file is too short to hold a header')
			const header = new DataView(raw.buffer)
			if (header.getUint32(0, true) !== MAGIC) throw new Error('Not a token file')
			if (header.getUint32(4, true) !== VERSION) throw new Error(`Unsupported token file version ${header.getUint32(4, true)}`)
			if (header.getUint32(8, true) !== 16) throw new Error('Unsupported token width')
			const count = Number(header.getBigUint64(16, true))
			if (fstatSync(fd).size !== HEADER_BYTES + count * 2) throw new Error('The token file is truncated or has trailing data')
			return new TokenFile(fd, count, header.getUint32(12, true))
		} catch (error) {
			closeSync(fd)
			throw error
		}
	}

	/** Reads `length` tokens starting at token index `start`. */
	read(start: number, length: number): Int32Array {
		if (!Number.isInteger(start) || !Number.isInteger(length) || start < 0 || length < 0 || start + length > this.count) throw new RangeError(`Read of ${length} tokens at ${start} is outside the file of ${this.count} tokens`)
		const bytes = new Uint8Array(length * 2)
		let filled = 0
		while (filled < bytes.length) {
			const got = readSync(this.#fd, bytes, filled, bytes.length - filled, HEADER_BYTES + start * 2 + filled)
			if (got === 0) throw new Error('Unexpected end of the token file')
			filled += got
		}
		return Int32Array.from(new Uint16Array(bytes.buffer))
	}

	readAll(): Int32Array {
		return this.read(0, this.count)
	}

	close(): void {
		closeSync(this.#fd)
	}
}

/** Endless random batches read straight from a token file. */
export class FileBatchSource implements BatchSource {
	constructor(private readonly file: TokenFile, private readonly length: number, private readonly batch: number, private readonly random: Random) {
		if (file.count < length + 1) throw new RangeError(`Need at least ${length + 1} tokens, the file has ${file.count}`)
	}

	next(): Batch {
		const inputs = new Int32Array(this.batch * this.length)
		const targets = new Int32Array(this.batch * this.length)
		for (let row = 0; row < this.batch; row += 1) {
			const chunk = this.file.read(this.random.nextInt(this.file.count - this.length), this.length + 1)
			inputs.set(chunk.subarray(0, this.length), row * this.length)
			targets.set(chunk.subarray(1), row * this.length)
		}
		return { inputs, targets, batch: this.batch, length: this.length }
	}
}

/** Deterministic, non-overlapping validation batches from the start of a token file. */
export function fileEvaluationBatches(file: TokenFile, length: number, batch: number, maxBatches: number): Batch[] {
	const windows = Math.floor((file.count - 1) / length)
	const batches: Batch[] = []
	for (let first = 0; first < windows && batches.length < maxBatches; first += batch) {
		const rows = Math.min(batch, windows - first)
		const chunk = file.read(first * length, rows * length + 1)
		batches.push({ inputs: chunk.slice(0, rows * length), targets: chunk.slice(1, rows * length + 1), batch: rows, length })
	}
	return batches
}
