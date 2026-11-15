const BLOCK_MAGIC_HIGH = 0x314159
const BLOCK_MAGIC_LOW = 0x265359
const END_MAGIC_HIGH = 0x177245
const END_MAGIC_LOW = 0x385090
const MAX_CODE_LENGTH = 20
const GROUP_SIZE = 50

const CRC_TABLE = ((): Uint32Array => {
	const table = new Uint32Array(256)
	for (let index = 0; index < 256; index += 1) {
		let crc = index << 24
		for (let bit = 0; bit < 8; bit += 1) crc = (crc & 0x80000000) !== 0 ? (crc << 1) ^ 0x04c11db7 : crc << 1
		table[index] = crc >>> 0
	}
	return table
})()

function blockCrc(bytes: Uint8Array, from: number, to: number): number {
	let crc = 0xffffffff
	for (let index = from; index < to; index += 1) crc = ((crc << 8) ^ CRC_TABLE[((crc >>> 24) ^ bytes[index]!) & 0xff]!) >>> 0
	return ~crc >>> 0
}

class BitReader {
	pos = 0
	#buffer = 0
	#count = 0
	readonly #data: Uint8Array

	constructor(data: Uint8Array) {
		this.#data = data
	}

	get exhausted(): boolean {
		return this.pos >= this.#data.length
	}

	read(n: number): number {
		while (this.#count < n) {
			const byte = this.#data[this.pos]
			if (byte === undefined) throw new Error('The bzip2 data ends too early')
			this.pos += 1
			this.#buffer = ((this.#buffer << 8) | byte) >>> 0
			this.#count += 8
		}
		this.#count -= n
		const value = (this.#buffer >>> this.#count) & ((1 << n) - 1)
		this.#buffer &= (1 << this.#count) - 1
		return value
	}

	align(): void {
		this.#count = 0
		this.#buffer = 0
	}
}

class Output {
	bytes: Uint8Array
	length = 0

	constructor(hint: number) {
		this.bytes = new Uint8Array(Math.max(1024, hint))
	}

	ensure(extra: number): void {
		if (this.length + extra <= this.bytes.length) return
		let size = this.bytes.length * 2
		while (size < this.length + extra) size *= 2
		const grown = new Uint8Array(size)
		grown.set(this.bytes.subarray(0, this.length))
		this.bytes = grown
	}

	result(): Uint8Array {
		return this.bytes.subarray(0, this.length)
	}
}

interface HuffmanTable {
	readonly minLength: number
	readonly permutation: Uint16Array
	readonly base: Int32Array
	readonly limit: Int32Array
}

function buildTable(lengths: Uint8Array, alphaSize: number): HuffmanTable {
	let minLength = 32
	let maxLength = 0
	for (let index = 0; index < alphaSize; index += 1) {
		const length = lengths[index]!
		if (length > maxLength) maxLength = length
		if (length < minLength) minLength = length
	}
	const permutation = new Uint16Array(alphaSize)
	let at = 0
	for (let length = minLength; length <= maxLength; length += 1) {
		for (let index = 0; index < alphaSize; index += 1) if (lengths[index] === length) permutation[at++] = index
	}
	const base = new Int32Array(MAX_CODE_LENGTH + 2)
	const limit = new Int32Array(MAX_CODE_LENGTH + 2)
	for (let index = 0; index < alphaSize; index += 1) {
		const slot = lengths[index]! + 1
		base[slot] = base[slot]! + 1
	}
	for (let index = 1; index < base.length; index += 1) base[index] = base[index]! + base[index - 1]!
	let vector = 0
	for (let index = minLength; index <= maxLength; index += 1) {
		vector += base[index + 1]! - base[index]!
		limit[index] = vector - 1
		vector <<= 1
	}
	for (let index = minLength + 1; index <= maxLength; index += 1) base[index] = ((limit[index - 1]! + 1) << 1) - base[index]!
	return { minLength, permutation, base, limit }
}

function decodeBlock(bits: BitReader, blockMax: number, output: Output, tt: Int32Array): void {
	if (bits.read(1) !== 0) throw new Error('Randomised bzip2 blocks are not supported')
	const origPtr = bits.read(24)
	const used: number[] = []
	const groups = bits.read(16)
	for (let high = 0; high < 16; high += 1) {
		if ((groups & (0x8000 >> high)) === 0) continue
		const inner = bits.read(16)
		for (let low = 0; low < 16; low += 1) if ((inner & (0x8000 >> low)) !== 0) used.push(high * 16 + low)
	}
	if (used.length === 0) throw new Error('The bzip2 block uses no symbols')
	const alphaSize = used.length + 2
	const tableCount = bits.read(3)
	if (tableCount < 2 || tableCount > 6) throw new Error('Bad number of bzip2 coding tables')
	const selectorCount = bits.read(15)
	if (selectorCount < 1) throw new Error('Bad number of bzip2 selectors')
	const order = [0, 1, 2, 3, 4, 5].slice(0, tableCount)
	const selectors = new Uint8Array(selectorCount)
	for (let index = 0; index < selectorCount; index += 1) {
		let position = 0
		while (bits.read(1) === 1) {
			position += 1
			if (position >= tableCount) throw new Error('Bad bzip2 selector')
		}
		const picked = order[position]!
		for (let k = position; k > 0; k -= 1) order[k] = order[k - 1]!
		order[0] = picked
		selectors[index] = picked
	}
	const tables: HuffmanTable[] = []
	for (let t = 0; t < tableCount; t += 1) {
		const lengths = new Uint8Array(alphaSize)
		let current = bits.read(5)
		for (let index = 0; index < alphaSize; index += 1) {
			for (; ;) {
				if (current < 1 || current > MAX_CODE_LENGTH) throw new Error('Bad bzip2 code length')
				if (bits.read(1) === 0) break
				current += bits.read(1) === 0 ? 1 : -1
			}
			lengths[index] = current
		}
		tables.push(buildTable(lengths, alphaSize))
	}
	const endOfBlock = used.length + 1
	const mtf = new Uint8Array(used.length)
	for (let index = 0; index < used.length; index += 1) mtf[index] = index
	const counts = new Int32Array(256)
	let size = 0
	let selectorIndex = 0
	let groupLeft = 0
	let table = tables[0]!
	let runLength = 0
	let runWeight = 1
	for (; ;) {
		if (groupLeft === 0) {
			if (selectorIndex >= selectorCount) throw new Error('The bzip2 block has too few selectors')
			table = tables[selectors[selectorIndex++]!]!
			groupLeft = GROUP_SIZE
		}
		groupLeft -= 1
		let length = table.minLength
		let code = bits.read(length)
		while (code > table.limit[length]!) {
			length += 1
			if (length > MAX_CODE_LENGTH) throw new Error('Bad bzip2 Huffman code')
			code = (code << 1) | bits.read(1)
		}
		const symbol = table.permutation[code - table.base[length]!]!
		if (symbol <= 1) {
			runLength += symbol === 0 ? runWeight : runWeight * 2
			runWeight *= 2
			continue
		}
		if (runLength > 0) {
			const byte = used[mtf[0]!]!
			if (size + runLength > blockMax) throw new Error('The bzip2 block is larger than declared')
			counts[byte] = counts[byte]! + runLength
			tt.fill(byte, size, size + runLength)
			size += runLength
			runLength = 0
			runWeight = 1
		}
		if (symbol === endOfBlock) break
		if (size >= blockMax) throw new Error('The bzip2 block is larger than declared')
		const index = symbol - 1
		const moved = mtf[index]!
		mtf.copyWithin(1, 0, index)
		mtf[0] = moved
		const byte = used[moved]!
		counts[byte] = counts[byte]! + 1
		tt[size++] = byte
	}
	if (origPtr >= size && size > 0) throw new Error('Bad bzip2 origin pointer')
	const starts = new Int32Array(256)
	let sum = 0
	for (let index = 0; index < 256; index += 1) {
		starts[index] = sum
		sum += counts[index]!
	}
	for (let index = 0; index < size; index += 1) {
		const byte = tt[index]! & 0xff
		const slot = starts[byte]!
		starts[byte] = slot + 1
		tt[slot] = tt[slot]! | (index << 8)
	}
	output.ensure(size * 2)
	let position = size === 0 ? 0 : tt[origPtr]! >> 8
	let run = 0
	let last = -1
	for (let index = 0; index < size; index += 1) {
		const entry = tt[position]!
		const byte = entry & 0xff
		position = entry >> 8
		if (run === 4) {
			output.ensure(byte + 1)
			output.bytes.fill(last, output.length, output.length + byte)
			output.length += byte
			run = 0
			last = -1
			continue
		}
		if (byte === last) {
			run += 1
		} else {
			run = 1
			last = byte
		}
		if (output.length >= output.bytes.length) output.ensure(1)
		output.bytes[output.length++] = byte
	}
}

/** Decodes one or more concatenated bzip2 streams, checking every block's CRC. */
export function bunzip2(input: Uint8Array): Uint8Array {
	const output = new Output(input.length * 4)
	const bits = new BitReader(input)
	const tt = new Int32Array(900000)
	let streams = 0
	while (!bits.exhausted) {
		if (bits.read(8) !== 0x42 || bits.read(8) !== 0x5a || bits.read(8) !== 0x68) {
			if (streams > 0) break
			throw new Error('Not bzip2 data')
		}
		const level = bits.read(8) - 0x30
		if (level < 1 || level > 9) throw new Error('Bad bzip2 block size')
		const blockMax = level * 100000
		for (; ;) {
			const high = bits.read(24)
			const low = bits.read(24)
			if (high === BLOCK_MAGIC_HIGH && low === BLOCK_MAGIC_LOW) {
				const stored = bits.read(16) * 65536 + bits.read(16)
				const before = output.length
				tt.fill(0, 0, blockMax)
				decodeBlock(bits, blockMax, output, tt)
				if (blockCrc(output.bytes, before, output.length) !== stored) throw new Error('A bzip2 block failed its checksum')
			} else if (high === END_MAGIC_HIGH && low === END_MAGIC_LOW) {
				bits.read(16)
				bits.read(16)
				bits.align()
				break
			} else {
				throw new Error('Bad bzip2 block marker')
			}
		}
		streams += 1
	}
	return output.result()
}
