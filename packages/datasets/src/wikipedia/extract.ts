import { closeSync, fstatSync, mkdirSync, openSync, readSync, writeSync } from 'node:fs'
import { join } from 'node:path'
import { bunzip2 } from './bunzip2.ts'
import { articlesOf } from './wikitext.ts'

const HEADER = Buffer.from('BZh')
const BLOCK = Buffer.from([0x31, 0x41, 0x59, 0x26, 0x53, 0x59])
/** Articles in a `.prose` file are separated by this control character. The first line of each is its title. */
export const SEPARATOR = '\u001e'

export interface ExtractOptions {
	readonly dump: string
	readonly shard: number
	readonly shards: number
	readonly out: string
	readonly windowBytes?: number
	readonly maxShardBytes?: number
	readonly onProgress?: (stats: ExtractStats) => void
}

export interface ExtractStats {
	streams: number
	articles: number
	bytes: number
	files: number
}

export function streamStarts(buffer: Buffer): number[] {
	const starts: number[] = []
	let at = buffer.indexOf(HEADER)
	while (at !== -1) {
		const level = buffer[at + 3]
		if (level !== undefined && level >= 0x31 && level <= 0x39 && buffer.compare(BLOCK, 0, 6, at + 4, at + 10) === 0) starts.push(at)
		at = buffer.indexOf(HEADER, at + 3)
	}
	return starts
}

function readAt(fd: number, position: number, length: number): Buffer {
	const buffer = Buffer.allocUnsafe(length)
	let filled = 0
	while (filled < length) {
		const got = readSync(fd, buffer, filled, length - filled, position + filled)
		if (got === 0) break
		filled += got
	}
	return buffer.subarray(0, filled)
}

/** Decodes every bzip2 stream that starts inside this shard's slice of the dump and writes its articles. */
export function extractRange(options: ExtractOptions): ExtractStats {
	const { dump, shard, shards, out } = options
	const windowBytes = options.windowBytes ?? 64 * 1024 * 1024
	const maxShardBytes = options.maxShardBytes ?? 64 * 1024 * 1024
	mkdirSync(out, { recursive: true })
	const fd = openSync(dump, 'r')
	const size = fstatSync(fd).size
	const rangeStart = Math.floor((size * shard) / shards)
	const rangeEnd = shard === shards - 1 ? size : Math.floor((size * (shard + 1)) / shards)
	const stats: ExtractStats = { streams: 0, articles: 0, bytes: 0, files: 0 }
	let outFd = -1
	let outBytes = 0
	const write = (text: string): void => {
		if (outFd === -1 || outBytes >= maxShardBytes) {
			if (outFd !== -1) closeSync(outFd)
			stats.files += 1
			outFd = openSync(join(out, `wiki-${String(shard).padStart(2, '0')}-${String(stats.files).padStart(4, '0')}.prose`), 'w')
			outBytes = 0
		}
		const data = Buffer.from(text + SEPARATOR, 'utf8')
		writeSync(outFd, data)
		outBytes += data.length
		stats.bytes += data.length
	}
	try {
		let first = -1
		for (let cursor = rangeStart; cursor < size && first === -1; cursor += 1024 * 1024) {
			const found = streamStarts(readAt(fd, cursor, 1024 * 1024 + 16)).find((start) => start < 1024 * 1024)
			if (found !== undefined) first = cursor + found
		}
		let position = first
		outer: while (position !== -1 && position < rangeEnd) {
			const window = readAt(fd, position, windowBytes)
			const starts = streamStarts(window)
			const atEnd = position + window.length >= size
			const usable = atEnd ? starts.length : starts.length - 1
			if (usable <= 0 && !atEnd) throw new Error('A bzip2 stream is larger than the read window')
			for (let k = 0; k < usable; k += 1) {
				const start = starts[k]!
				if (position + start >= rangeEnd) break outer
				const bytes = bunzip2(window.subarray(start, k + 1 < starts.length ? starts[k + 1]! : window.length))
				for (const article of articlesOf(Buffer.from(bytes.buffer, bytes.byteOffset, bytes.length).toString('utf8'))) {
					write(article)
					stats.articles += 1
				}
				stats.streams += 1
				options.onProgress?.(stats)
			}
			if (atEnd) break
			position += starts[usable]!
		}
	} finally {
		if (outFd !== -1) closeSync(outFd)
		closeSync(fd)
	}
	return stats
}
