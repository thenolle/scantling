import { availableParallelism } from 'node:os'
import { join } from 'node:path'
import type { Tokenizer } from '@scantling/tokenizer'

export interface EncodeOptions {
	/** Worker threads. 0 encodes on the calling thread. Defaults to the cores minus two, at most 8. */
	readonly workers?: number
	/** Folder of cached encodings, keyed by tokenizer and exact text. Omit to disable the cache. */
	readonly cacheDirectory?: string
	readonly onProgress?: (done: number, total: number, cached: number) => void
}

export interface EncodeResult {
	readonly ids: Int32Array[]
	readonly cached: number
}

const BATCH_BYTES = 256 * 1024
const INLINE_BYTES = 1024 * 1024
const sha256 = (text: string): string => new Bun.CryptoHasher('sha256').update(text).digest('hex')

/** Encodes every text, in order. Cached results are reused and new ones are stored. */
export async function encodeAll(texts: readonly string[], tokenizer: Tokenizer, options: EncodeOptions = {}): Promise<EncodeResult> {
	const json = tokenizer.toJson()
	const root = options.cacheDirectory === undefined ? undefined : join(options.cacheDirectory, sha256(json).slice(0, 16))
	const keys = root === undefined ? [] : texts.map(sha256)
	const pathOf = (index: number): string => join(root!, keys[index]!.slice(0, 2), `${keys[index]}.bin`)
	const ids: (Int32Array | undefined)[] = new Array<Int32Array | undefined>(texts.length).fill(undefined)
	let cached = 0
	if (root !== undefined) {
		for (let from = 0; from < texts.length; from += 256) {
			const count = Math.min(256, texts.length - from)
			await Promise.all(Array.from({ length: count }, async (_, offset) => {
				const index = from + offset
				try {
					const file = Bun.file(pathOf(index))
					if (await file.exists()) {
						ids[index] = new Int32Array(await file.arrayBuffer())
						cached += 1
					}
				} catch {
					ids[index] = undefined
				}
			}))
		}
	}
	options.onProgress?.(cached, texts.length, cached)
	const missing = texts.map((_, index) => index).filter((index) => ids[index] === undefined).sort((a, b) => texts[b]!.length - texts[a]!.length)
	let done = cached
	const store = (index: number, result: Int32Array): void => {
		ids[index] = result
		done += 1
	}
	const missingBytes = missing.reduce((sum, index) => sum + texts[index]!.length, 0)
	const workers = options.workers ?? Math.max(1, Math.min(8, availableParallelism() - 2))
	if (workers === 0 || missingBytes < INLINE_BYTES) {
		for (const index of missing) {
			store(index, Int32Array.from(tokenizer.encode(texts[index]!)))
			options.onProgress?.(done, texts.length, cached)
		}
	} else {
		const batches: number[][] = []
		let current: number[] = []
		let bytes = 0
		for (const index of missing) {
			current.push(index)
			bytes += texts[index]!.length
			if (bytes >= BATCH_BYTES || current.length >= 64) {
				batches.push(current)
				current = []
				bytes = 0
			}
		}
		if (current.length > 0) batches.push(current)
		await runPool(json, batches, texts, workers, (indices, results) => {
			indices.forEach((index, position) => store(index, results[position]!))
			options.onProgress?.(done, texts.length, cached)
		})
	}
	if (root !== undefined) {
		for (let from = 0; from < missing.length; from += 256) await Promise.all(missing.slice(from, from + 256).map((index) => Bun.write(pathOf(index), ids[index]!)))
	}
	return { ids: ids as Int32Array[], cached }
}

async function runPool(json: string, batches: number[][], texts: readonly string[], size: number, onResult: (indices: number[], results: Int32Array[]) => void): Promise<void> {
	if (batches.length === 0) return
	const pool = Array.from({ length: Math.min(size, batches.length) }, () => new Worker(new URL('./encode-worker.ts', import.meta.url)))
	let next = 0
	let finished = 0
	try {
		await new Promise<void>((resolve, reject) => {
			const dispatch = (worker: Worker): void => {
				const id = next++
				if (id >= batches.length) {
					finished += 1
					if (finished === pool.length) resolve()
					return
				}
				worker.postMessage({ kind: 'job', id, texts: batches[id]!.map((index) => texts[index]!) })
			}
			for (const worker of pool) {
				worker.onerror = (event) => reject(new Error(event.message))
				worker.onmessage = (event: MessageEvent) => {
					const message = event.data as { kind: 'ready' } | { kind: 'done'; id: number; results: Int32Array[] }
					if (message.kind === 'done') onResult(batches[message.id]!, message.results)
					dispatch(worker)
				}
				worker.postMessage({ kind: 'init', json })
			}
		})
	} finally {
		for (const worker of pool) worker.terminate()
	}
}
