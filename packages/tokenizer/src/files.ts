import { readdir } from 'node:fs/promises'
import { join } from 'node:path'

const SKIPPED = new Set(['node_modules', '.git', 'dist', 'data', 'vendor', '.cache'])

export interface TextFile {
	readonly path: string
	readonly text: string
}

/** Decides from the path and size alone whether a file is worth reading, so rejected files cost no read. */
export type FileFilter = (path: string, size: number) => boolean

async function readText(path: string, maxBytes: number, accept: FileFilter | undefined): Promise<string | undefined> {
	const file = Bun.file(path)
	if (file.size === 0 || file.size > maxBytes) return undefined
	if (accept !== undefined && !accept(path, file.size)) return undefined
	const bytes = new Uint8Array(await file.arrayBuffer())
	if (bytes.subarray(0, 8192).includes(0)) return undefined
	try { return new TextDecoder('utf-8', { fatal: true }).decode(bytes) } catch { return undefined }
}

async function* walk(paths: readonly string[]): AsyncGenerator<string> {
	for (const root of paths) {
		const stack = [root]
		while (stack.length > 0) {
			const current = stack.pop()!
			const entries = await readdir(current, { withFileTypes: true }).catch(() => undefined)
			if (entries === undefined) {
				yield current
				continue
			}
			for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
				if (SKIPPED.has(entry.name) || entry.name.endsWith('.lock')) continue
				if (entry.isDirectory()) stack.push(join(current, entry.name))
				else if (entry.isFile()) yield join(current, entry.name)
			}
		}
	}
}

/**
 * Yields the UTF-8 text files under the given paths, in a fixed order. Only the paths you pass are read.
 * Binary files, files above `maxBytes` (1 MiB by default), lockfiles, common build or dependency folders and
 * files rejected by `accept` are skipped. Up to `concurrency` files are read at the same time.
 */
export async function* collectTextFiles(paths: readonly string[], maxBytes = 1_048_576, accept?: FileFilter, concurrency = 16): AsyncGenerator<TextFile> {
	const read = (batch: string[]): Promise<(TextFile | undefined)[]> =>
		Promise.all(batch.map(async (path) => {
			const text = await readText(path, maxBytes, accept).catch(() => undefined)
			return text === undefined ? undefined : { path, text }
		}))
	let batch: string[] = []
	for await (const path of walk(paths)) {
		batch.push(path)
		if (batch.length < concurrency) continue
		for (const file of await read(batch)) if (file !== undefined) yield file
		batch = []
	}
	for (const file of await read(batch)) if (file !== undefined) yield file
}
