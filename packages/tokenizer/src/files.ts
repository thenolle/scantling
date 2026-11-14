import { readdir } from 'node:fs/promises'
import { join } from 'node:path'

const SKIPPED = new Set(['node_modules', '.git', 'dist', 'data', 'vendor', '.cache'])

export interface TextFile {
	readonly path: string
	readonly text: string
}

async function readText(path: string, maxBytes: number): Promise<string | undefined> {
	const file = Bun.file(path)
	if (file.size === 0 || file.size > maxBytes) return undefined
	const bytes = new Uint8Array(await file.arrayBuffer())
	if (bytes.subarray(0, 8192).includes(0)) return undefined
	try { return new TextDecoder('utf-8', { fatal: true }).decode(bytes) } catch { return undefined }
}

/**
 * Yields the UTF-8 text files under the given paths. Only the paths you pass are read.
 * Binary files, files above `maxBytes` (1 MiB by default), lockfiles and common build or
 * dependency folders are skipped.
 */
export async function* collectTextFiles(paths: readonly string[], maxBytes = 1_048_576): AsyncGenerator<TextFile> {
	for (const root of paths) {
		const stack = [root]
		while (stack.length > 0) {
			const current = stack.pop()!
			const entries = await readdir(current, { withFileTypes: true }).catch(() => undefined)
			if (entries === undefined) {
				const text = await readText(current, maxBytes)
				if (text !== undefined) yield { path: current, text }
				continue
			}
			for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
				if (SKIPPED.has(entry.name) || entry.name.endsWith('.lock')) continue
				if (entry.isDirectory()) stack.push(join(current, entry.name))
				else if (entry.isFile()) {
					const text = await readText(join(current, entry.name), maxBytes)
					if (text !== undefined) yield { path: join(current, entry.name), text }
				}
			}
		}
	}
}
