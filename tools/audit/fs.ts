import { readdir } from 'node:fs/promises'
import { join, relative } from 'node:path'

/**
 * Recursively yields regular files under `dir`, skipping ignored directory names.
 * Symbolic links are not followed.
 */
export async function* walk(directory: string, ignoredDirectories: ReadonlySet<string>): AsyncGenerator<string> {
	const entries = await readdir(directory, { withFileTypes: true })
	for (const entry of entries) {
		const full = join(directory, entry.name)
		if (entry.isDirectory()) {
			if (!ignoredDirectories.has(entry.name)) yield* walk(full, ignoredDirectories)
		} else if (entry.isFile()) {
			yield full
		}
	}
}

/**
 * Returns a repository-relative path using forward slashes on every platform.
 */
export function toPosix(root: string, file: string): string {
	return relative(root, file).split('\\').join('/')
}
