import * as fsPromises from 'node:fs/promises'
import { dirname, join } from 'node:path'
import type { DiskInfo } from './types.ts'

const MEGABYTE = 1024 * 1024

/**
 * Walks up from `path` to the nearest directory that exists.
 */
export async function nearestExistingDirectory(path: string): Promise<string> {
	let current = path
	for (; ;) {
		try {
			if ((await fsPromises.stat(current)).isDirectory()) return current
			return dirname(current)
		} catch {
			const parent = dirname(current)
			if (parent === current) return current
			current = parent
		}
	}
}

/**
 * Reads total and free space of the volume that holds `path`.
 * Returns `undefined` when the runtime cannot report it.
 */
export async function readDisk(path: string): Promise<DiskInfo | undefined> {
	try {
		const stats = await fsPromises.statfs(path)
		return { path, totalBytes: stats.blocks * stats.bsize, freeBytes: stats.bavail * stats.bsize }
	} catch {
		return undefined
	}
}

/**
 * Measures synced sequential write throughput by writing a temporary file in `directory`.
 * The file is always removed. Read speed is not measured because the operating system
 * cache would make the result misleading.
 */
export async function measureWriteSpeed(directory: string, megabytes = 64): Promise<number | undefined> {
	const file = join(directory, `.scantling-io-${process.pid}-${Date.now()}.tmp`)
	const chunk = new Uint8Array(MEGABYTE)
	for (let offset = 0; offset < chunk.length; offset += 65536) crypto.getRandomValues(chunk.subarray(offset, offset + 65536))
	try {
		const handle = await fsPromises.open(file, 'w')
		try {
			const started = performance.now()
			for (let index = 0; index < megabytes; index += 1) await handle.write(chunk)
			await handle.sync()
			return megabytes / ((performance.now() - started) / 1000)
		} finally {
			await handle.close()
		}
	} catch {
		return undefined
	} finally {
		await fsPromises.rm(file, { force: true })
	}
}
