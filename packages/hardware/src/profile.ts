import { measureWriteSpeed, nearestExistingDirectory, readDisk } from './disk.ts'
import { detectGpu } from './gpu.ts'
import { detectCpu, detectMemory, detectRuntime } from './system.ts'
import type { DiskInfo, HardwareProfile } from './types.ts'

export interface DetectOptions {
	/** Paths whose volumes are inspected. Defaults to the current directory. */
	readonly storagePaths?: readonly string[]
	/** Writes a temporary 64 MiB file per path to measure throughput. Off by default. */
	readonly measureStorage?: boolean
}

async function detectDisks(paths: readonly string[], measure: boolean): Promise<DiskInfo[]> {
	const disks: DiskInfo[] = []
	for (const path of paths) {
		const directory = await nearestExistingDirectory(path)
		const disk = await readDisk(directory)
		if (disk === undefined) continue
		const speed = measure ? await measureWriteSpeed(directory) : undefined
		disks.push({ ...disk, path, ...(speed === undefined ? {} : { writeMegabytesPerSecond: speed }) })
	}
	return disks
}

/**
 * Collects a snapshot of the machine: processor, memory, volumes, NVIDIA GPUs and runtime.
 * Nothing is cached, so call it again for current free memory and VRAM.
 */
export async function detectHardware(options: DetectOptions = {}): Promise<HardwareProfile> {
	const paths = options.storagePaths ?? [process.cwd()]
	const [gpu, disks] = await Promise.all([detectGpu(), detectDisks(paths, options.measureStorage === true)])
	return {
		cpu: detectCpu(),
		memory: detectMemory(),
		disks,
		gpu,
		runtime: detectRuntime(),
		detectedAt: new Date().toISOString()
	}
}
