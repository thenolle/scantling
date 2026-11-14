import { arch, availableParallelism, cpus, freemem, platform, release, totalmem } from 'node:os'
import type { CpuInfo, MemoryInfo, RuntimeInfo } from './types.ts'

/**
 * Describes the processor. Only logical cores are reported because Bun exposes no
 * portable way to tell performance cores from efficiency cores.
 */
export function detectCpu(): CpuInfo {
	const first = cpus()[0]
	return {
		model: first?.model.trim() ?? 'unknown',
		architecture: arch(),
		logicalCores: availableParallelism(),
		speedMhz: first?.speed ?? 0
	}
}

/**
 * Reports total and currently available physical memory.
 */
export function detectMemory(): MemoryInfo {
	return { totalBytes: totalmem(), availableBytes: freemem() }
}

function osName(kind: string, version: string): string {
	if (kind !== 'win32') return kind
	const build = Number(version.split('.')[2])
	return Number.isFinite(build) && build >= 22000 ? 'Windows 11' : 'Windows'
}

/**
 * Reports the Bun version and operating system.
 */
export function detectRuntime(): RuntimeInfo {
	const kind = platform()
	return { bunVersion: Bun.version, platform: kind, osName: osName(kind, release()), osRelease: release() }
}
