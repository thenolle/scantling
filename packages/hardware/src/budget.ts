import { GIB, formatBytes } from './format.ts'
import type { HardwareProfile } from './types.ts'

export interface ResourceLimits {
	/** Largest share of total RAM that jobs may use, in (0, 1]. */
	readonly memoryFraction: number
	/** Memory always left to the operating system and the interface. */
	readonly reserveBytes: number
	readonly maxWorkers: number
	/** Logical cores left free for the server and the browser. */
	readonly reservedCores: number
	/** Free disk space that must remain after a job writes its output. */
	readonly minFreeDiskBytes: number
}

export const DEFAULT_LIMITS: ResourceLimits = {
	memoryFraction: 0.6,
	reserveBytes: 3 * GIB,
	maxWorkers: 8,
	reservedCores: 2,
	minFreeDiskBytes: 10 * GIB
}

export interface ResourceBudget {
	readonly limits: ResourceLimits
	readonly workerThreads: number
	/** Memory that training and inference together must stay within. */
	readonly usableMemoryBytes: number
	readonly trainingMemoryBytes: number
	readonly inferenceMemoryBytes: number
	/** Free disk space above the configured minimum on the tightest volume, when known. */
	readonly diskHeadroomBytes?: number
	readonly accelerator: 'cpu'
	readonly notes: readonly string[]
}

export class ResourceExceededError extends Error {
	readonly resource: 'memory' | 'disk'
	readonly requiredBytes: number
	readonly budgetedBytes: number

	constructor(resource: 'memory' | 'disk', requiredBytes: number, budgetedBytes: number, label: string) {
		super(`${label} needs ${formatBytes(requiredBytes)} of ${resource} but only ${formatBytes(budgetedBytes)} is budgeted`)
		this.name = 'ResourceExceededError'
		this.resource = resource
		this.requiredBytes = requiredBytes
		this.budgetedBytes = budgetedBytes
	}
}

/**
 * Merges overrides into the defaults and validates every value.
 */
export function resolveLimits(overrides: Partial<ResourceLimits> = {}): ResourceLimits {
	const limits = { ...DEFAULT_LIMITS, ...overrides }
	if (!(limits.memoryFraction > 0 && limits.memoryFraction <= 1)) throw new RangeError('memoryFraction must be in (0, 1]')
	if (!(limits.reserveBytes >= 0)) throw new RangeError('reserveBytes must not be negative')
	if (!Number.isInteger(limits.maxWorkers) || limits.maxWorkers < 1) throw new RangeError('maxWorkers must be an integer of at least 1')
	if (!Number.isInteger(limits.reservedCores) || limits.reservedCores < 0) throw new RangeError('reservedCores must be a non-negative integer')
	if (!(limits.minFreeDiskBytes >= 0)) throw new RangeError('minFreeDiskBytes must not be negative')
	return limits
}

/**
 * Derives what jobs may use right now from a hardware snapshot.
 * The accelerator is always `cpu`: a detected GPU is reported in the notes but not used.
 */
export function computeBudget(profile: HardwareProfile, overrides: Partial<ResourceLimits> = {}): ResourceBudget {
	const limits = resolveLimits(overrides)
	const { totalBytes, availableBytes } = profile.memory
	const usable = Math.max(0, Math.min(availableBytes - limits.reserveBytes, totalBytes * limits.memoryFraction))
	const workerThreads = Math.max(1, Math.min(profile.cpu.logicalCores - limits.reservedCores, limits.maxWorkers))
	const freeSpaces = profile.disks.map((disk) => disk.freeBytes - limits.minFreeDiskBytes)
	const notes: string[] = []
	if (usable < 2 * GIB) notes.push('Less than 2 GiB of memory is available to jobs; only micro configurations are advisable.')
	if (freeSpaces.some((space) => space < 0)) notes.push('A checked volume has less free space than the configured minimum.')
	if (profile.disks.length === 0) notes.push('Disk space could not be read, so disk checks are skipped.')
	if (profile.gpu.devices.length > 0) notes.push('A GPU was detected, but no verified GPU backend exists yet, so jobs run on the CPU.')
	return {
		limits,
		workerThreads,
		usableMemoryBytes: usable,
		trainingMemoryBytes: usable,
		inferenceMemoryBytes: Math.min(usable, totalBytes * 0.25),
		...(freeSpaces.length === 0 ? {} : { diskHeadroomBytes: Math.max(0, Math.min(...freeSpaces)) }),
		accelerator: 'cpu',
		notes
	}
}

/**
 * Throws `ResourceExceededError` when a job would need more memory than the training budget.
 * Call it before starting a job so an oversized job is refused instead of crashing the process.
 */
export function assertMemoryFits(budget: ResourceBudget, requiredBytes: number, label: string): void {
	if (requiredBytes > budget.trainingMemoryBytes) throw new ResourceExceededError('memory', requiredBytes, budget.trainingMemoryBytes, label)
}

/**
 * Throws `ResourceExceededError` when a job would write more than the available disk headroom.
 * Does nothing when free space is unknown.
 */
export function assertDiskFits(budget: ResourceBudget, requiredBytes: number, label: string): void {
	if (budget.diskHeadroomBytes !== undefined && requiredBytes > budget.diskHeadroomBytes) throw new ResourceExceededError('disk', requiredBytes, budget.diskHeadroomBytes, label)
}
