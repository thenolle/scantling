import { expect, test } from 'bun:test'
import { assertDiskFits, assertMemoryFits, computeBudget, resolveLimits, ResourceExceededError } from '../src/budget.ts'
import { GIB } from '../src/format.ts'
import type { HardwareProfile } from '../src/types.ts'

const profile: HardwareProfile = {
	cpu: { model: 'test', architecture: 'x64', logicalCores: 12, speedMhz: 2000 },
	memory: { totalBytes: 16 * GIB, availableBytes: 9 * GIB },
	disks: [{ path: '/data', totalBytes: 512 * GIB, freeBytes: 97 * GIB }],
	gpu: { source: 'unavailable', devices: [] },
	runtime: { bunVersion: '0', platform: 'linux', osName: 'linux', osRelease: '0' },
	detectedAt: '2026-01-01T00:00:00.000Z'
}

test('derives workers and memory from the profile', () => {
	const budget = computeBudget(profile)
	expect(budget.workerThreads).toBe(8)
	expect(budget.usableMemoryBytes).toBe(6 * GIB)
	expect(budget.inferenceMemoryBytes).toBe(4 * GIB)
	expect(budget.diskHeadroomBytes).toBe(87 * GIB)
	expect(budget.accelerator).toBe('cpu')
})

test('refuses jobs that exceed the budget', () => {
	const budget = computeBudget(profile)
	expect(() => assertMemoryFits(budget, 7 * GIB, 'job')).toThrow(ResourceExceededError)
	expect(() => assertMemoryFits(budget, 5 * GIB, 'job')).not.toThrow()
	expect(() => assertDiskFits(budget, 90 * GIB, 'job')).toThrow(ResourceExceededError)
})

test('never reports negative memory', () => {
	const tight = { ...profile, memory: { totalBytes: 4 * GIB, availableBytes: 1 * GIB } }
	expect(computeBudget(tight).usableMemoryBytes).toBe(0)
})

test('rejects invalid limits', () => {
	expect(() => resolveLimits({ memoryFraction: 0 })).toThrow(RangeError)
	expect(() => resolveLimits({ maxWorkers: 0 })).toThrow(RangeError)
})
