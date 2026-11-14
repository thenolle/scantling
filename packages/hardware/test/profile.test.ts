import { expect, test } from 'bun:test'
import { tmpdir } from 'node:os'
import { ResourceMonitor } from '../src/monitor.ts'
import { detectHardware } from '../src/profile.ts'

test('detects the machine it runs on', async () => {
	const profile = await detectHardware({ storagePaths: [tmpdir()] })
	expect(profile.cpu.logicalCores).toBeGreaterThanOrEqual(1)
	expect(profile.memory.totalBytes).toBeGreaterThan(0)
	expect(['nvidia-smi', 'unavailable']).toContain(profile.gpu.source)
	expect(Array.isArray(profile.disks)).toBe(true)
})

test('the monitor records a peak', async () => {
	const monitor = new ResourceMonitor(10)
	monitor.start()
	const ballast = new Float32Array(4 * 1024 * 1024)
	ballast.fill(1)
	await Bun.sleep(40)
	const summary = monitor.stop()
	expect(summary.samples).toBeGreaterThanOrEqual(2)
	expect(summary.peakRssBytes).toBeGreaterThan(0)
})
