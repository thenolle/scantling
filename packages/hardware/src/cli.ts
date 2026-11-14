import { computeBudget } from './budget.ts'
import { formatBytes } from './format.ts'
import { detectHardware } from './profile.ts'

function readPaths(args: readonly string[]): string[] {
	const paths: string[] = []
	args.forEach((arg, index) => {
		const value = args[index + 1]
		if (arg === '--path' && value !== undefined) paths.push(value)
	})
	return paths.length > 0 ? paths : [process.cwd()]
}

const args = Bun.argv.slice(2)
const profile = await detectHardware({ storagePaths: readPaths(args), measureStorage: args.includes('--storage') })
const budget = computeBudget(profile)

if (args.includes('--json')) {
	console.log(JSON.stringify({ profile, budget }, null, 2))
} else {
	const { cpu, memory, gpu, runtime } = profile
	console.log(`Runtime  Bun ${runtime.bunVersion} on ${runtime.osName} (${runtime.osRelease})`)
	console.log(`CPU      ${cpu.model} · ${cpu.architecture} · ${cpu.logicalCores} logical cores`)
	console.log(`Memory   ${formatBytes(memory.totalBytes)} total · ${formatBytes(memory.availableBytes)} available`)
	for (const disk of profile.disks) {
		const speed = disk.writeMegabytesPerSecond === undefined ? '' : ` · write ${disk.writeMegabytesPerSecond.toFixed(0)} MiB/s`
		console.log(`Disk     ${disk.path} · ${formatBytes(disk.totalBytes)} total · ${formatBytes(disk.freeBytes)} free${speed}`)
	}
	if (gpu.devices.length === 0) console.log(`GPU      none detected (${gpu.reason ?? 'no devices'})`)
	for (const device of gpu.devices) {
		const compute = device.computeCapability === undefined ? '' : ` · compute ${device.computeCapability}`
		const cuda = gpu.cudaDriverVersion === undefined ? '' : ` · CUDA driver ${gpu.cudaDriverVersion}`
		console.log(`GPU      ${device.name} · ${formatBytes(device.vramTotalBytes)} VRAM (${formatBytes(device.vramFreeBytes)} free) · driver ${device.driverVersion}${compute}${cuda}`)
	}
	console.log(`Budget   ${budget.workerThreads} workers · ${formatBytes(budget.trainingMemoryBytes)} training memory · ${formatBytes(budget.inferenceMemoryBytes)} inference memory · accelerator ${budget.accelerator}`)
	for (const note of budget.notes) console.log(`Note     ${note}`)
}
