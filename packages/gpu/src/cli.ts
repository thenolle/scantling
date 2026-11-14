import { listCudaDevices } from './cuda/driver.ts'
import type { DeviceRequest } from './device.ts'
import { selectDevice } from './select.ts'

const args = Bun.argv.slice(2)
const option = (name: string): string | undefined => (args.includes(name) ? args[args.indexOf(name) + 1] : undefined)
const request = (option('--device') ?? 'auto') as DeviceRequest
if (request !== 'cpu' && request !== 'gpu' && request !== 'auto') throw new Error(`--device must be cpu, gpu or auto, received "${request}"`)
const gib = (bytes: number): string => `${(bytes / 1_073_741_824).toFixed(1)} GiB`

try {
	for (const device of listCudaDevices()) console.log(`GPU ${device.index}        ${device.name} · compute ${device.computeCapability} · ${gib(device.totalMemoryBytes)} · ${device.multiprocessors} multiprocessors`)
} catch (error) {
	console.log(`CUDA         unavailable: ${error instanceof Error ? error.message : String(error)}`)
}

try {
	const selection = selectDevice(request, Number(option('--index') ?? 0))
	const { info } = selection.device
	const memory = selection.device.memory()
	console.log(`Requested    ${request}`)
	console.log(`Selected     ${info.kind} · ${info.name} · precisions ${info.precisions.join(', ')}${info.driverVersion === undefined ? '' : ` · driver CUDA ${info.driverVersion}`}`)
	if (selection.fallbackReason !== undefined) console.log(`Fallback     the GPU was not used: ${selection.fallbackReason}`)
	else if (info.kind === 'cuda') console.log('Self-test    passed: every matmul mode, add and scale match a double-precision reference')
	console.log(`Memory       ${gib(memory.freeBytes)} free of ${gib(memory.totalBytes)}`)
	selection.device.dispose()
} catch (error) {
	console.error(error instanceof Error ? error.message : String(error))
	process.exitCode = 1
}
