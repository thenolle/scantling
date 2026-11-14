import { expect, test } from 'bun:test'
import { parseCudaVersion, parseNvidiaSmiCsv } from '../src/gpu.ts'

test('parses a row with compute capability', () => {
	const [gpu] = parseNvidiaSmiCsv('Example GPU, 8192, 7000, 500.00, 8.9\n', true)
	expect(gpu?.name).toBe('Example GPU')
	expect(gpu?.vramTotalBytes).toBe(8192 * 1024 * 1024)
	expect(gpu?.vramFreeBytes).toBe(7000 * 1024 * 1024)
	expect(gpu?.driverVersion).toBe('500.00')
	expect(gpu?.computeCapability).toBe('8.9')
})

test('parses rows without compute capability and skips malformed rows', () => {
	const gpus = parseNvidiaSmiCsv('Example GPU, 4096, 4000, 500.00\nnot a row\n', false)
	expect(gpus).toHaveLength(1)
	expect(gpus[0]?.computeCapability).toBeUndefined()
})

test('extracts the CUDA driver version', () => {
	expect(parseCudaVersion('| NVIDIA-SMI 500.00   Driver Version: 500.00   CUDA Version: 12.4 |')).toBe('12.4')
	expect(parseCudaVersion('no cuda here')).toBeUndefined()
})
