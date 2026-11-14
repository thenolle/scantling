import { CpuDevice } from './cpu-device.ts'
import type { ComputeDevice, DeviceRequest } from './device.ts'
import { CudaDevice } from './cuda/device.ts'

/** `--device gpu` was requested and no usable GPU exists. */
export class GpuUnavailableError extends Error {
	constructor(readonly reason: string) {
		super(`GPU unavailable: ${reason}`)
		this.name = 'GpuUnavailableError'
	}
}

export interface DeviceSelection {
	readonly device: ComputeDevice
	readonly requested: DeviceRequest
	/** Set only when `auto` fell back to the CPU: why the GPU was not used. */
	readonly fallbackReason?: string
}

/**
 * Checks a device numerically before it is trusted: every transpose mode against a double-precision reference on
 * awkward sizes, then the elementwise operations. Throws with the first disagreement.
 */
export function selfTest(device: ComputeDevice): void {
	const [m, k, n] = [37, 29, 53]
	let seed = 12345
	const next = (): number => {
		seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0
		return seed / 2 ** 32 - 0.5
	}
	for (const transposeA of [false, true]) {
		for (const transposeB of [false, true]) {
			const a = Float32Array.from({ length: m * k }, next)
			const b = Float32Array.from({ length: k * n }, next)
			const bufferA = device.alloc(a.length)
			const bufferB = device.alloc(b.length)
			const bufferC = device.alloc(m * n)
			try {
				device.upload(bufferA, a)
				device.upload(bufferB, b)
				device.matmul(bufferA, bufferB, bufferC, { m, k, n, transposeA, transposeB, batch: 1, strideA: 0, strideB: 0, strideC: m * n })
				const c = new Float32Array(m * n)
				device.download(bufferC, c)
				for (let i = 0; i < m; i += 1) {
					for (let j = 0; j < n; j += 1) {
						let expected = 0
						let magnitude = 0
						for (let p = 0; p < k; p += 1) {
							const term = (transposeA ? a[p * m + i]! : a[i * k + p]!) * (transposeB ? b[j * k + p]! : b[p * n + j]!)
							expected += term
							magnitude += Math.abs(term)
						}
						if (!(Math.abs(c[i * n + j]! - expected) <= 1e-5 * (1 + magnitude))) {
							throw new Error(`self-test failed: matmul (transposeA=${transposeA}, transposeB=${transposeB}) element [${i}, ${j}] is ${c[i * n + j]} but should be ${expected}`)
						}
					}
				}
			} finally {
				device.free(bufferA)
				device.free(bufferB)
				device.free(bufferC)
			}
		}
	}
	const x = Float32Array.of(1, 2, 3, 4, 5)
	const [left, right, out] = [device.alloc(5), device.alloc(5), device.alloc(5)]
	try {
		device.upload(left, x)
		device.upload(right, x)
		device.add(left, right, out)
		device.scale(out, out, 0.5)
		const result = new Float32Array(5)
		device.download(out, result)
		if (result.some((value, index) => value !== x[index])) throw new Error(`self-test failed: add and scale returned [${Array.from(result).join(', ')}]`)
	} finally {
		device.free(left)
		device.free(right)
		device.free(out)
	}
}

/**
 * Picks a device. `cpu` always works. `gpu` requires CUDA to initialise and pass `selfTest`, and throws
 * `GpuUnavailableError` otherwise. `auto` tries the GPU the same way and falls back to the CPU, recording why.
 */
export function selectDevice(request: DeviceRequest = 'auto', index = 0): DeviceSelection {
	if (request === 'cpu') return { device: new CpuDevice(), requested: request }
	try {
		const device = CudaDevice.create(index)
		try {
			selfTest(device)
		} catch (error) {
			device.dispose()
			throw error
		}
		return { device, requested: request }
	} catch (error) {
		const reason = error instanceof Error ? error.message : String(error)
		if (request === 'gpu') throw new GpuUnavailableError(reason)
		return { device: new CpuDevice(), requested: request, fallbackReason: reason }
	}
}
