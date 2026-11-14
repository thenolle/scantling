import { Tensor, planMatmul } from '@scantling/tensor'
import type { MatmulOptions } from '@scantling/tensor'
import type { ComputeDevice, DeviceBuffer } from './device.ts'

const sizeOfShape = (shape: readonly number[]): number => shape.reduce((total, dimension) => total * dimension, 1)

/**
 * A float32 tensor that lives on a device. It owns its buffer: call `dispose()` when done. It is a different
 * type from `Tensor`, so CPU-only operations cannot receive it by accident, and operations between tensors
 * of different devices throw.
 */
export class DeviceTensor {
	readonly dtype = 'f32' as const

	constructor(readonly device: ComputeDevice, readonly buffer: DeviceBuffer, readonly shape: readonly number[]) {
		if (buffer.length !== sizeOfShape(shape)) throw new RangeError(`Shape [${shape.join(', ')}] needs ${sizeOfShape(shape)} values but the buffer holds ${buffer.length}`)
	}

	get rank(): number {
		return this.shape.length
	}

	get size(): number {
		return this.buffer.length
	}

	get disposed(): boolean {
		return this.buffer.disposed
	}

	dispose(): void {
		if (!this.buffer.disposed) this.device.free(this.buffer)
	}
}

function live(tensor: DeviceTensor): void {
	if (tensor.disposed) throw new Error('This device tensor has been disposed')
}

/** Copies a CPU tensor to the device. */
export function toDevice(device: ComputeDevice, tensor: Tensor): DeviceTensor {
	const buffer = device.alloc(tensor.size)
	try {
		device.upload(buffer, tensor.data)
	} catch (error) {
		device.free(buffer)
		throw error
	}
	return new DeviceTensor(device, buffer, tensor.shape)
}

/** A device tensor of zeros. */
export function deviceZeros(device: ComputeDevice, shape: readonly number[]): DeviceTensor {
	const buffer = device.alloc(sizeOfShape(shape))
	device.fill(buffer, 0)
	return new DeviceTensor(device, buffer, shape)
}

/** Copies a device tensor back to the CPU. */
export function toHost(tensor: DeviceTensor): Tensor {
	live(tensor)
	const data = new Float32Array(tensor.size)
	tensor.device.download(tensor.buffer, data)
	return new Tensor(tensor.shape, data)
}

/** The constant distance between consecutive matrices of a batch, or undefined when the offsets are irregular. */
function regularStride(offsets: readonly number[]): number | undefined {
	if (offsets.length <= 1) return 0
	if (offsets[0] !== 0) return undefined
	const stride = offsets[1]!
	return offsets.every((offset, index) => offset === index * stride) ? stride : undefined
}

/**
 * Matrix product on the device with the same semantics as the CPU `matmul`: batch broadcasting and transpose
 * flags. Regular batches run in one launch; irregular broadcasts run one matrix at a time.
 */
export function deviceMatmul(a: DeviceTensor, b: DeviceTensor, options: MatmulOptions = {}): DeviceTensor {
	if (a.device !== b.device) throw new Error('The operands of matmul live on different devices; move one with toDevice() first')
	live(a)
	live(b)
	const plan = planMatmul(a, b, options)
	const device = a.device
	const out = device.alloc(plan.count * plan.m * plan.n)
	const strideA = regularStride(plan.offsetsA)
	const strideB = regularStride(plan.offsetsB)
	const base = { m: plan.m, k: plan.k, n: plan.n, transposeA: options.transposeA === true, transposeB: options.transposeB === true, strideC: plan.m * plan.n }
	try {
		if (strideA !== undefined && strideB !== undefined) {
			device.matmul(a.buffer, b.buffer, out, { ...base, batch: plan.count, strideA, strideB })
		} else {
			for (let index = 0; index < plan.count; index += 1) device.matmul(a.buffer, b.buffer, out, { ...base, batch: 1, strideA: 0, strideB: 0, offsetA: plan.offsetsA[index]!, offsetB: plan.offsetsB[index]!, offsetC: index * plan.m * plan.n })
		}
	} catch (error) {
		device.free(out)
		throw error
	}
	return new DeviceTensor(device, out, plan.shape)
}
