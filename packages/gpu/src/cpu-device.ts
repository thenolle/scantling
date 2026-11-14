import { cpus, freemem, totalmem } from 'node:os'
import { Tensor, matmul } from '@scantling/tensor'
import { checkMatmul } from './device.ts'
import type { ComputeDevice, DeviceBuffer, DeviceInfo, DeviceMemory, MatmulSpec } from './device.ts'

class CpuBuffer implements DeviceBuffer {
	readonly device = 'cpu' as const
	disposed = false

	constructor(readonly owner: CpuDevice, readonly data: Float32Array) { }

	get length(): number {
		return this.data.length
	}

	get byteLength(): number {
		return this.data.byteLength
	}
}

/**
 * The reference device. It implements the same interface as the GPU backend on top of the existing CPU kernels,
 * so code written against `ComputeDevice` runs everywhere and can be checked against it.
 */
export class CpuDevice implements ComputeDevice {
	readonly info: DeviceInfo = { kind: 'cpu', name: cpus()[0]?.model ?? 'CPU', index: 0, totalMemoryBytes: totalmem(), precisions: ['f32'] }
	#allocated = 0
	#disposed = false

	alloc(length: number): DeviceBuffer {
		this.#alive()
		if (!Number.isInteger(length) || length < 0) throw new RangeError(`A buffer needs a non-negative integer length, received ${length}`)
		this.#allocated += length * 4
		return new CpuBuffer(this, new Float32Array(length))
	}

	free(buffer: DeviceBuffer): void {
		const own = this.#own(buffer)
		own.disposed = true
		this.#allocated -= own.byteLength
	}

	upload(target: DeviceBuffer, source: Float32Array, targetOffset = 0): void {
		const own = this.#own(target)
		if (targetOffset < 0 || targetOffset + source.length > own.length) throw new RangeError('The upload does not fit in the buffer')
		own.data.set(source, targetOffset)
	}

	download(source: DeviceBuffer, target: Float32Array, sourceOffset = 0): void {
		const own = this.#own(source)
		if (sourceOffset < 0 || sourceOffset + target.length > own.length) throw new RangeError('The download reads past the end of the buffer')
		target.set(own.data.subarray(sourceOffset, sourceOffset + target.length))
	}

	fill(target: DeviceBuffer, value: number): void {
		this.#own(target).data.fill(value)
	}

	add(a: DeviceBuffer, b: DeviceBuffer, out: DeviceBuffer): void {
		const [x, y, z] = [this.#own(a), this.#own(b), this.#own(out)]
		if (x.length !== y.length || x.length !== z.length) throw new RangeError('add needs buffers of equal length')
		for (let index = 0; index < z.length; index += 1) z.data[index] = x.data[index]! + y.data[index]!
	}

	scale(x: DeviceBuffer, out: DeviceBuffer, alpha: number): void {
		const [source, target] = [this.#own(x), this.#own(out)]
		if (source.length !== target.length) throw new RangeError('scale needs buffers of equal length')
		for (let index = 0; index < target.length; index += 1) target.data[index] = source.data[index]! * alpha
	}

	matmul(a: DeviceBuffer, b: DeviceBuffer, c: DeviceBuffer, spec: MatmulSpec): void {
		const [left, right, out] = [this.#own(a), this.#own(b), this.#own(c)]
		if (out === left || out === right) throw new RangeError('The output of matmul must not alias an input')
		checkMatmul(spec, { a: left.length, b: right.length, c: out.length })
		const { m, k, n } = spec
		const alpha = spec.alpha ?? 1
		const beta = spec.beta ?? 0
		for (let index = 0; index < spec.batch; index += 1) {
			const aOffset = (spec.offsetA ?? 0) + index * spec.strideA
			const bOffset = (spec.offsetB ?? 0) + index * spec.strideB
			const cOffset = (spec.offsetC ?? 0) + index * spec.strideC
			const dense = (data: Float32Array, offset: number, rows: number, cols: number, ld: number): Float32Array => {
				if (ld === cols) return data.subarray(offset, offset + rows * cols)
				const packed = new Float32Array(rows * cols)
				for (let row = 0; row < rows; row += 1) packed.set(data.subarray(offset + row * ld, offset + row * ld + cols), row * cols)
				return packed
			}
			const [rowsA, colsA] = spec.transposeA ? [k, m] : [m, k]
			const [rowsB, colsB] = spec.transposeB ? [n, k] : [k, n]
			const product = matmul(
				new Tensor([rowsA, colsA], dense(left.data, aOffset, rowsA, colsA, spec.lda ?? colsA)),
				new Tensor([rowsB, colsB], dense(right.data, bOffset, rowsB, colsB, spec.ldb ?? colsB)),
				{ transposeA: spec.transposeA, transposeB: spec.transposeB }
			)
			const ldc = spec.ldc ?? n
			for (let row = 0; row < m; row += 1) {
				for (let col = 0; col < n; col += 1) {
					const at = cOffset + row * ldc + col
					out.data[at] = alpha * product.data[row * n + col]! + (beta === 0 ? 0 : beta * out.data[at]!)
				}
			}
		}
	}

	synchronize(): void { }

	measure(work: () => void): number {
		const start = performance.now()
		work()
		return performance.now() - start
	}

	memory(): DeviceMemory {
		return { freeBytes: freemem(), totalBytes: totalmem(), allocatedBytes: this.#allocated }
	}

	dispose(): void {
		this.#disposed = true
	}

	#alive(): void {
		if (this.#disposed) throw new Error('This device has been disposed')
	}

	#own(buffer: DeviceBuffer): CpuBuffer {
		this.#alive()
		if (!(buffer instanceof CpuBuffer) || buffer.owner !== this) throw new TypeError('This buffer belongs to a different device')
		if (buffer.disposed) throw new Error('This buffer has been freed')
		return buffer
	}
}
