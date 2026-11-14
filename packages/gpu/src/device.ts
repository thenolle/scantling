/** The backend a buffer lives on. */
export type DeviceKind = 'cpu' | 'cuda'

/** What the user asked for: a specific backend, or the best one that works. */
export type DeviceRequest = 'cpu' | 'gpu' | 'auto'

export type Precision = 'f32' | 'f16' | 'bf16'

export interface DeviceInfo {
	readonly kind: DeviceKind
	readonly name: string
	readonly index: number
	readonly totalMemoryBytes: number
	/** Precisions this backend computes in today. Only implemented and tested ones are listed. */
	readonly precisions: readonly Precision[]
	readonly computeCapability?: string
	readonly driverVersion?: string
	readonly multiprocessors?: number
}

export interface DeviceMemory {
	readonly freeBytes: number
	readonly totalBytes: number
	/** Bytes this process holds through the device wrapper. */
	readonly allocatedBytes: number
}

/** Opaque float32 storage owned by exactly one device. */
export interface DeviceBuffer {
	readonly device: DeviceKind
	/** Number of float32 elements. */
	readonly length: number
	readonly byteLength: number
	readonly disposed: boolean
}

/**
 * `C = alpha * op(A) * op(B) + beta * C` for `batch` matrices. `A` is stored `[m, k]` (or `[k, m]` when
 * transposed) and `B` as `[k, n]` (or `[n, k]`); `C` is `[m, n]`. Strides count elements between consecutive
 * matrices of the batch; a stride of 0 reuses one matrix for the whole batch. With `beta` of 0 the old
 * contents of `C` are never read.
 */
export interface MatmulSpec {
	readonly m: number
	readonly k: number
	readonly n: number
	readonly transposeA: boolean
	readonly transposeB: boolean
	readonly batch: number
	readonly strideA: number
	readonly strideB: number
	readonly strideC: number
	readonly offsetA?: number
	readonly offsetB?: number
	readonly offsetC?: number
	readonly alpha?: number
	readonly beta?: number
	/** Elements between consecutive rows of the stored matrices. They default to tightly packed. */
	readonly lda?: number
	readonly ldb?: number
	readonly ldc?: number
}

export interface ComputeDevice {
	readonly info: DeviceInfo
	alloc(length: number): DeviceBuffer
	free(buffer: DeviceBuffer): void
	upload(target: DeviceBuffer, source: Float32Array, targetOffset?: number): void
	download(source: DeviceBuffer, target: Float32Array, sourceOffset?: number): void
	fill(target: DeviceBuffer, value: number): void
	add(a: DeviceBuffer, b: DeviceBuffer, out: DeviceBuffer): void
	scale(x: DeviceBuffer, out: DeviceBuffer, alpha: number): void
	matmul(a: DeviceBuffer, b: DeviceBuffer, c: DeviceBuffer, spec: MatmulSpec): void
	/** Waits for all queued work and reports any error it hit. */
	synchronize(): void
	/** Runs `work` and returns the milliseconds the device spent on what it queued, measured after synchronising. */
	measure(work: () => void): number
	memory(): DeviceMemory
	dispose(): void
}

/** Validates a matmul specification against the sizes of its three buffers. Throws `RangeError`. */
export function checkMatmul(spec: MatmulSpec, lengths: { readonly a: number; readonly b: number; readonly c: number }): void {
	for (const [name, value] of [['m', spec.m], ['k', spec.k], ['n', spec.n], ['batch', spec.batch]] as const) {
		if (!Number.isInteger(value) || value < 1) throw new RangeError(`matmul ${name} must be a positive integer, received ${value}`)
	}
	const { m, k, n, batch } = spec
	if (batch > 1 && spec.strideC === 0) throw new RangeError('A batch of products needs a non-zero output stride')
	const check = (label: string, lengthOfBuffer: number, offset: number, stride: number, rows: number, cols: number, ld: number): void => {
		if (!Number.isInteger(ld) || ld < cols) throw new RangeError(`The leading dimension ${ld} of ${label} is smaller than its ${cols} columns`)
		const reach = offset + (batch - 1) * stride + (rows - 1) * ld + cols
		if (reach > lengthOfBuffer) throw new RangeError(`${label} has ${lengthOfBuffer} elements but the product reaches ${reach}`)
	}
	check('operand A', lengths.a, spec.offsetA ?? 0, spec.strideA, spec.transposeA ? k : m, spec.transposeA ? m : k, spec.lda ?? (spec.transposeA ? m : k))
	check('operand B', lengths.b, spec.offsetB ?? 0, spec.strideB, spec.transposeB ? n : k, spec.transposeB ? k : n, spec.ldb ?? (spec.transposeB ? k : n))
	check('the output', lengths.c, spec.offsetC ?? 0, spec.strideC, m, n, spec.ldc ?? n)
}
