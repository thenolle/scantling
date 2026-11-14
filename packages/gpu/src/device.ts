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
	if (batch > 1 && spec.strideC < m * n) throw new RangeError(`The output stride ${spec.strideC} is smaller than one result (${m * n})`)
	const needs = (offset: number, stride: number, size: number): number => offset + (batch - 1) * stride + size
	const a = needs(spec.offsetA ?? 0, spec.strideA, m * k)
	const b = needs(spec.offsetB ?? 0, spec.strideB, k * n)
	const c = needs(spec.offsetC ?? 0, spec.strideC, m * n)
	if (a > lengths.a) throw new RangeError(`Operand A has ${lengths.a} elements but the product reads up to ${a}`)
	if (b > lengths.b) throw new RangeError(`Operand B has ${lengths.b} elements but the product reads up to ${b}`)
	if (c > lengths.c) throw new RangeError(`The output has ${lengths.c} elements but the product writes up to ${c}`)
}
