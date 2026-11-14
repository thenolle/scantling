import { gemmModuleBytes, simdProbeBytes } from './wasm.ts'

/** Whether the runtime accepts WebAssembly SIMD at all. */
export const simdSupported: boolean = typeof WebAssembly === 'object' && WebAssembly.validate(simdProbeBytes())

/** Whether the SIMD matrix-multiply module validated. Differs from `simdSupported` only if the module itself is wrong. */
export const gemmValid: boolean = simdSupported && WebAssembly.validate(gemmModuleBytes())

/** Products with fewer multiply-adds than this stay on the scalar kernels, where copying would cost more than it saves. */
const MIN_MACS = 16_384

let enabled = gemmValid && process.env['SCANTLING_SIMD'] !== '0'

interface Engine {
	readonly memory: WebAssembly.Memory
	readonly gemm: (a: number, b: number, c: number, m: number, k: number, n: number) => void
}

let engine: Engine | undefined

function load(): Engine {
	if (engine === undefined) {
		const instance = new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array(gemmModuleBytes())))
		engine = { memory: instance.exports['mem'] as WebAssembly.Memory, gemm: instance.exports['gemm'] as Engine['gemm'] }
	}
	return engine
}

/** Turns the SIMD path on or off. Has no effect when it is not supported. */
export function setSimdEnabled(value: boolean): void {
	enabled = value && gemmValid
}

/** True when a product with these dimensions should use the SIMD kernel. */
export function simdActive(m: number, k: number, n: number): boolean {
	return enabled && m >= 4 && m * k * n >= MIN_MACS
}

const align16 = (bytes: number): number => Math.ceil(bytes / 16) * 16

/**
 * `C = op(A) * op(B)` for one matrix pair on the SIMD kernel. Operands are copied into WebAssembly memory,
 * padded with zeros to whole 4-row and 8-column blocks (transposed operands are transposed during the copy),
 * and the result is copied back. Accumulation is in float32.
 */
export function gemmSimd(a: Float32Array, ao: number, transposeA: boolean, b: Float32Array, bo: number, transposeB: boolean, c: Float32Array, co: number, m: number, k: number, n: number): void {
	const { memory, gemm } = load()
	const paddedM = Math.ceil(m / 4) * 4
	const paddedN = Math.ceil(n / 8) * 8
	const aPointer = 0
	const bPointer = align16(paddedM * k * 4)
	const cPointer = align16(bPointer + k * paddedN * 4)
	const total = cPointer + paddedM * paddedN * 4
	const needed = Math.ceil(total / 65536)
	if (memory.buffer.byteLength / 65536 < needed) memory.grow(needed - memory.buffer.byteLength / 65536)
	const inputA = new Float32Array(memory.buffer, aPointer, paddedM * k)
	const inputB = new Float32Array(memory.buffer, bPointer, k * paddedN)
	const output = new Float32Array(memory.buffer, cPointer, paddedM * paddedN)
	if (transposeA) {
		for (let p = 0; p < k; p += 1) for (let i = 0; i < m; i += 1) inputA[i * k + p] = a[ao + p * m + i]!
	} else {
		inputA.set(a.subarray(ao, ao + m * k))
	}
	inputA.fill(0, m * k)
	if (transposeB) {
		inputB.fill(0)
		for (let j = 0; j < n; j += 1) for (let p = 0; p < k; p += 1) inputB[p * paddedN + j] = b[bo + j * k + p]!
	} else if (paddedN === n) {
		inputB.set(b.subarray(bo, bo + k * n))
	} else {
		for (let p = 0; p < k; p += 1) {
			inputB.set(b.subarray(bo + p * n, bo + (p + 1) * n), p * paddedN)
			inputB.fill(0, p * paddedN + n, (p + 1) * paddedN)
		}
	}
	gemm(aPointer, bPointer, cPointer, paddedM, k, paddedN)
	if (paddedN === n) c.set(output.subarray(0, m * n), co)
	else for (let i = 0; i < m; i += 1) c.set(output.subarray(i * paddedN, i * paddedN + n), co + i * n)
}
