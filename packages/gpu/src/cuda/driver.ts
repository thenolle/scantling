import { CString, FFIType, dlopen, ptr } from 'bun:ffi'
import type { Pointer } from 'bun:ffi'

/** The driver library or a usable device is missing. Not an error in the code. */
export class CudaUnavailableError extends Error {
	constructor(reason: string) {
		super(`CUDA is unavailable: ${reason}`)
		this.name = 'CudaUnavailableError'
	}
}

/** A CUDA driver call returned a failure code. */
export class CudaError extends Error {
	constructor(readonly code: number, readonly call: string, detail: string) {
		super(`${call} failed (${detail})`)
		this.name = 'CudaError'
	}
}

function openLibrary() {
	return dlopen(process.platform === 'win32' ? 'nvcuda.dll' : 'libcuda.so.1', {
		cuInit: { args: [FFIType.u32], returns: FFIType.i32 },
		cuDriverGetVersion: { args: [FFIType.ptr], returns: FFIType.i32 },
		cuDeviceGetCount: { args: [FFIType.ptr], returns: FFIType.i32 },
		cuDeviceGet: { args: [FFIType.ptr, FFIType.i32], returns: FFIType.i32 },
		cuDeviceGetName: { args: [FFIType.ptr, FFIType.i32, FFIType.i32], returns: FFIType.i32 },
		cuDeviceTotalMem_v2: { args: [FFIType.ptr, FFIType.i32], returns: FFIType.i32 },
		cuDeviceGetAttribute: { args: [FFIType.ptr, FFIType.i32, FFIType.i32], returns: FFIType.i32 },
		cuDevicePrimaryCtxRetain: { args: [FFIType.ptr, FFIType.i32], returns: FFIType.i32 },
		cuDevicePrimaryCtxRelease_v2: { args: [FFIType.i32], returns: FFIType.i32 },
		cuCtxSetCurrent: { args: [FFIType.u64], returns: FFIType.i32 },
		cuCtxSynchronize: { args: [], returns: FFIType.i32 },
		cuMemAlloc_v2: { args: [FFIType.ptr, FFIType.u64], returns: FFIType.i32 },
		cuMemFree_v2: { args: [FFIType.u64], returns: FFIType.i32 },
		cuMemcpyHtoD_v2: { args: [FFIType.u64, FFIType.ptr, FFIType.u64], returns: FFIType.i32 },
		cuMemcpyDtoH_v2: { args: [FFIType.ptr, FFIType.u64, FFIType.u64], returns: FFIType.i32 },
		cuMemsetD32_v2: { args: [FFIType.u64, FFIType.u32, FFIType.u64], returns: FFIType.i32 },
		cuMemGetInfo_v2: { args: [FFIType.ptr, FFIType.ptr], returns: FFIType.i32 },
		cuModuleLoadDataEx: { args: [FFIType.ptr, FFIType.ptr, FFIType.u32, FFIType.ptr, FFIType.ptr], returns: FFIType.i32 },
		cuModuleGetFunction: { args: [FFIType.ptr, FFIType.u64, FFIType.ptr], returns: FFIType.i32 },
		cuModuleUnload: { args: [FFIType.u64], returns: FFIType.i32 },
		cuLaunchKernel: { args: [FFIType.u64, FFIType.u32, FFIType.u32, FFIType.u32, FFIType.u32, FFIType.u32, FFIType.u32, FFIType.u32, FFIType.u64, FFIType.ptr, FFIType.ptr], returns: FFIType.i32 },
		cuEventCreate: { args: [FFIType.ptr, FFIType.u32], returns: FFIType.i32 },
		cuEventRecord: { args: [FFIType.u64, FFIType.u64], returns: FFIType.i32 },
		cuEventSynchronize: { args: [FFIType.u64], returns: FFIType.i32 },
		cuEventElapsedTime: { args: [FFIType.ptr, FFIType.u64, FFIType.u64], returns: FFIType.i32 },
		cuEventDestroy_v2: { args: [FFIType.u64], returns: FFIType.i32 },
		cuGetErrorName: { args: [FFIType.i32, FFIType.ptr], returns: FFIType.i32 },
		cuGetErrorString: { args: [FFIType.i32, FFIType.ptr], returns: FFIType.i32 }
	})
}

export type CudaSymbols = ReturnType<typeof openLibrary>['symbols']

/** One kernel argument, packed into the layout `cuLaunchKernel` expects. */
export type KernelArgument =
	| { readonly type: 'u64'; readonly value: bigint }
	| { readonly type: 's64'; readonly value: bigint }
	| { readonly type: 'u32'; readonly value: number }
	| { readonly type: 's32'; readonly value: number }
	| { readonly type: 'f32'; readonly value: number }

const ATTRIBUTE_MULTIPROCESSORS = 16
const ATTRIBUTE_CAPABILITY_MAJOR = 75
const ATTRIBUTE_CAPABILITY_MINOR = 76

export interface CudaDeviceProperties {
	readonly index: number
	readonly name: string
	readonly computeCapability: string
	readonly totalMemoryBytes: number
	readonly multiprocessors: number
}

function text(bytes: Uint8Array): string {
	const end = bytes.indexOf(0)
	return new TextDecoder().decode(bytes.subarray(0, end === -1 ? bytes.length : end))
}

/** Thin, checked access to the CUDA driver API. Loaded once per process. */
export class CudaApi {
	static #instance: CudaApi | undefined
	readonly symbols: CudaSymbols
	readonly driverVersion: string

	private constructor(symbols: CudaSymbols) {
		this.symbols = symbols
		const version = new Int32Array(1)
		this.check(symbols.cuInit(0), 'cuInit')
		this.check(symbols.cuDriverGetVersion(version), 'cuDriverGetVersion')
		this.driverVersion = `${Math.floor(version[0]! / 1000)}.${Math.floor((version[0]! % 1000) / 10)}`
	}

	static load(): CudaApi {
		if (CudaApi.#instance === undefined) {
			let library: ReturnType<typeof openLibrary>
			try { library = openLibrary() } catch (error) { throw new CudaUnavailableError(`the driver library could not be loaded (${error instanceof Error ? error.message : String(error)})`) }
			CudaApi.#instance = new CudaApi(library.symbols)
		}
		return CudaApi.#instance
	}

	describe(code: number): string {
		const name = new BigUint64Array(1)
		const message = new BigUint64Array(1)
		const label = this.symbols.cuGetErrorName(code, name) === 0 && name[0] !== 0n ? String(new CString(Number(name[0]) as Pointer)) : `code ${code}`
		const detail = this.symbols.cuGetErrorString(code, message) === 0 && message[0] !== 0n ? String(new CString(Number(message[0]) as Pointer)) : ''
		return detail === '' ? label : `${label}: ${detail}`
	}

	check(status: number, call: string): void {
		if (status !== 0) throw new CudaError(status, call, this.describe(status))
	}

	deviceCount(): number {
		const out = new Int32Array(1)
		this.check(this.symbols.cuDeviceGetCount(out), 'cuDeviceGetCount')
		return out[0]!
	}

	deviceHandle(index: number): number {
		const out = new Int32Array(1)
		this.check(this.symbols.cuDeviceGet(out, index), 'cuDeviceGet')
		return out[0]!
	}

	properties(index: number): CudaDeviceProperties {
		const handle = this.deviceHandle(index)
		const name = new Uint8Array(256)
		const memory = new BigUint64Array(1)
		this.check(this.symbols.cuDeviceGetName(name, name.length, handle), 'cuDeviceGetName')
		this.check(this.symbols.cuDeviceTotalMem_v2(memory, handle), 'cuDeviceTotalMem')
		const attribute = (id: number): number => {
			const out = new Int32Array(1)
			this.check(this.symbols.cuDeviceGetAttribute(out, id, handle), 'cuDeviceGetAttribute')
			return out[0]!
		}
		return {
			index,
			name: text(name),
			computeCapability: `${attribute(ATTRIBUTE_CAPABILITY_MAJOR)}.${attribute(ATTRIBUTE_CAPABILITY_MINOR)}`,
			totalMemoryBytes: Number(memory[0]!),
			multiprocessors: attribute(ATTRIBUTE_MULTIPROCESSORS)
		}
	}

	/** Compiles PTX with the driver. A compile error carries the driver's own log. */
	loadModule(source: string): bigint {
		const image = new TextEncoder().encode(`${source}\0`)
		const infoLog = new Uint8Array(8192)
		const errorLog = new Uint8Array(8192)
		const options = Int32Array.of(3, 4, 5, 6)
		const values = BigUint64Array.of(BigInt(ptr(infoLog)), BigInt(infoLog.length), BigInt(ptr(errorLog)), BigInt(errorLog.length))
		const module = new BigUint64Array(1)
		const status = this.symbols.cuModuleLoadDataEx(module, image, 4, options, values)
		if (status !== 0) throw new CudaError(status, 'cuModuleLoadDataEx', `${this.describe(status)}\n${text(errorLog)}${text(infoLog)}`)
		return module[0]!
	}

	getFunction(module: bigint, name: string): bigint {
		const out = new BigUint64Array(1)
		this.check(this.symbols.cuModuleGetFunction(out, module, new TextEncoder().encode(`${name}\0`)), `cuModuleGetFunction(${name})`)
		return out[0]!
	}

	launch(kernel: bigint, grid: readonly [number, number, number], block: readonly [number, number, number], args: readonly KernelArgument[]): void {
		const bytes = new Uint8Array(args.length * 8)
		const view = new DataView(bytes.buffer)
		args.forEach((argument, index) => {
			const at = index * 8
			if (argument.type === 'u64') view.setBigUint64(at, argument.value, true)
			else if (argument.type === 's64') view.setBigInt64(at, argument.value, true)
			else if (argument.type === 'u32') view.setUint32(at, argument.value, true)
			else if (argument.type === 's32') view.setInt32(at, argument.value, true)
			else view.setFloat32(at, argument.value, true)
		})
		const base = BigInt(ptr(bytes))
		const pointers = new BigUint64Array(args.length)
		for (let index = 0; index < args.length; index += 1) pointers[index] = base + BigInt(index * 8)
		this.check(this.symbols.cuLaunchKernel(kernel, grid[0], grid[1], grid[2], block[0], block[1], block[2], 0, 0n, pointers, null), 'cuLaunchKernel')
	}
}

/** Lists the CUDA devices the driver reports. Throws `CudaUnavailableError` or `CudaError` when none can be queried. */
export function listCudaDevices(): CudaDeviceProperties[] {
	const api = CudaApi.load()
	return Array.from({ length: api.deviceCount() }, (_, index) => api.properties(index))
}
