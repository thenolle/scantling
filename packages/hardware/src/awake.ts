import { FFIType, dlopen, ptr } from 'bun:ffi'

const ES_CONTINUOUS = 0x80000000
const ES_SYSTEM_REQUIRED = 0x00000001
const ES_DISPLAY_REQUIRED = 0x00000002
const REQUEST_DISPLAY = 0
const REQUEST_SYSTEM = 1
const REQUEST_EXECUTION = 3
const INVALID_HANDLE = 0xffffffffffffffffn

export interface AwakeLock {
	readonly active: boolean
	readonly method: 'power-request' | 'execution-state' | 'none'
	release(): void
}

export interface AwakeOptions {
	readonly reason?: string
	/** Also keep the display on. Off by default, so the screen may turn off. */
	readonly display?: boolean
}

/**
 * Asks Windows not to sleep while this process runs. The display is left alone unless `display` is set.
 * The request ends on `release` or when the process exits. Other systems get an inactive lock.
 */
export function keepAwake(options: AwakeOptions = {}): AwakeLock {
	const none: AwakeLock = { active: false, method: 'none', release: () => undefined }
	if (process.platform !== 'win32') return none
	try {
		const { symbols } = dlopen('kernel32.dll', {
			SetThreadExecutionState: { args: [FFIType.u32], returns: FFIType.u32 },
			PowerCreateRequest: { args: [FFIType.ptr], returns: FFIType.u64 },
			PowerSetRequest: { args: [FFIType.u64, FFIType.u32], returns: FFIType.i32 },
			PowerClearRequest: { args: [FFIType.u64, FFIType.u32], returns: FFIType.i32 },
			CloseHandle: { args: [FFIType.u64], returns: FFIType.i32 }
		})
		const display = options.display === true
		const stateAccepted = symbols.SetThreadExecutionState(ES_CONTINUOUS + ES_SYSTEM_REQUIRED + (display ? ES_DISPLAY_REQUIRED : 0)) !== 0
		const reason = Buffer.from(`${options.reason ?? 'Scantling is running a long job'}\0`, 'utf16le')
		const context = new BigUint64Array(4)
		context[0] = 1n << 32n
		context[1] = BigInt(ptr(reason))
		const handle = symbols.PowerCreateRequest(ptr(context))
		const valid = handle !== 0n && handle !== INVALID_HANDLE
		const types = [REQUEST_SYSTEM, REQUEST_EXECUTION, ...(display ? [REQUEST_DISPLAY] : [])]
		const taken = valid ? types.filter((type) => symbols.PowerSetRequest(handle, type) !== 0) : []
		let released = false
		return {
			active: stateAccepted || taken.length > 0,
			method: taken.length > 0 ? 'power-request' : stateAccepted ? 'execution-state' : 'none',
			release: () => {
				if (released) return
				released = true
				for (const type of taken) symbols.PowerClearRequest(handle, type)
				if (valid) symbols.CloseHandle(handle)
				symbols.SetThreadExecutionState(ES_CONTINUOUS)
				void [reason, context]
			}
		}
	} catch {
		return none
	}
}
