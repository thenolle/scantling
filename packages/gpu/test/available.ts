import { test } from 'bun:test'
import { CudaDevice } from '../src/index.ts'

const reason = ((): string | undefined => {
	try {
		CudaDevice.create().dispose()
		return undefined
	} catch (error) {
		return error instanceof Error ? error.message : String(error)
	}
})()

if (reason !== undefined) console.log(`GPU tests skipped: ${reason}`)

/** `test` when a usable GPU exists, `test.skip` otherwise. */
export const gpu = reason === undefined ? test : test.skip
