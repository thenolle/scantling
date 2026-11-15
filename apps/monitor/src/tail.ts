import { existsSync, statSync } from 'node:fs'
import type { LogRecord } from './types.ts'

function parse(line: string): LogRecord[] {
	try {
		const value = JSON.parse(line) as Partial<LogRecord>
		if (typeof value.time !== 'string' || typeof value.message !== 'string') return []
		return [{ time: value.time, level: value.level ?? 'info', scope: value.scope ?? '', message: value.message, fields: value.fields ?? {} }]
	} catch {
		return []
	}
}

/** Reads the lines appended to a JSON-lines log since the last call. A shorter file means it was rotated, so reading restarts. */
export class LogTail {
	readonly #path: string
	#offset = 0
	#carry = ''
	#decoder = new TextDecoder()

	constructor(path: string) {
		this.#path = path
	}

	async read(): Promise<LogRecord[]> {
		if (!existsSync(this.#path)) return []
		const size = statSync(this.#path).size
		if (size < this.#offset) {
			this.#offset = 0
			this.#carry = ''
			this.#decoder = new TextDecoder()
		}
		if (size === this.#offset) return []
		const bytes = await Bun.file(this.#path).slice(this.#offset, size).arrayBuffer()
		this.#offset = size
		const lines = (this.#carry + this.#decoder.decode(bytes, { stream: true })).split('\n')
		this.#carry = lines.pop() ?? ''
		return lines.flatMap(parse)
	}
}
