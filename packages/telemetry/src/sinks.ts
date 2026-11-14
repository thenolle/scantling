import { appendFileSync, existsSync, mkdirSync, renameSync, rmSync, statSync } from 'node:fs'
import { join } from 'node:path'
import type { LogEntry, LogLevel } from './levels.ts'

export interface LogSink {
	write(entry: LogEntry): void
}

const COLORS: Readonly<Record<LogLevel, string>> = {
	debug: '\x1b[90m',
	info: '\x1b[36m',
	warn: '\x1b[33m',
	error: '\x1b[31m'
}

/**
 * Human-readable output. Warnings and errors go to stderr.
 */
export class ConsoleSink implements LogSink {
	readonly #color: boolean

	constructor(color: boolean = process.stdout.isTTY === true) {
		this.#color = color
	}

	write(entry: LogEntry): void {
		const level = entry.level.toUpperCase().padEnd(5)
		const head = this.#color ? `${COLORS[entry.level]}${level}\x1b[0m` : level
		const scope = entry.scope === '' ? '' : `${entry.scope} `
		const fields = Object.keys(entry.fields).length > 0 ? ` ${JSON.stringify(entry.fields)}` : ''
		const stream = entry.level === 'warn' || entry.level === 'error' ? process.stderr : process.stdout
		stream.write(`${entry.time.slice(11, 23)} ${head} ${scope}${entry.message}${fields}\n`)
	}
}

/**
 * Keeps the most recent entries in memory, for example to show in the interface.
 */
export class MemorySink implements LogSink {
	readonly #capacity: number
	readonly #buffer: LogEntry[] = []

	constructor(capacity = 500) {
		this.#capacity = capacity
	}

	write(entry: LogEntry): void {
		this.#buffer.push(entry)
		if (this.#buffer.length > this.#capacity) this.#buffer.shift()
	}

	entries(): readonly LogEntry[] {
		return [...this.#buffer]
	}

	clear(): void {
		this.#buffer.length = 0
	}
}

export interface FileSinkOptions {
	readonly directory: string
	readonly fileName?: string
	readonly maxBytes: number
	readonly maxFiles: number
}

const encoder = new TextEncoder()

/**
 * Appends JSON lines to a file and rotates it at `maxBytes`, keeping `maxFiles` archives
 * named `<file>.1` (newest) to `<file>.<maxFiles>` (oldest).
 * Writes are synchronous: log volume is low and ordering matters more than throughput.
 */
export class FileSink implements LogSink {
	readonly #path: string
	readonly #options: FileSinkOptions
	#size: number

	constructor(options: FileSinkOptions) {
		mkdirSync(options.directory, { recursive: true })
		this.#options = options
		this.#path = join(options.directory, options.fileName ?? 'scantling.log')
		this.#size = existsSync(this.#path) ? statSync(this.#path).size : 0
	}

	write(entry: LogEntry): void {
		const line = `${JSON.stringify(entry)}\n`
		const bytes = encoder.encode(line).length
		if (this.#size > 0 && this.#size + bytes > this.#options.maxBytes) this.#rotate()
		appendFileSync(this.#path, line)
		this.#size += bytes
	}

	#rotate(): void {
		const archive = (index: number): string => `${this.#path}.${index}`
		rmSync(archive(this.#options.maxFiles), { force: true })
		for (let index = this.#options.maxFiles - 1; index >= 1; index -= 1) {
			if (existsSync(archive(index))) renameSync(archive(index), archive(index + 1))
		}
		renameSync(this.#path, archive(1))
		this.#size = 0
	}
}
