import type { LogEntry, LogLevel } from './levels.ts'
import { levelRank } from './levels.ts'
import { redactString, sanitize } from './redact.ts'
import type { LogSink } from './sinks.ts'

export interface LoggerState {
	level: LogLevel
	dropped: number
	readonly sinks: readonly LogSink[]
}

export interface LoggerOptions {
	readonly sinks: readonly LogSink[]
	readonly level?: LogLevel
	readonly scope?: string
	readonly fields?: Readonly<Record<string, unknown>>
}

type Fields = Readonly<Record<string, unknown>>

/**
 * A structured logger. Every entry is sanitised before any sink sees it, and a failing sink
 * never throws into application code; it is counted in `droppedEntries` instead.
 */
export class Logger {
	readonly #state: LoggerState
	readonly #scope: string
	readonly #fields: Fields

	constructor(state: LoggerState, scope = '', fields: Fields = {}) {
		this.#state = state
		this.#scope = scope
		this.#fields = fields
	}

	get droppedEntries(): number {
		return this.#state.dropped
	}

	setLevel(level: LogLevel): void {
		this.#state.level = level
	}

	debug(message: string, fields?: Fields): void {
		this.#emit('debug', message, fields)
	}

	info(message: string, fields?: Fields): void {
		this.#emit('info', message, fields)
	}

	warn(message: string, fields?: Fields): void {
		this.#emit('warn', message, fields)
	}

	error(message: string, fields?: Fields): void {
		this.#emit('error', message, fields)
	}

	/**
	 * Creates a logger for a sub-system. Scopes are joined with dots and share level and sinks.
	 */
	child(scope: string, fields: Fields = {}): Logger {
		return new Logger(this.#state, this.#scope === '' ? scope : `${this.#scope}.${scope}`, { ...this.#fields, ...fields })
	}

	#emit(level: LogLevel, message: string, fields: Fields = {}): void {
		if (levelRank(level) < levelRank(this.#state.level)) return
		const entry: LogEntry = { time: new Date().toISOString(), level, scope: this.#scope, message: redactString(message), fields: sanitize({ ...this.#fields, ...fields }) as Record<string, unknown> }
		for (const sink of this.#state.sinks) try { sink.write(entry) } catch { this.#state.dropped += 1 }
	}
}

export function createLogger(options: LoggerOptions): Logger {
	return new Logger({ level: options.level ?? 'info', dropped: 0, sinks: options.sinks }, options.scope ?? '', options.fields ?? {})
}
