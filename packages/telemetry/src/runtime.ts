import type { LogLevel } from './levels.ts'
import type { Logger } from './logger.ts'
import { createLogger } from './logger.ts'
import type { LogSink } from './sinks.ts'
import { ConsoleSink, FileSink, MemorySink } from './sinks.ts'

export interface RuntimeLoggerOptions {
	readonly level: LogLevel
	readonly directory: string
	readonly toFile: boolean
	readonly maxFileBytes: number
	readonly maxFiles: number
	readonly console?: boolean
}

export interface RuntimeLogger {
	readonly logger: Logger
	/** The most recent entries, for the interface. */
	readonly recent: MemorySink
}

/**
 * Builds the application logger: console, optional rotating file, and an in-memory buffer.
 */
export function createRuntimeLogger(options: RuntimeLoggerOptions): RuntimeLogger {
	const recent = new MemorySink()
	const sinks: LogSink[] = [recent]
	if (options.console !== false) sinks.push(new ConsoleSink())
	if (options.toFile) sinks.push(new FileSink({ directory: options.directory, maxBytes: options.maxFileBytes, maxFiles: options.maxFiles }))
	return { logger: createLogger({ level: options.level, sinks }), recent }
}
