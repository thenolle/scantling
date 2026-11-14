export const LEVELS = ['debug', 'info', 'warn', 'error'] as const

export type LogLevel = (typeof LEVELS)[number]

export interface LogEntry {
	readonly time: string
	readonly level: LogLevel
	readonly scope: string
	readonly message: string
	readonly fields: Readonly<Record<string, unknown>>
}

/**
 * Numeric severity, higher is more severe.
 */
export function levelRank(level: LogLevel): number {
	return LEVELS.indexOf(level)
}

export function isLevel(value: string): value is LogLevel {
	return (LEVELS as readonly string[]).includes(value)
}
