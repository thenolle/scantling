import type { Logger } from './logger.ts'

export interface Timed<T> {
	readonly value: T
	readonly milliseconds: number
}

/**
 * Runs `work` and measures wall-clock time with the monotonic clock.
 */
export async function timed<T>(work: () => Promise<T> | T): Promise<Timed<T>> {
	const started = performance.now()
	const value = await work()
	return { value, milliseconds: performance.now() - started }
}

/**
 * Like `timed`, and logs the duration at debug level (or at error level when `work` throws).
 */
export async function logTimed<T>(logger: Logger, label: string, work: () => Promise<T> | T): Promise<T> {
	const started = performance.now()
	try {
		const value = await work()
		logger.debug(`${label} finished`, { milliseconds: Math.round(performance.now() - started) })
		return value
	} catch (error) {
		logger.error(`${label} failed`, { milliseconds: Math.round(performance.now() - started), error })
		throw error
	}
}

/**
 * Items per second for a count processed in `milliseconds`.
 */
export function throughput(count: number, milliseconds: number): number {
	return milliseconds > 0 ? (count * 1000) / milliseconds : 0
}
