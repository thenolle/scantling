import { freemem } from 'node:os'

export interface UsageSummary {
	readonly samples: number
	readonly peakRssBytes: number
	readonly minAvailableBytes: number
	readonly durationMs: number
}

/**
 * Samples process memory on an interval and keeps the peak.
 * Bun workers share the process, so resident memory covers them too.
 */
export class ResourceMonitor {
	readonly #intervalMs: number
	#timer: ReturnType<typeof setInterval> | undefined
	#startedAt = 0
	#samples = 0
	#peakRss = 0
	#minAvailable = Number.POSITIVE_INFINITY

	constructor(intervalMs = 250) {
		this.#intervalMs = intervalMs
	}

	/** Takes one sample immediately. */
	sample(): void {
		this.#samples += 1
		this.#peakRss = Math.max(this.#peakRss, process.memoryUsage().rss)
		this.#minAvailable = Math.min(this.#minAvailable, freemem())
	}

	/** Starts periodic sampling. Calling it twice has no effect. */
	start(): void {
		if (this.#timer !== undefined) return
		this.#startedAt = performance.now()
		this.sample()
		this.#timer = setInterval(() => this.sample(), this.#intervalMs)
	}

	/** Stops sampling and returns the summary. */
	stop(): UsageSummary {
		if (this.#timer !== undefined) clearInterval(this.#timer)
		this.#timer = undefined
		this.sample()
		return {
			samples: this.#samples,
			peakRssBytes: this.#peakRss,
			minAvailableBytes: this.#minAvailable,
			durationMs: performance.now() - this.#startedAt
		}
	}
}
