/**
 * Detects stop sequences in streamed text. Text that might still become a stop sequence is held back,
 * so a partial match is never emitted.
 */
export class StopMatcher {
	readonly #stops: readonly string[]
	#pending = ''

	constructor(stops: readonly string[]) {
		if (stops.some((stop) => stop === '')) throw new RangeError('Stop sequences must not be empty')
		this.#stops = stops
	}

	/** Returns the text that is safe to emit and whether a stop sequence was reached. */
	push(text: string): { readonly emit: string; readonly stopped: boolean } {
		this.#pending += text
		let cut = -1
		for (const stop of this.#stops) {
			const at = this.#pending.indexOf(stop)
			if (at !== -1 && (cut === -1 || at < cut)) cut = at
		}
		if (cut !== -1) {
			const emit = this.#pending.slice(0, cut)
			this.#pending = ''
			return { emit, stopped: true }
		}
		let hold = 0
		for (const stop of this.#stops) {
			for (let length = Math.min(stop.length - 1, this.#pending.length); length > hold; length -= 1) {
				if (this.#pending.endsWith(stop.slice(0, length))) {
					hold = length
					break
				}
			}
		}
		const emit = this.#pending.slice(0, this.#pending.length - hold)
		this.#pending = this.#pending.slice(this.#pending.length - hold)
		return { emit, stopped: false }
	}

	/** Releases anything still held back. */
	flush(): string {
		const rest = this.#pending
		this.#pending = ''
		return rest
	}
}
