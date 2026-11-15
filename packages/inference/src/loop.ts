export interface LoopOptions {
	/** Longest repeating block to look for, in tokens. Defaults to 64. */
	readonly maxPeriod?: number
	/** How many back-to-back copies make a loop. Defaults to 4. */
	readonly repeats?: number
	/** The repeated stretch must cover at least this many tokens. Defaults to 24. */
	readonly minSpan?: number
}

/** True when the end of `tokens` is one block repeated back to back, which means generation has stopped making progress. */
export function endsInLoop(tokens: readonly number[], options: LoopOptions = {}): boolean {
	const maxPeriod = options.maxPeriod ?? 64
	const repeats = options.repeats ?? 4
	const minSpan = options.minSpan ?? 24
	const end = tokens.length
	for (let period = 1; period <= maxPeriod; period += 1) {
		const needed = Math.max(period * repeats, minSpan) - period
		if (end < needed + period) continue
		let run = 0
		for (let index = end - 1; index >= period && tokens[index] === tokens[index - period]; index -= 1) run += 1
		if (run >= needed) return true
	}
	return false
}
