export interface Sample {
	readonly label: string
	readonly runs: number
	readonly medianMs: number
	readonly minMs: number
}

export interface MeasureOptions {
	readonly warmup?: number
	readonly minMs?: number
	readonly minRuns?: number
	readonly maxRuns?: number
}

export function median(values: readonly number[]): number {
	const sorted = [...values].sort((a, b) => a - b)
	const middle = Math.floor(sorted.length / 2)
	return sorted.length % 2 === 1 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2
}

/**
 * Times `work` with the monotonic clock. It runs a few warm-up iterations, then repeats until at
 * least `minMs` of measured time and `minRuns` runs have accumulated (or `maxRuns` is reached).
 */
export async function measure(label: string, work: () => unknown, options: MeasureOptions = {}): Promise<Sample> {
	const warmup = options.warmup ?? 2
	const minMs = options.minMs ?? 300
	const minRuns = options.minRuns ?? 5
	const maxRuns = options.maxRuns ?? 200
	for (let index = 0; index < warmup; index += 1) await work()
	const times: number[] = []
	let total = 0
	while (times.length < maxRuns && (times.length < minRuns || total < minMs)) {
		const started = performance.now()
		await work()
		const elapsed = performance.now() - started
		times.push(elapsed)
		total += elapsed
	}
	return { label, runs: times.length, medianMs: median(times), minMs: Math.min(...times) }
}
