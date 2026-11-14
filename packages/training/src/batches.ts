import type { Random } from '@scantling/tensor'

export interface Batch {
	readonly inputs: ArrayLike<number>
	readonly targets: ArrayLike<number>
	readonly batch: number
	readonly length: number
}

export interface BatchSource {
	next(): Batch
}

/**
 * Next-token windows over one token stream: the input is `tokens[s .. s+length)` and the target is
 * the same window shifted by one.
 */
export class TokenWindows {
	constructor(readonly tokens: Int32Array, readonly length: number) {
		if (!Number.isInteger(length) || length < 1) throw new RangeError('The window length must be a positive integer')
		if (tokens.length < length + 1) throw new RangeError(`Need at least ${length + 1} tokens, received ${tokens.length}`)
	}

	/** Number of possible window start positions. */
	get count(): number {
		return this.tokens.length - this.length
	}

	window(start: number): { inputs: Int32Array; targets: Int32Array } {
		return { inputs: this.tokens.slice(start, start + this.length), targets: this.tokens.slice(start + 1, start + this.length + 1) }
	}

	/** `batch` windows starting at random positions. */
	sample(batch: number, random: Random): Batch {
		const inputs = new Int32Array(batch * this.length)
		const targets = new Int32Array(batch * this.length)
		for (let row = 0; row < batch; row += 1) {
			const start = random.nextInt(this.count)
			inputs.set(this.tokens.subarray(start, start + this.length), row * this.length)
			targets.set(this.tokens.subarray(start + 1, start + this.length + 1), row * this.length)
		}
		return { inputs, targets, batch, length: this.length }
	}

	/** Deterministic, non-overlapping windows for validation: starts 0, length, 2*length, and so on. */
	evaluationBatches(batch: number, maxBatches: number): Batch[] {
		const starts: number[] = []
		for (let start = 0; start + this.length + 1 <= this.tokens.length; start += this.length) starts.push(start)
		const batches: Batch[] = []
		for (let first = 0; first < starts.length && batches.length < maxBatches; first += batch) {
			const rows = starts.slice(first, first + batch)
			const inputs = new Int32Array(rows.length * this.length)
			const targets = new Int32Array(rows.length * this.length)
			rows.forEach((start, row) => {
				inputs.set(this.tokens.subarray(start, start + this.length), row * this.length)
				targets.set(this.tokens.subarray(start + 1, start + this.length + 1), row * this.length)
			})
			batches.push({ inputs, targets, batch: rows.length, length: this.length })
		}
		return batches
	}
}

/** Endless random batches from a `TokenWindows`. */
export class RandomBatchSource implements BatchSource {
	constructor(private readonly windows: TokenWindows, private readonly batch: number, private readonly random: Random) { }

	next(): Batch {
		return this.windows.sample(this.batch, this.random)
	}
}
