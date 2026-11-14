import type { Shape } from './shape.ts'
import { Tensor } from './tensor.ts'

export interface RandomState {
	readonly words: readonly [number, number, number, number]
	readonly spare: number | null
}

function splitmix32(seed: number): () => number {
	let state = seed | 0
	return () => {
		state = (state + 0x9e3779b9) | 0
		let z = state
		z = Math.imul(z ^ (z >>> 16), 0x85ebca6b)
		z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35)
		return (z ^ (z >>> 16)) >>> 0
	}
}

const rotl = (value: number, shift: number): number => ((value << shift) | (value >>> (32 - shift))) >>> 0

/**
 * Seeded xoshiro128** generator. The same seed always yields the same sequence on every machine,
 * and `state()` / `restore()` allow a run to resume exactly where a checkpoint left off.
 */
export class Random {
	#s0 = 0
	#s1 = 0
	#s2 = 0
	#s3 = 0
	#spare: number | null = null

	constructor(seed: number) {
		const next = splitmix32(seed)
		this.#s0 = next()
		this.#s1 = next()
		this.#s2 = next()
		this.#s3 = next()
	}

	/** A uniformly distributed unsigned 32-bit integer. */
	nextUint32(): number {
		const result = Math.imul(rotl(Math.imul(this.#s1, 5), 7), 9) >>> 0
		const t = (this.#s1 << 9) >>> 0
		this.#s2 = (this.#s2 ^ this.#s0) >>> 0
		this.#s3 = (this.#s3 ^ this.#s1) >>> 0
		this.#s1 = (this.#s1 ^ this.#s2) >>> 0
		this.#s0 = (this.#s0 ^ this.#s3) >>> 0
		this.#s2 = (this.#s2 ^ t) >>> 0
		this.#s3 = rotl(this.#s3, 11)
		return result
	}

	/** A uniform number in [0, 1) with 53 bits of resolution. */
	next(): number {
		const high = this.nextUint32() >>> 5
		const low = this.nextUint32() >>> 6
		return (high * 67108864 + low) / 9007199254740992
	}

	/** A uniform integer in [0, bound) without modulo bias. */
	nextInt(bound: number): number {
		if (!Number.isInteger(bound) || bound < 1 || bound > 0x100000000) throw new RangeError(`Invalid bound ${bound}`)
		const limit = 0x100000000 - (0x100000000 % bound)
		let value = this.nextUint32()
		while (value >= limit) value = this.nextUint32()
		return value % bound
	}

	/** A standard normal sample (Box-Muller transform). */
	normal(): number {
		if (this.#spare !== null) {
			const value = this.#spare
			this.#spare = null
			return value
		}
		const radius = Math.sqrt(-2 * Math.log(1 - this.next()))
		const angle = 2 * Math.PI * this.next()
		this.#spare = radius * Math.sin(angle)
		return radius * Math.cos(angle)
	}

	/** Fisher-Yates shuffle, in place. */
	shuffle<T>(items: T[]): T[] {
		for (let index = items.length - 1; index > 0; index -= 1) {
			const other = this.nextInt(index + 1)
			const held = items[index]!
			items[index] = items[other]!
			items[other] = held
		}
		return items
	}

	/** A new generator seeded from this one, for independent streams such as workers. */
	split(): Random {
		return new Random(this.nextUint32())
	}

	state(): RandomState {
		return { words: [this.#s0, this.#s1, this.#s2, this.#s3], spare: this.#spare }
	}

	restore(state: RandomState): void {
		const [s0, s1, s2, s3] = state.words
		this.#s0 = s0
		this.#s1 = s1
		this.#s2 = s2
		this.#s3 = s3
		this.#spare = state.spare
	}
}

/** Tensor of independent normal samples `mean + std * z`. */
export function randomNormal(shape: Shape, random: Random, std = 1, mean = 0): Tensor {
	const tensor = Tensor.zeros(shape)
	for (let index = 0; index < tensor.size; index += 1) tensor.data[index] = mean + std * random.normal()
	return tensor
}

/** Tensor of independent uniform samples in [low, high). */
export function randomUniform(shape: Shape, random: Random, low = 0, high = 1): Tensor {
	const tensor = Tensor.zeros(shape)
	for (let index = 0; index < tensor.size; index += 1) tensor.data[index] = low + (high - low) * random.next()
	return tensor
}
