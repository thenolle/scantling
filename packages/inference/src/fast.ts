import { headDim } from '@scantling/nn'
import type { Transformer } from '@scantling/nn'
import { DenseProjection, Int8Projection } from './projection.ts'
import type { Projection } from './projection.ts'
import { quantizeRows } from './quantize.ts'

export type Precision = 'float32' | 'int8'

/** Anything that turns tokens into next-token logits one position at a time. */
export interface Engine {
	readonly length: number
	reset(): void
	/** Feeds tokens and returns the logits after the last one. The returned array may be reused by the next call. */
	feed(ids: readonly number[]): Float32Array
}

interface Layer {
	readonly attentionNorm: Float32Array
	readonly query: Projection
	readonly key: Projection
	readonly value: Projection
	readonly output: Projection
	readonly ffnNorm: Float32Array
	readonly gate: Projection
	readonly up: Projection
	readonly down: Projection
}

function transposed(values: Float32Array, rows: number, cols: number): Float32Array {
	const out = new Float32Array(values.length)
	for (let row = 0; row < rows; row += 1) for (let column = 0; column < cols; column += 1) out[column * rows + row] = values[row * cols + column]!
	return out
}

function rmsNorm(x: Float32Array, weight: Float32Array, out: Float32Array, eps: number): void {
	let total = 0
	for (let index = 0; index < x.length; index += 1) total += x[index]! * x[index]!
	const scale = 1 / Math.sqrt(total / x.length + eps)
	for (let index = 0; index < x.length; index += 1) out[index] = x[index]! * scale * weight[index]!
}

function dot(a: Float32Array, ao: number, b: Float32Array, bo: number, length: number): number {
	let s0 = 0
	let s1 = 0
	let s2 = 0
	let s3 = 0
	let index = 0
	for (; index + 4 <= length; index += 4) {
		s0 += a[ao + index]! * b[bo + index]!
		s1 += a[ao + index + 1]! * b[bo + index + 1]!
		s2 += a[ao + index + 2]! * b[bo + index + 2]!
		s3 += a[ao + index + 3]! * b[bo + index + 3]!
	}
	let tail = 0
	for (; index < length; index += 1) tail += a[ao + index]! * b[bo + index]!
	return s0 + s1 + (s2 + s3) + tail
}

/**
 * A single-sequence decoder with no per-token allocations: weights are stored transposed (one row per output),
 * the KV cache is preallocated, and every intermediate lives in a scratch buffer created once. With `int8`
 * precision every projection, including the output head, uses 8-bit weights with one scale per row.
 * Prompts are processed one token at a time, so prefill is slower than the batched tensor path.
 */
export class FastDecoder implements Engine {
	readonly model: Transformer
	readonly precision: Precision
	readonly #layers: Layer[] = []
	readonly #head: Projection
	readonly #embedding: Float32Array
	readonly #finalNorm: Float32Array
	readonly #keys: Float32Array[]
	readonly #values: Float32Array[]
	readonly #cos: Float32Array
	readonly #sin: Float32Array
	readonly #x: Float32Array
	readonly #h: Float32Array
	readonly #q: Float32Array
	readonly #k: Float32Array
	readonly #v: Float32Array
	readonly #context: Float32Array
	readonly #contextSum: Float64Array
	readonly #gate: Float32Array
	readonly #up: Float32Array
	readonly #scores: Float32Array
	readonly #logits: Float32Array
	#length = 0

	constructor(model: Transformer, precision: Precision = 'float32') {
		this.model = model
		this.precision = precision
		const { config } = model
		const dh = headDim(config)
		const half = dh / 2
		const projection = (weights: Float32Array, rows: number, cols: number): Projection => {
			const rowMajor = transposed(weights, rows, cols)
			return precision === 'int8' ? new Int8Projection(quantizeRows(rowMajor, cols, rows)) : new DenseProjection(rowMajor, cols, rows)
		}
		for (const block of model.blocks) {
			this.#layers.push({
				attentionNorm: block.attentionNorm.weight.value.data,
				query: projection(block.attention.query.weight.value.data, config.dim, config.heads * dh),
				key: projection(block.attention.key.weight.value.data, config.dim, config.kvHeads * dh),
				value: projection(block.attention.value.weight.value.data, config.dim, config.kvHeads * dh),
				output: projection(block.attention.output.weight.value.data, config.heads * dh, config.dim),
				ffnNorm: block.feedForwardNorm.weight.value.data,
				gate: projection(block.feedForward.gate.weight.value.data, config.dim, config.ffnHidden),
				up: projection(block.feedForward.up.weight.value.data, config.dim, config.ffnHidden),
				down: projection(block.feedForward.down.weight.value.data, config.ffnHidden, config.dim)
			})
		}
		this.#embedding = model.embedding.table.value.data
		const headWeights = model.head === undefined ? this.#embedding : transposed(model.head.weight.value.data, config.dim, config.vocabSize)
		this.#head = precision === 'int8' ? new Int8Projection(quantizeRows(headWeights, config.vocabSize, config.dim)) : new DenseProjection(headWeights, config.vocabSize, config.dim)
		this.#finalNorm = model.finalNorm.weight.value.data
		const cache = (): Float32Array => new Float32Array(config.kvHeads * config.maxSeqLen * dh)
		this.#keys = Array.from({ length: config.layers }, cache)
		this.#values = Array.from({ length: config.layers }, cache)
		this.#cos = new Float32Array(config.maxSeqLen * half)
		this.#sin = new Float32Array(config.maxSeqLen * half)
		for (let position = 0; position < config.maxSeqLen; position += 1) {
			for (let index = 0; index < half; index += 1) {
				const angle = position * Math.pow(config.ropeBase, -(2 * index) / dh)
				this.#cos[position * half + index] = Math.cos(angle)
				this.#sin[position * half + index] = Math.sin(angle)
			}
		}
		this.#x = new Float32Array(config.dim)
		this.#h = new Float32Array(Math.max(config.dim, config.ffnHidden))
		this.#q = new Float32Array(config.heads * dh)
		this.#k = new Float32Array(config.kvHeads * dh)
		this.#v = new Float32Array(config.kvHeads * dh)
		this.#context = new Float32Array(config.heads * dh)
		this.#contextSum = new Float64Array(dh)
		this.#gate = new Float32Array(config.ffnHidden)
		this.#up = new Float32Array(config.ffnHidden)
		this.#scores = new Float32Array(config.maxSeqLen)
		this.#logits = new Float32Array(config.vocabSize)
	}

	get length(): number {
		return this.#length
	}

	reset(): void {
		this.#length = 0
	}

	feed(ids: readonly number[]): Float32Array {
		if (ids.length === 0) throw new RangeError('feed needs at least one token')
		let logits = this.#logits
		for (const id of ids) logits = this.step(id)
		return logits
	}

	/** Processes one token and returns the shared logits buffer. Allocates nothing. */
	step(token: number): Float32Array {
		const { config } = this.model
		if (this.#length >= config.maxSeqLen) throw new RangeError(`The context window of ${config.maxSeqLen} tokens is full`)
		if (!Number.isInteger(token) || token < 0 || token >= config.vocabSize) throw new RangeError(`Token id ${token} is outside the vocabulary of ${config.vocabSize}`)
		const dh = headDim(config)
		const half = dh / 2
		const position = this.#length
		const groups = config.heads / config.kvHeads
		const scale = 1 / Math.sqrt(dh)
		const x = this.#x
		const h = this.#h.subarray(0, config.dim)
		x.set(this.#embedding.subarray(token * config.dim, (token + 1) * config.dim))
		for (let layerIndex = 0; layerIndex < config.layers; layerIndex += 1) {
			const layer = this.#layers[layerIndex]!
			rmsNorm(x, layer.attentionNorm, h, config.normEps)
			layer.query.apply(h, this.#q)
			layer.key.apply(h, this.#k)
			layer.value.apply(h, this.#v)
			this.#rotate(this.#q, config.heads, half, position)
			this.#rotate(this.#k, config.kvHeads, half, position)
			const keys = this.#keys[layerIndex]!
			const values = this.#values[layerIndex]!
			for (let kvHead = 0; kvHead < config.kvHeads; kvHead += 1) {
				const slot = (kvHead * config.maxSeqLen + position) * dh
				for (let index = 0; index < dh; index += 1) {
					keys[slot + index] = this.#k[kvHead * dh + index]!
					values[slot + index] = this.#v[kvHead * dh + index]!
				}
			}
			for (let head = 0; head < config.heads; head += 1) {
				const base = Math.floor(head / groups) * config.maxSeqLen * dh
				let top = Number.NEGATIVE_INFINITY
				for (let t = 0; t <= position; t += 1) {
					const score = dot(this.#q, head * dh, keys, base + t * dh, dh) * scale
					this.#scores[t] = score
					if (score > top) top = score
				}
				let total = 0
				for (let t = 0; t <= position; t += 1) {
					const weight = Math.fround(Math.exp(this.#scores[t]! - top))
					this.#scores[t] = weight
					total += weight
				}
				this.#contextSum.fill(0)
				for (let t = 0; t <= position; t += 1) {
					const weight = this.#scores[t]! / total
					const row = base + t * dh
					for (let index = 0; index < dh; index += 1) this.#contextSum[index] = this.#contextSum[index]! + weight * values[row + index]!
				}
				for (let index = 0; index < dh; index += 1) this.#context[head * dh + index] = this.#contextSum[index]!
			}
			layer.output.apply(this.#context, h)
			for (let index = 0; index < config.dim; index += 1) x[index] = x[index]! + h[index]!
			rmsNorm(x, layer.ffnNorm, h, config.normEps)
			layer.gate.apply(h, this.#gate)
			layer.up.apply(h, this.#up)
			for (let index = 0; index < config.ffnHidden; index += 1) {
				const gate = this.#gate[index]!
				this.#gate[index] = (gate / (1 + Math.exp(-gate))) * this.#up[index]!
			}
			layer.down.apply(this.#gate, h)
			for (let index = 0; index < config.dim; index += 1) x[index] = x[index]! + h[index]!
		}
		rmsNorm(x, this.#finalNorm, h, config.normEps)
		this.#head.apply(h, this.#logits)
		this.#length += 1
		return this.#logits
	}

	#rotate(vector: Float32Array, heads: number, half: number, position: number): void {
		for (let head = 0; head < heads; head += 1) {
			for (let index = 0; index < half; index += 1) {
				const at = head * half * 2 + 2 * index
				const c = this.#cos[position * half + index]!
				const s = this.#sin[position * half + index]!
				const x0 = vector[at]!
				const x1 = vector[at + 1]!
				vector[at] = x0 * c - x1 * s
				vector[at + 1] = x0 * s + x1 * c
			}
		}
	}
}

/** The tensor-based session as an `Engine`. */
export class TensorEngine implements Engine {
	constructor(private readonly session: { readonly length: number; reset(): void; feed(ids: ArrayLike<number>): { readonly data: Float32Array } }) { }

	get length(): number {
		return this.session.length
	}

	reset(): void {
		this.session.reset()
	}

	feed(ids: readonly number[]): Float32Array {
		return this.session.feed(ids).data
	}
}
