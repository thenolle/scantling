import { noGrad, ops, variable } from '@scantling/autodiff'
import type { RopeTables, Variable } from '@scantling/autodiff'
import { headDim } from '@scantling/nn'
import type { Attention, Transformer } from '@scantling/nn'
import * as T from '@scantling/tensor'

export interface FeedOptions {
	/** Return logits for every fed token instead of only the last one. */
	readonly all?: boolean
}

/**
 * Runs a Transformer incrementally with cached attention keys and values. `feed` takes the next tokens of
 * one sequence and returns next-token logits; earlier tokens are never recomputed. No autodiff graph is built.
 * Cached keys carry the rotary positions they were computed at, so the cache can only grow or be reset.
 */
export class InferenceSession {
	readonly model: Transformer
	readonly #keys: (T.Tensor | undefined)[]
	readonly #values: (T.Tensor | undefined)[]
	#length = 0

	constructor(model: Transformer) {
		this.model = model
		this.#keys = new Array<T.Tensor | undefined>(model.config.layers).fill(undefined)
		this.#values = new Array<T.Tensor | undefined>(model.config.layers).fill(undefined)
	}

	/** Number of tokens currently in the cache. */
	get length(): number {
		return this.#length
	}

	reset(): void {
		this.#keys.fill(undefined)
		this.#values.fill(undefined)
		this.#length = 0
	}

	/** Returns logits of shape `[rows, vocab]`: one row per fed token with `all`, otherwise only the last. */
	feed(ids: ArrayLike<number>, options: FeedOptions = {}): T.Tensor {
		const { config } = this.model
		const count = ids.length
		if (count === 0) throw new RangeError('feed needs at least one token')
		if (this.#length + count > config.maxSeqLen) throw new RangeError(`The context window of ${config.maxSeqLen} tokens is full`)
		const position = this.#length
		return noGrad(() => {
			const rope = ops.ropeTables(count, headDim(config), { base: config.ropeBase, offset: position })
			let x = ops.reshape(this.model.embedding.forward(ids), [1, count, config.dim])
			this.model.blocks.forEach((block, layer) => {
				x = ops.add(x, this.#attend(block.attention, block.attentionNorm.forward(x), rope, layer, position))
				x = ops.add(x, block.feedForward.forward(block.feedForwardNorm.forward(x)))
			})
			x = this.model.finalNorm.forward(x)
			const rows = options.all === true ? count : 1
			if (rows === 1) x = ops.sliceAxis(x, 1, count - 1, count)
			const logits = this.model.head === undefined ? ops.matmul(x, this.model.embedding.table, { transposeB: true }) : this.model.head.forward(x)
			this.#length += count
			return logits.value.reshape([rows, config.vocabSize])
		})
	}

	#attend(attention: Attention, h: Variable, rope: RopeTables, layer: number, position: number): Variable {
		const { heads, kvHeads } = this.model.config
		const dh = headDim(this.model.config)
		const count = h.shape[1]!
		const split = (projected: Variable, groups: number): Variable => ops.permute(ops.reshape(projected, [1, count, groups, dh]), [0, 2, 1, 3])
		const q = ops.rope(split(attention.query.forward(h), heads), rope).value
		const k = ops.rope(split(attention.key.forward(h), kvHeads), rope).value
		const v = split(attention.value.forward(h), kvHeads).value
		const keys = this.#keys[layer]
		const values = this.#values[layer]
		const allKeys = keys === undefined ? k : T.concat([keys, k], 2)
		const allValues = values === undefined ? v : T.concat([values, v], 2)
		this.#keys[layer] = allKeys
		this.#values[layer] = allValues
		const total = allKeys.shape[2]!
		const groups = heads / kvHeads
		const spread = (t: T.Tensor): T.Tensor => groups === 1 ? t : T.zip(t.reshape([1, kvHeads, 1, total, dh]), T.Tensor.zeros([1, kvHeads, groups, total, dh]), (x) => x).reshape([1, heads, total, dh])
		let scores = T.mulScalar(T.matmul(q, spread(allKeys), { transposeB: true }), 1 / Math.sqrt(dh))
		if (count > 1) {
			const mask = T.Tensor.zeros([count, total])
			for (let row = 0; row < count; row += 1) for (let column = position + row + 1; column < total; column += 1) mask.data[row * total + column] = 1
			scores = T.zip(scores, mask, (score, masked) => (masked !== 0 ? -1e9 : score))
		}
		const context = T.matmul(T.softmax(scores), spread(allValues))
		const merged = T.permute(context, [0, 2, 1, 3]).reshape([1, count, heads * dh])
		return attention.output.forward(variable(merged))
	}
}
