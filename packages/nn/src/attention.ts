import type { RopeTables, Variable } from '@scantling/autodiff'
import { ops, variable } from '@scantling/autodiff'
import type { Random, Tensor } from '@scantling/tensor'
import { Tensor as TensorClass } from '@scantling/tensor'
import type { ModelConfig } from './config.ts'
import { headDim } from './config.ts'
import type { Module, NamedParameter } from './layers.ts'
import { Linear } from './layers.ts'

/**
 * Causal multi-head attention with rotary position embeddings and optional grouped-query sharing
 * of key/value heads. For input `[batch, length, dim]` it computes, per head,
 * `softmax(mask(Q K^T / sqrt(headDim))) V` and projects the concatenated heads back to `dim`.
 */
export class Attention implements Module {
	readonly query: Linear
	readonly key: Linear
	readonly value: Linear
	readonly output: Linear
	readonly #heads: number
	readonly #kvHeads: number
	readonly #headDim: number

	constructor(config: ModelConfig, random: Random, outputStd: number) {
		this.#heads = config.heads
		this.#kvHeads = config.kvHeads
		this.#headDim = headDim(config)
		this.query = new Linear(config.dim, config.heads * this.#headDim, random, config.initStd, 'attention.query')
		this.key = new Linear(config.dim, config.kvHeads * this.#headDim, random, config.initStd, 'attention.key')
		this.value = new Linear(config.dim, config.kvHeads * this.#headDim, random, config.initStd, 'attention.value')
		this.output = new Linear(config.heads * this.#headDim, config.dim, random, outputStd, 'attention.output')
	}

	forward(x: Variable, rope: RopeTables, causalMask: Tensor): Variable {
		const [batch, length] = x.shape as [number, number, number]
		const split = (projected: Variable, heads: number): Variable => ops.permute(ops.reshape(projected, [batch, length, heads, this.#headDim]), [0, 2, 1, 3])
		const q = ops.rope(split(this.query.forward(x), this.#heads), rope)
		const k = this.#expand(ops.rope(split(this.key.forward(x), this.#kvHeads), rope), batch, length)
		const v = this.#expand(split(this.value.forward(x), this.#kvHeads), batch, length)
		const scores = ops.mulScalar(ops.matmul(q, k, { transposeB: true }), 1 / Math.sqrt(this.#headDim))
		const probabilities = ops.softmax(ops.maskedFill(scores, causalMask, -1e9))
		const context = ops.matmul(probabilities, v)
		const merged = ops.reshape(ops.permute(context, [0, 2, 1, 3]), [batch, length, this.#heads * this.#headDim])
		return this.output.forward(merged)
	}

	namedParameters(prefix: string): NamedParameter[] {
		return [
			...this.query.namedParameters(`${prefix}.query`),
			...this.key.namedParameters(`${prefix}.key`),
			...this.value.namedParameters(`${prefix}.value`),
			...this.output.namedParameters(`${prefix}.output`)
		]
	}

	#expand(kv: Variable, batch: number, length: number): Variable {
		const groups = this.#heads / this.#kvHeads
		if (groups === 1) return kv
		const spread = ops.reshape(kv, [batch, this.#kvHeads, 1, length, this.#headDim])
		const widened = ops.add(spread, variable(TensorClass.zeros([batch, this.#kvHeads, groups, length, this.#headDim])))
		return ops.reshape(widened, [batch, this.#heads, length, this.#headDim])
	}
}
