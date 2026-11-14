import type { RopeTables, Variable } from '@scantling/autodiff'
import { ops } from '@scantling/autodiff'
import type { Tensor } from '@scantling/tensor'
import { Random, Tensor as TensorClass } from '@scantling/tensor'
import { Attention } from './attention.ts'
import type { ModelConfig, ModelConfigInput } from './config.ts'
import { headDim, resolveConfig } from './config.ts'
import { SwiGlu } from './feedforward.ts'
import type { Module, NamedParameter } from './layers.ts'
import { Embedding, Linear, RmsNorm } from './layers.ts'

/** Pre-norm residual block: `x + attention(norm(x))`, then `x + feedForward(norm(x))`. */
export class Block implements Module {
	readonly attentionNorm: RmsNorm
	readonly attention: Attention
	readonly feedForwardNorm: RmsNorm
	readonly feedForward: SwiGlu

	constructor(config: ModelConfig, random: Random, outputStd: number) {
		this.attentionNorm = new RmsNorm(config.dim, config.normEps)
		this.attention = new Attention(config, random, outputStd)
		this.feedForwardNorm = new RmsNorm(config.dim, config.normEps)
		this.feedForward = new SwiGlu(config, random, outputStd)
	}

	forward(x: Variable, rope: RopeTables, causalMask: Tensor): Variable {
		const attended = ops.add(x, this.attention.forward(this.attentionNorm.forward(x), rope, causalMask))
		return ops.add(attended, this.feedForward.forward(this.feedForwardNorm.forward(attended)))
	}

	namedParameters(prefix: string): NamedParameter[] {
		return [
			...this.attentionNorm.namedParameters(`${prefix}.attention_norm`),
			...this.attention.namedParameters(`${prefix}.attention`),
			...this.feedForwardNorm.namedParameters(`${prefix}.ffn_norm`),
			...this.feedForward.namedParameters(`${prefix}.ffn`)
		]
	}
}

export interface LossOptions {
	readonly ignoreIndex?: number
}

/**
 * A decoder-only Transformer language model. Weights are drawn from a seeded generator, so the
 * same configuration and seed always produce the same initial model.
 */
export class Transformer implements Module {
	readonly config: ModelConfig
	readonly embedding: Embedding
	readonly blocks: Block[]
	readonly finalNorm: RmsNorm
	readonly head: Linear | undefined
	readonly #masks = new Map<number, Tensor>()
	readonly #ropes = new Map<number, RopeTables>()

	constructor(input: ModelConfigInput, seed = 1) {
		this.config = resolveConfig(input)
		const random = new Random(seed)
		const outputStd = this.config.initStd / Math.sqrt(2 * this.config.layers)
		this.embedding = new Embedding(this.config.vocabSize, this.config.dim, random, this.config.initStd)
		this.blocks = Array.from({ length: this.config.layers }, () => new Block(this.config, random, outputStd))
		this.finalNorm = new RmsNorm(this.config.dim, this.config.normEps)
		this.head = this.config.tieEmbeddings ? undefined : new Linear(this.config.dim, this.config.vocabSize, random, this.config.initStd, 'head')
	}

	/** Logits of shape `[batch, length, vocab]` for `ids` laid out as `batch` rows of `length` tokens. */
	forward(ids: ArrayLike<number>, batch: number, length: number): Variable {
		if (ids.length !== batch * length) throw new RangeError(`Expected ${batch * length} token ids, received ${ids.length}`)
		if (length > this.config.maxSeqLen) throw new RangeError(`Sequence length ${length} exceeds the maximum of ${this.config.maxSeqLen}`)
		const rope = this.#rope(length)
		const mask = this.#mask(length)
		let x = ops.reshape(this.embedding.forward(ids), [batch, length, this.config.dim])
		for (const block of this.blocks) x = block.forward(x, rope, mask)
		x = this.finalNorm.forward(x)
		return this.head === undefined ? ops.matmul(x, this.embedding.table, { transposeB: true }) : this.head.forward(x)
	}

	/** Mean next-token cross-entropy for `targets` laid out like the inputs. */
	loss(inputs: ArrayLike<number>, targets: ArrayLike<number>, batch: number, length: number, options: LossOptions = {}): Variable {
		return ops.crossEntropy(this.forward(inputs, batch, length), targets, options)
	}

	namedParameters(prefix = ''): NamedParameter[] {
		const join = (name: string): string => (prefix === '' ? name : `${prefix}.${name}`)
		return [
			...this.embedding.namedParameters(join('embedding')),
			...this.blocks.flatMap((block, index) => block.namedParameters(join(`blocks.${index}`))),
			...this.finalNorm.namedParameters(join('final_norm')),
			...(this.head === undefined ? [] : this.head.namedParameters(join('head')))
		]
	}

	parameters(): Variable[] {
		return this.namedParameters().map((entry) => entry.variable)
	}

	/** Counts the elements of every parameter. Equals `parameterCount(config)`. */
	parameterCount(): number {
		return this.parameters().reduce((total, parameter) => total + parameter.value.size, 0)
	}

	#rope(length: number): RopeTables {
		let tables = this.#ropes.get(length)
		if (tables === undefined) {
			tables = ops.ropeTables(length, headDim(this.config), { base: this.config.ropeBase })
			this.#ropes.set(length, tables)
		}
		return tables
	}

	#mask(length: number): Tensor {
		let mask = this.#masks.get(length)
		if (mask === undefined) {
			mask = TensorClass.zeros([length, length])
			for (let row = 0; row < length; row += 1) for (let column = row + 1; column < length; column += 1) mask.data[row * length + column] = 1
			this.#masks.set(length, mask)
		}
		return mask
	}
}
