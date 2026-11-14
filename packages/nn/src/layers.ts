import type { Variable } from '@scantling/autodiff'
import { ops, parameter } from '@scantling/autodiff'
import type { Random } from '@scantling/tensor'
import { randomNormal, Tensor } from '@scantling/tensor'

export interface NamedParameter {
	readonly name: string
	readonly variable: Variable
}

export interface Module {
	namedParameters(prefix: string): NamedParameter[]
}

/** A bias-free linear map `x @ W` with `W` of shape `[in, out]`. */
export class Linear implements Module {
	readonly weight: Variable

	constructor(inFeatures: number, outFeatures: number, random: Random, std: number, label = 'linear') {
		this.weight = parameter(randomNormal([inFeatures, outFeatures], random, std), `${label}.weight`)
	}

	forward(x: Variable): Variable {
		return ops.matmul(x, this.weight)
	}

	namedParameters(prefix: string): NamedParameter[] {
		return [{ name: `${prefix}.weight`, variable: this.weight }]
	}
}

/** A lookup table of shape `[vocab, dim]`. */
export class Embedding implements Module {
	readonly table: Variable

	constructor(vocabSize: number, dim: number, random: Random, std: number) {
		this.table = parameter(randomNormal([vocabSize, dim], random, std), 'embedding.table')
	}

	/** Returns `[ids.length, dim]`. */
	forward(ids: ArrayLike<number>): Variable {
		return ops.embedding(this.table, ids)
	}

	namedParameters(prefix: string): NamedParameter[] {
		return [{ name: `${prefix}.table`, variable: this.table }]
	}
}

/** Root-mean-square normalisation over the last axis with a learned scale that starts at 1. */
export class RmsNorm implements Module {
	readonly weight: Variable

	constructor(dim: number, readonly eps: number) {
		this.weight = parameter(Tensor.ones([dim]), 'norm.weight')
	}

	forward(x: Variable): Variable {
		return ops.rmsNorm(x, this.weight, this.eps)
	}

	namedParameters(prefix: string): NamedParameter[] {
		return [{ name: `${prefix}.weight`, variable: this.weight }]
	}
}
