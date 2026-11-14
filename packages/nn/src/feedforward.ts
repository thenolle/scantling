import type { Variable } from '@scantling/autodiff'
import { ops } from '@scantling/autodiff'
import type { Random } from '@scantling/tensor'
import type { ModelConfig } from './config.ts'
import type { Module, NamedParameter } from './layers.ts'
import { Linear } from './layers.ts'

/** SwiGLU feed-forward network: `down(silu(gate(x)) * up(x))`. */
export class SwiGlu implements Module {
	readonly gate: Linear
	readonly up: Linear
	readonly down: Linear

	constructor(config: ModelConfig, random: Random, outputStd: number) {
		this.gate = new Linear(config.dim, config.ffnHidden, random, config.initStd, 'ffn.gate')
		this.up = new Linear(config.dim, config.ffnHidden, random, config.initStd, 'ffn.up')
		this.down = new Linear(config.ffnHidden, config.dim, random, outputStd, 'ffn.down')
	}

	forward(x: Variable): Variable {
		return this.down.forward(ops.mul(ops.silu(this.gate.forward(x)), this.up.forward(x)))
	}

	namedParameters(prefix: string): NamedParameter[] {
		return [
			...this.gate.namedParameters(`${prefix}.gate`),
			...this.up.namedParameters(`${prefix}.up`),
			...this.down.namedParameters(`${prefix}.down`)
		]
	}
}
