export interface ModelConfig {
	readonly vocabSize: number
	readonly dim: number
	readonly layers: number
	readonly heads: number
	/** Number of key/value heads. Equal to `heads` for standard attention, smaller for grouped-query attention. */
	readonly kvHeads: number
	readonly ffnHidden: number
	readonly maxSeqLen: number
	readonly ropeBase: number
	readonly normEps: number
	readonly tieEmbeddings: boolean
	/** Standard deviation of the weight initialisation. */
	readonly initStd: number
}

export interface ModelConfigInput {
	readonly vocabSize: number
	readonly dim: number
	readonly layers: number
	readonly heads: number
	readonly maxSeqLen: number
	readonly kvHeads?: number
	readonly ffnHidden?: number
	readonly ropeBase?: number
	readonly normEps?: number
	readonly tieEmbeddings?: boolean
	readonly initStd?: number
}

export type PresetName = 'micro' | 'tiny' | 'small'

const PRESETS: Readonly<Record<PresetName, Omit<ModelConfigInput, 'vocabSize'>>> = {
	micro: { dim: 64, layers: 2, heads: 4, maxSeqLen: 128 },
	tiny: { dim: 128, layers: 4, heads: 4, maxSeqLen: 256 },
	small: { dim: 384, layers: 6, heads: 6, maxSeqLen: 512 }
}

function positive(name: string, value: number): void {
	if (!Number.isInteger(value) || value < 1) throw new RangeError(`${name} must be a positive integer, received ${value}`)
}

/**
 * Applies defaults and validates a configuration. The feed-forward hidden size defaults to
 * 8/3 of the width rounded up to a multiple of 8.
 */
export function resolveConfig(input: ModelConfigInput): ModelConfig {
	for (const [name, value] of Object.entries({ vocabSize: input.vocabSize, dim: input.dim, layers: input.layers, heads: input.heads, maxSeqLen: input.maxSeqLen })) positive(name, value)
	const kvHeads = input.kvHeads ?? input.heads
	const ffnHidden = input.ffnHidden ?? Math.ceil((8 * input.dim) / 3 / 8) * 8
	positive('kvHeads', kvHeads)
	positive('ffnHidden', ffnHidden)
	if (input.dim % input.heads !== 0) throw new RangeError(`dim ${input.dim} must be divisible by heads ${input.heads}`)
	if (input.heads % kvHeads !== 0) throw new RangeError(`heads ${input.heads} must be divisible by kvHeads ${kvHeads}`)
	if ((input.dim / input.heads) % 2 !== 0) throw new RangeError('The head size must be even for rotary embeddings')
	return {
		vocabSize: input.vocabSize,
		dim: input.dim,
		layers: input.layers,
		heads: input.heads,
		kvHeads,
		ffnHidden,
		maxSeqLen: input.maxSeqLen,
		ropeBase: input.ropeBase ?? 10000,
		normEps: input.normEps ?? 1e-5,
		tieEmbeddings: input.tieEmbeddings ?? true,
		initStd: input.initStd ?? 0.02
	}
}

/** A named size from the progression micro, tiny, small. */
export function presetConfig(name: PresetName, vocabSize: number, overrides: Partial<Omit<ModelConfigInput, 'vocabSize'>> = {}): ModelConfig {
	return resolveConfig({ ...PRESETS[name], ...overrides, vocabSize })
}

export const headDim = (config: ModelConfig): number => config.dim / config.heads

/**
 * Number of trainable parameters, computed from the configuration alone.
 */
export function parameterCount(config: ModelConfig): number {
	const dh = headDim(config)
	const attention = config.dim * config.heads * dh + 2 * config.dim * config.kvHeads * dh + config.heads * dh * config.dim
	const feedForward = 3 * config.dim * config.ffnHidden
	const layer = attention + feedForward + 2 * config.dim
	return config.vocabSize * config.dim + config.layers * layer + config.dim + (config.tieEmbeddings ? 0 : config.vocabSize * config.dim)
}
