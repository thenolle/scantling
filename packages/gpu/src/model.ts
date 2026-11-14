import type { Variable } from '@scantling/autodiff'
import type { ModelConfig, Transformer } from '@scantling/nn'
import { Tensor } from '@scantling/tensor'
import type { CudaDevice } from './cuda/device.ts'
import { NeuralOps, ropeTables } from './cuda/neural.ts'
import type { DeviceBuffer } from './device.ts'

const RESERVE_BYTES = 256 * 1024 * 1024

function layout(c: ModelConfig, batch: number, length: number): Map<string, number> {
	const [d, F, V, H] = [c.dim, c.ffnHidden, c.vocabSize, c.heads]
	const N = batch * length
	const sizes = new Map<string, number>()
	const set = (names: readonly string[], size: number): void => {
		for (const name of names) sizes.set(name, size)
	}
	set(['ids', 'targets', 'losses', 'invf'], N)
	set(['logits', 'dlogits'], N * V)
	set(['xf', 'dx', 'dxf', 'rawq', 'rawk', 'tmp', 'dxn', 'dtmp', 'dctx', 'dq', 'dk', 'dv', 'dqraw', 'dkraw'], N * d)
	set(['dh', 'dgate', 'dup'], N * F)
	set(['dp'], batch * H * length * length)
	set(['cos', 'sin', 'isin'], length * (d / H))
	for (let layer = 0; layer <= c.layers; layer += 1) set([`x${layer}`], N * d)
	for (let layer = 0; layer < c.layers; layer += 1) {
		set([`${layer}.inv1`, `${layer}.inv2`], N)
		set(['xn1', 'qr', 'kr', 'v', 'ctx', 'x2', 'xn2'].map((name) => `${layer}.${name}`), N * d)
		set(['gate', 'up', 'hidden'].map((name) => `${layer}.${name}`), N * F)
		set([`${layer}.probs`], batch * H * length * length)
	}
	return sizes
}

/** Bytes of device memory a model needs for micro-batches of `batch` sequences of `length` tokens: activations, scratch, weights and gradients. */
export function estimateGpuBytes(config: ModelConfig, parameters: number, batch: number, length: number): number {
	let floats = parameters * 2
	for (const size of layout(config, batch, length).values()) floats += size
	return floats * 4
}

interface Param {
	readonly variable: Variable
	readonly weights: DeviceBuffer
	readonly grads: DeviceBuffer
}

/**
 * The Transformer's forward and backward pass on the GPU, with the weights of a CPU `Transformer` (same parameter
 * names and shapes). Activations and gradients stay on the device; weights are copied up before a step and
 * gradients copied back after it. Supports standard multi-head attention and tied embeddings.
 */
export class GpuModel {
	readonly config: ModelConfig
	readonly neural: NeuralOps
	readonly capacity: number
	readonly length: number
	readonly #params = new Map<string, Param>()
	readonly #buffers = new Map<string, DeviceBuffer>()

	constructor(readonly device: CudaDevice, model: Transformer, size: { readonly batch: number; readonly length: number }) {
		this.config = model.config
		this.capacity = size.batch
		this.length = size.length
		const c = this.config
		if (c.kvHeads !== c.heads) throw new Error('The GPU model does not support grouped-query attention yet')
		if (!c.tieEmbeddings) throw new Error('The GPU model needs tied embeddings')
		if (size.length > c.maxSeqLen) throw new RangeError(`Sequence length ${size.length} exceeds the model maximum of ${c.maxSeqLen}`)
		this.neural = new NeuralOps(device)
		const entries = model.namedParameters()
		const estimate = estimateGpuBytes(c, model.parameterCount(), size.batch, size.length)
		const free = device.memory().freeBytes - RESERVE_BYTES
		if (estimate > free) throw new RangeError(`A micro-batch of ${size.batch} x ${size.length} needs about ${(estimate / 1_048_576).toFixed(0)} MiB of GPU memory but only ${(free / 1_048_576).toFixed(0)} MiB is free. Lower --gpu-micro-batch or --length, or close programs using the GPU.`)
		try {
			for (const { name, variable } of entries) {
				const grads = device.alloc(variable.value.size)
				device.fill(grads, 0)
				this.#params.set(name, { variable, weights: device.alloc(variable.value.size), grads })
			}
			for (const [name, floats] of layout(c, size.batch, size.length)) this.#buffers.set(name, device.alloc(floats))
			const tables = ropeTables(size.length, c.dim / c.heads, c.ropeBase)
			device.upload(this.#b('cos'), tables.cos)
			device.upload(this.#b('sin'), tables.sin)
			device.upload(this.#b('isin'), tables.inverse)
		} catch (error) {
			this.dispose()
			throw error
		}
	}

	/** Copies the CPU weights to the device. Call before every step and evaluation. */
	uploadWeights(): void {
		for (const param of this.#params.values()) this.device.upload(param.weights, param.variable.value.data)
	}

	zeroGrads(): void {
		for (const param of this.#params.values()) this.device.fill(param.grads, 0)
	}

	/** Copies the accumulated gradients, multiplied by `scale`, onto the CPU parameters' `grad`. */
	downloadGrads(scale: number): void {
		for (const param of this.#params.values()) {
			const data = new Float32Array(param.variable.value.size)
			this.device.download(param.grads, data)
			if (scale !== 1) for (let index = 0; index < data.length; index += 1) data[index] = data[index]! * scale
			param.variable.grad = new Tensor(param.variable.value.shape, data)
		}
	}

	dispose(): void {
		for (const param of this.#params.values()) {
			this.device.free(param.weights)
			this.device.free(param.grads)
		}
		for (const buffer of this.#buffers.values()) this.device.free(buffer)
		this.#params.clear()
		this.#buffers.clear()
	}

	/** Forward pass for `rows` sequences. Returns the mean next-token loss and leaves the activations for `backward`. */
	forward(inputs: ArrayLike<number>, targets: ArrayLike<number>, rows: number): number {
		const c = this.config
		const [d, F, V, H, T] = [c.dim, c.ffnHidden, c.vocabSize, c.heads, this.length]
		const [D, N] = [d / H, rows * T]
		if (!Number.isInteger(rows) || rows < 1 || rows > this.capacity) throw new RangeError(`The GPU model holds up to ${this.capacity} sequences, received ${rows}`)
		if (inputs.length !== N || targets.length !== N) throw new RangeError(`Expected ${N} token ids, received ${inputs.length} inputs and ${targets.length} targets`)
		const { device, neural } = this
		device.upload(this.#b('ids'), Float32Array.from(inputs))
		device.upload(this.#b('targets'), Float32Array.from(targets))
		neural.embedding(this.#w('embedding.table'), this.#b('ids'), this.#b('x0'), N, d)
		const scale = 1 / Math.sqrt(D)
		for (let layer = 0; layer < c.layers; layer += 1) {
			const [x, next] = [this.#b(`x${layer}`), this.#b(`x${layer + 1}`)]
			const a = (name: string): DeviceBuffer => this.#b(`${layer}.${name}`)
			const w = (name: string): DeviceBuffer => this.#w(`blocks.${layer}.${name}`)
			neural.rmsNorm(x, w('attention_norm.weight'), a('xn1'), a('inv1'), N, d, c.normEps)
			this.#gemm(a('xn1'), w('attention.query.weight'), this.#b('rawq'), N, d, d, false, false)
			this.#gemm(a('xn1'), w('attention.key.weight'), this.#b('rawk'), N, d, d, false, false)
			this.#gemm(a('xn1'), w('attention.value.weight'), a('v'), N, d, d, false, false)
			neural.rope(this.#b('rawq'), this.#b('cos'), this.#b('sin'), a('qr'), N, H, D, T)
			neural.rope(this.#b('rawk'), this.#b('cos'), this.#b('sin'), a('kr'), N, H, D, T)
			for (let s = 0; s < rows; s += 1) device.matmul(a('qr'), a('kr'), a('probs'), { m: T, k: D, n: T, transposeA: false, transposeB: true, batch: H, strideA: D, strideB: D, strideC: T * T, lda: d, ldb: d, ldc: T, offsetA: s * T * d, offsetB: s * T * d, offsetC: s * H * T * T })
			neural.softmax(a('probs'), a('probs'), rows * H * T, T, scale, T)
			for (let s = 0; s < rows; s += 1) device.matmul(a('probs'), a('v'), a('ctx'), { m: T, k: T, n: D, transposeA: false, transposeB: false, batch: H, strideA: T * T, strideB: D, strideC: D, lda: T, ldb: d, ldc: d, offsetA: s * H * T * T, offsetB: s * T * d, offsetC: s * T * d })
			this.#gemm(a('ctx'), w('attention.output.weight'), this.#b('tmp'), N, d, d, false, false)
			device.add(x, this.#b('tmp'), a('x2'))
			neural.rmsNorm(a('x2'), w('ffn_norm.weight'), a('xn2'), a('inv2'), N, d, c.normEps)
			this.#gemm(a('xn2'), w('ffn.gate.weight'), a('gate'), N, d, F, false, false)
			this.#gemm(a('xn2'), w('ffn.up.weight'), a('up'), N, d, F, false, false)
			neural.swiglu(a('gate'), a('up'), a('hidden'), N * F)
			this.#gemm(a('hidden'), w('ffn.down.weight'), this.#b('tmp'), N, F, d, false, false)
			device.add(a('x2'), this.#b('tmp'), next)
		}
		neural.rmsNorm(this.#b(`x${c.layers}`), this.#w('final_norm.weight'), this.#b('xf'), this.#b('invf'), N, d, c.normEps)
		this.#gemm(this.#b('xf'), this.#w('embedding.table'), this.#b('logits'), N, d, V, false, true)
		neural.crossEntropy(this.#b('logits'), this.#b('targets'), this.#b('dlogits'), this.#b('losses'), N, V)
		const losses = new Float32Array(N)
		device.download(this.#b('losses'), losses)
		let total = 0
		for (const value of losses) total += value
		return total / N
	}

	/** Backward pass for the `rows` sequences of the last `forward`. Adds to the weight gradients; call `zeroGrads` first. */
	backward(rows: number): void {
		const c = this.config
		const [d, F, V, H, T] = [c.dim, c.ffnHidden, c.vocabSize, c.heads, this.length]
		const [D, N] = [d / H, rows * T]
		const { device, neural } = this
		const scale = 1 / Math.sqrt(D)
		const [dx, dtmp, dxn] = [this.#b('dx'), this.#b('dtmp'), this.#b('dxn')]
		this.#gemm(this.#b('dlogits'), this.#b('xf'), this.#g('embedding.table'), V, N, d, true, false, 1)
		this.#gemm(this.#b('dlogits'), this.#w('embedding.table'), this.#b('dxf'), N, V, d, false, false)
		neural.rmsNormBackward(this.#b('dxf'), this.#b(`x${c.layers}`), this.#w('final_norm.weight'), this.#b('invf'), dx, this.#g('final_norm.weight'), N, d, true)
		for (let layer = c.layers - 1; layer >= 0; layer -= 1) {
			const x = this.#b(`x${layer}`)
			const a = (name: string): DeviceBuffer => this.#b(`${layer}.${name}`)
			const w = (name: string): DeviceBuffer => this.#w(`blocks.${layer}.${name}`)
			const g = (name: string): DeviceBuffer => this.#g(`blocks.${layer}.${name}`)
			this.#gemm(dx, w('ffn.down.weight'), this.#b('dh'), N, d, F, false, true)
			this.#gemm(a('hidden'), dx, g('ffn.down.weight'), F, N, d, true, false, 1)
			neural.swigluBackward(this.#b('dh'), a('gate'), a('up'), this.#b('dgate'), this.#b('dup'), N * F)
			this.#gemm(a('xn2'), this.#b('dgate'), g('ffn.gate.weight'), d, N, F, true, false, 1)
			this.#gemm(a('xn2'), this.#b('dup'), g('ffn.up.weight'), d, N, F, true, false, 1)
			this.#gemm(this.#b('dgate'), w('ffn.gate.weight'), dxn, N, F, d, false, true)
			this.#gemm(this.#b('dup'), w('ffn.up.weight'), dxn, N, F, d, false, true, 1)
			neural.rmsNormBackward(dxn, a('x2'), w('ffn_norm.weight'), a('inv2'), dtmp, g('ffn_norm.weight'), N, d, true)
			device.add(dx, dtmp, dx)
			this.#gemm(a('ctx'), dx, g('attention.output.weight'), d, N, d, true, false, 1)
			this.#gemm(dx, w('attention.output.weight'), this.#b('dctx'), N, d, d, false, true)
			for (let s = 0; s < rows; s += 1) {
				device.matmul(this.#b('dctx'), a('v'), this.#b('dp'), { m: T, k: D, n: T, transposeA: false, transposeB: true, batch: H, strideA: D, strideB: D, strideC: T * T, lda: d, ldb: d, ldc: T, offsetA: s * T * d, offsetB: s * T * d, offsetC: s * H * T * T })
				device.matmul(a('probs'), this.#b('dctx'), this.#b('dv'), { m: T, k: T, n: D, transposeA: true, transposeB: false, batch: H, strideA: T * T, strideB: D, strideC: D, lda: T, ldb: d, ldc: d, offsetA: s * H * T * T, offsetB: s * T * d, offsetC: s * T * d })
			}
			neural.softmaxBackward(this.#b('dp'), a('probs'), this.#b('dp'), rows * H * T, T, scale)
			for (let s = 0; s < rows; s += 1) {
				const spec = { m: T, k: T, n: D, transposeB: false, batch: H, strideA: T * T, strideB: D, strideC: D, lda: T, ldb: d, ldc: d, offsetA: s * H * T * T, offsetB: s * T * d, offsetC: s * T * d }
				device.matmul(this.#b('dp'), a('kr'), this.#b('dq'), { ...spec, transposeA: false })
				device.matmul(this.#b('dp'), a('qr'), this.#b('dk'), { ...spec, transposeA: true })
			}
			neural.rope(this.#b('dq'), this.#b('cos'), this.#b('isin'), this.#b('dqraw'), N, H, D, T)
			neural.rope(this.#b('dk'), this.#b('cos'), this.#b('isin'), this.#b('dkraw'), N, H, D, T)
			this.#gemm(a('xn1'), this.#b('dqraw'), g('attention.query.weight'), d, N, d, true, false, 1)
			this.#gemm(a('xn1'), this.#b('dkraw'), g('attention.key.weight'), d, N, d, true, false, 1)
			this.#gemm(a('xn1'), this.#b('dv'), g('attention.value.weight'), d, N, d, true, false, 1)
			this.#gemm(this.#b('dqraw'), w('attention.query.weight'), dxn, N, d, d, false, true)
			this.#gemm(this.#b('dkraw'), w('attention.key.weight'), dxn, N, d, d, false, true, 1)
			this.#gemm(this.#b('dv'), w('attention.value.weight'), dxn, N, d, d, false, true, 1)
			neural.rmsNormBackward(dxn, x, w('attention_norm.weight'), a('inv1'), dtmp, g('attention_norm.weight'), N, d, true)
			device.add(dx, dtmp, dx)
		}
		neural.embeddingBackward(dx, this.#b('ids'), this.#g('embedding.table'), N, d)
	}

	#gemm(a: DeviceBuffer, b: DeviceBuffer, c: DeviceBuffer, m: number, k: number, n: number, transposeA: boolean, transposeB: boolean, beta = 0): void {
		this.device.matmul(a, b, c, { m, k, n, transposeA, transposeB, batch: 1, strideA: 0, strideB: 0, strideC: 0, beta })
	}

	#b(name: string): DeviceBuffer {
		const buffer = this.#buffers.get(name)
		if (buffer === undefined) throw new Error(`The GPU model has no buffer "${name}"`)
		return buffer
	}

	#param(name: string): Param {
		const param = this.#params.get(name)
		if (param === undefined) throw new Error(`The model has no parameter "${name}"`)
		return param
	}

	#w(name: string): DeviceBuffer {
		return this.#param(name).weights
	}

	#g(name: string): DeviceBuffer {
		return this.#param(name).grads
	}
}
