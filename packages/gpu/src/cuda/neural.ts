import type { DeviceBuffer } from '../device.ts'
import type { BufferSlice, CudaDevice } from './device.ts'

type Operand = DeviceBuffer | BufferSlice

const BLOCK = 256
const COLUMN_BLOCK = 128
const grid = (count: number, block: number): readonly [number, number, number] => [Math.ceil(count / block), 1, 1]

function fits(operand: Operand, needed: number, label: string): void {
	const [buffer, offset] = 'buffer' in operand ? [operand.buffer, operand.offset] : [operand, 0]
	if (offset + needed > buffer.length) throw new RangeError(`${label} needs ${needed} elements but only ${buffer.length - offset} are available`)
}

export interface AdamWStep {
	/** The 1-based step count, used for bias correction. */
	readonly step: number
	readonly learningRate: number
	readonly beta1?: number
	readonly beta2?: number
	readonly epsilon?: number
	readonly weightDecay?: number
	/** Whether weight decay applies to this tensor: true for matrices, false for vectors such as norm scales. */
	readonly decayed: boolean
	/** Multiplies the gradient first, for example the clipping factor. */
	readonly gradScale?: number
}

/**
 * The neural-network operations of the Transformer on device buffers, each checked against the autograd operations
 * in the tests. Every method validates buffer sizes before launching. Token ids and targets are stored as float32,
 * which is exact for ids below 16.7 million. Launches are asynchronous: synchronise or download before reading.
 */
export class NeuralOps {
	constructor(readonly device: CudaDevice) {}

	swiglu(gate: Operand, up: Operand, out: Operand, count: number): void {
		for (const [operand, label] of [[gate, 'gate'], [up, 'up'], [out, 'out']] as const) fits(operand, count, `swiglu ${label}`)
		this.device.runProgram('swiglu', grid(count, BLOCK), BLOCK, { gate, up, out }, { n: count }, {})
	}

	swigluBackward(dOut: Operand, gate: Operand, up: Operand, dGate: Operand, dUp: Operand, count: number): void {
		for (const [operand, label] of [[dOut, 'dout'], [gate, 'gate'], [up, 'up'], [dGate, 'dgate'], [dUp, 'dup']] as const) fits(operand, count, `swigluBackward ${label}`)
		this.device.runProgram('swigluBackward', grid(count, BLOCK), BLOCK, { dout: dOut, gate, up, dgate: dGate, dup: dUp }, { n: count }, {})
	}

	adamw(p: Operand, g: Operand, m: Operand, v: Operand, count: number, options: AdamWStep): void {
		for (const [operand, label] of [[p, 'p'], [g, 'g'], [m, 'm'], [v, 'v']] as const) fits(operand, count, `adamw ${label}`)
		if (!Number.isInteger(options.step) || options.step < 1) throw new RangeError('adamw needs a step count of at least 1')
		const beta1 = options.beta1 ?? 0.9
		const beta2 = options.beta2 ?? 0.95
		this.device.runProgram('adamw', grid(count, BLOCK), BLOCK, { p, g, m, v }, { n: count }, {
			lr: options.learningRate,
			beta1,
			beta2,
			omb1: 1 - beta1,
			omb2: 1 - beta2,
			eps: options.epsilon ?? 1e-8,
			decay: options.decayed ? options.learningRate * (options.weightDecay ?? 0.1) : 0,
			invC1: 1 / (1 - beta1 ** options.step),
			invC2: 1 / (1 - beta2 ** options.step),
			gscale: options.gradScale ?? 1
		})
	}

	/** `out = softmax(scale * x)` per row. With `window` set, row `r` only sees its first `(r mod window) + 1` columns. */
	softmax(x: Operand, out: Operand, rows: number, cols: number, scale: number, window = 0): void {
		fits(x, rows * cols, 'softmax input')
		fits(out, rows * cols, 'softmax output')
		this.device.runProgram('softmax', [rows, 1, 1], BLOCK, { x, out }, { cols, window }, { scale })
	}

	softmaxBackward(dy: Operand, y: Operand, dx: Operand, rows: number, cols: number, scale: number): void {
		for (const [operand, label] of [[dy, 'dy'], [y, 'y'], [dx, 'dx']] as const) fits(operand, rows * cols, `softmaxBackward ${label}`)
		this.device.runProgram('softmaxBackward', [rows, 1, 1], BLOCK, { dy, y, dx }, { cols, window: 0 }, { scale })
	}

	/** `out = x / sqrt(mean(x^2) + eps) * weight`, saving one inverse root-mean-square per row for the backward pass. */
	rmsNorm(x: Operand, weight: Operand, out: Operand, inverse: Operand, rows: number, cols: number, eps: number): void {
		fits(x, rows * cols, 'rmsNorm input')
		fits(out, rows * cols, 'rmsNorm output')
		fits(weight, cols, 'rmsNorm weight')
		fits(inverse, rows, 'rmsNorm inverse')
		this.device.runProgram('rmsNorm', [rows, 1, 1], BLOCK, { x, w: weight, out, inverse }, { cols, window: 0 }, { eps, invCols: 1 / cols })
	}

	/** Writes `dx` and adds to (or, without `accumulate`, overwrites) `dWeight`. */
	rmsNormBackward(dy: Operand, x: Operand, weight: Operand, inverse: Operand, dx: Operand, dWeight: Operand, rows: number, cols: number, accumulate: boolean): void {
		for (const [operand, label] of [[dy, 'dy'], [x, 'x'], [dx, 'dx']] as const) fits(operand, rows * cols, `rmsNormBackward ${label}`)
		fits(weight, cols, 'rmsNormBackward weight')
		fits(dWeight, cols, 'rmsNormBackward dweight')
		fits(inverse, rows, 'rmsNormBackward inverse')
		this.device.runProgram('rmsNormBackward', [rows, 1, 1], BLOCK, { dy, x, w: weight, inverse, dx }, { cols, window: 0 }, { invCols: 1 / cols })
		this.device.runProgram('rmsNormWeightGrad', grid(cols, COLUMN_BLOCK), COLUMN_BLOCK, { dy, x, inverse, dweight: dWeight }, { rows, cols }, { keep: accumulate ? 1 : 0 })
	}

	/**
	 * Mean cross-entropy over `rows` positions. Writes the gradient `(softmax - onehot) / rows` into `grad` and one loss per
	 * row into `losses`; the mean loss is the sum of `losses` divided by `rows`. Every row must have a valid target.
	 */
	crossEntropy(logits: Operand, targets: Operand, grad: Operand, losses: Operand, rows: number, vocab: number): void {
		fits(logits, rows * vocab, 'crossEntropy logits')
		fits(grad, rows * vocab, 'crossEntropy grad')
		fits(targets, rows, 'crossEntropy targets')
		fits(losses, rows, 'crossEntropy losses')
		this.device.runProgram('crossEntropy', [rows, 1, 1], BLOCK, { logits, target: targets, grad, loss: losses }, { cols: vocab, window: 0 }, { scale: 1 / rows })
	}

	/** Writes the sum of squares of `count` elements to `out[outOffset]`. Used for the global gradient norm. */
	sumSquares(x: Operand, out: DeviceBuffer, count: number, outOffset = 0): void {
		fits(x, count, 'sumSquares input')
		fits({ buffer: out, offset: outOffset }, 1, 'sumSquares output')
		this.device.runProgram('sumSquares', [1, 1, 1], BLOCK, { x, out: { buffer: out, offset: outOffset } }, { cols: count, window: 0 }, {})
	}
}
