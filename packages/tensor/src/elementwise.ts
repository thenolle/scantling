import { broadcastShapes, broadcastStrides, shapesEqual, sizeOf } from './shape.ts'
import { Tensor } from './tensor.ts'

export type UnaryFn = (x: number) => number
export type BinaryFn = (a: number, b: number) => number

/**
 * Applies `fn` to every element.
 */
export function map(t: Tensor, fn: UnaryFn): Tensor {
	const out = new Float32Array(t.size)
	const x = t.data
	for (let index = 0; index < out.length; index += 1) out[index] = fn(x[index]!)
	return new Tensor(t.shape, out)
}

/**
 * Applies `fn` to pairs of elements, broadcasting the shapes against each other.
 */
export function zip(a: Tensor, b: Tensor, fn: BinaryFn): Tensor {
	const x = a.data
	const y = b.data
	if (shapesEqual(a.shape, b.shape)) {
		const same = new Float32Array(x.length)
		for (let index = 0; index < same.length; index += 1) same[index] = fn(x[index]!, y[index]!)
		return new Tensor(a.shape, same)
	}
	const shape = broadcastShapes(a.shape, b.shape)
	const out = new Float32Array(sizeOf(shape))
	if (out.length === 0) return new Tensor(shape, out)
	const rank = shape.length
	const stridesA = broadcastStrides(a.shape, shape)
	const stridesB = broadcastStrides(b.shape, shape)
	const inner = shape[rank - 1]!
	const innerA = stridesA[rank - 1]!
	const innerB = stridesB[rank - 1]!
	const outer = out.length / inner
	const counter = new Array<number>(rank).fill(0)
	let offsetA = 0
	let offsetB = 0
	let written = 0
	for (let row = 0; row < outer; row += 1) {
		for (let column = 0, pa = offsetA, pb = offsetB; column < inner; column += 1, pa += innerA, pb += innerB) out[written++] = fn(x[pa]!, y[pb]!)
		for (let axis = rank - 2; axis >= 0; axis -= 1) {
			const next = counter[axis]! + 1
			offsetA += stridesA[axis]!
			offsetB += stridesB[axis]!
			if (next < shape[axis]!) {
				counter[axis] = next
				break
			}
			counter[axis] = 0
			offsetA -= stridesA[axis]! * shape[axis]!
			offsetB -= stridesB[axis]! * shape[axis]!
		}
	}
	return new Tensor(shape, out)
}

function direct(a: Tensor, b: Tensor, same: (x: Float32Array, y: Float32Array, out: Float32Array) => void, general: BinaryFn): Tensor {
	if (!shapesEqual(a.shape, b.shape)) return zip(a, b, general)
	const out = new Float32Array(a.data.length)
	same(a.data, b.data, out)
	return new Tensor(a.shape, out)
}

export const add = (a: Tensor, b: Tensor): Tensor => direct(a, b, (x, y, out) => { for (let i = 0; i < out.length; i += 1) out[i] = x[i]! + y[i]! }, (x, y) => x + y)
export const sub = (a: Tensor, b: Tensor): Tensor => direct(a, b, (x, y, out) => { for (let i = 0; i < out.length; i += 1) out[i] = x[i]! - y[i]! }, (x, y) => x - y)
export const mul = (a: Tensor, b: Tensor): Tensor => direct(a, b, (x, y, out) => { for (let i = 0; i < out.length; i += 1) out[i] = x[i]! * y[i]! }, (x, y) => x * y)
export const div = (a: Tensor, b: Tensor): Tensor => direct(a, b, (x, y, out) => { for (let i = 0; i < out.length; i += 1) out[i] = x[i]! / y[i]! }, (x, y) => x / y)
export const maximum = (a: Tensor, b: Tensor): Tensor => zip(a, b, (x, y) => (x > y || x !== x ? x : y))
export const minimum = (a: Tensor, b: Tensor): Tensor => zip(a, b, (x, y) => (x < y || x !== x ? x : y))
export const pow = (a: Tensor, b: Tensor): Tensor => zip(a, b, Math.pow)

export function addScalar(t: Tensor, value: number): Tensor {
	const x = t.data
	const out = new Float32Array(x.length)
	for (let i = 0; i < out.length; i += 1) out[i] = x[i]! + value
	return new Tensor(t.shape, out)
}
export function mulScalar(t: Tensor, value: number): Tensor {
	const x = t.data
	const out = new Float32Array(x.length)
	for (let i = 0; i < out.length; i += 1) out[i] = x[i]! * value
	return new Tensor(t.shape, out)
}

export function neg(t: Tensor): Tensor {
	const x = t.data
	const out = new Float32Array(x.length)
	for (let i = 0; i < out.length; i += 1) out[i] = -x[i]!
	return new Tensor(t.shape, out)
}
export const abs = (t: Tensor): Tensor => map(t, Math.abs)
export function square(t: Tensor): Tensor {
	const x = t.data
	const out = new Float32Array(x.length)
	for (let i = 0; i < out.length; i += 1) out[i] = x[i]! * x[i]!
	return new Tensor(t.shape, out)
}
export const sqrt = (t: Tensor): Tensor => map(t, Math.sqrt)
export const exp = (t: Tensor): Tensor => map(t, Math.exp)
export const log = (t: Tensor): Tensor => map(t, Math.log)
export const tanh = (t: Tensor): Tensor => map(t, Math.tanh)
export const relu = (t: Tensor): Tensor => map(t, (x) => (x > 0 ? x : 0))
export const clamp = (t: Tensor, low: number, high: number): Tensor => map(t, (x) => Math.min(high, Math.max(low, x)))

/**
 * Logistic function, evaluated so that large magnitudes never overflow.
 */
export function sigmoid(t: Tensor): Tensor {
	return map(t, (x) => (x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x))))
}

/**
 * Sigmoid-weighted linear unit: `x * sigmoid(x)`.
 */
export function silu(t: Tensor): Tensor {
	return map(t, (x) => x / (1 + Math.exp(-x)))
}

const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI)

/**
 * Tanh approximation of the Gaussian error linear unit.
 */
export function gelu(t: Tensor): Tensor {
	return map(t, (x) => 0.5 * x * (1 + Math.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x))))
}

function assertSameShape(a: Tensor, b: Tensor, operation: string): void {
	if (!shapesEqual(a.shape, b.shape)) throw new RangeError(`${operation} needs equal shapes, received [${a.shape.join(', ')}] and [${b.shape.join(', ')}]`)
}

/** `target += other` for tensors of equal shape. Returns `target`. */
export function addInPlace(target: Tensor, other: Tensor): Tensor {
	assertSameShape(target, other, 'addInPlace')
	const out = target.data
	const x = other.data
	for (let index = 0; index < out.length; index += 1) out[index] = out[index]! + x[index]!
	return target
}

/** `target += alpha * x` for tensors of equal shape. Returns `target`. */
export function axpyInPlace(target: Tensor, alpha: number, x: Tensor): Tensor {
	assertSameShape(target, x, 'axpyInPlace')
	const out = target.data
	const values = x.data
	for (let index = 0; index < out.length; index += 1) out[index] = out[index]! + alpha * values[index]!
	return target
}

/** `target *= scale`. Returns `target`. */
export function scaleInPlace(target: Tensor, scale: number): Tensor {
	const out = target.data
	for (let index = 0; index < out.length; index += 1) out[index] = out[index]! * scale
	return target
}

/** Copies `source` into `target`; shapes must be equal. Returns `target`. */
export function copyInto(target: Tensor, source: Tensor): Tensor {
	assertSameShape(target, source, 'copyInto')
	target.data.set(source.data)
	return target
}
