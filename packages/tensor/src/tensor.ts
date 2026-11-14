import type { Shape } from './shape.ts'
import { assertValidShape, resolveReshape, sizeOf, stridesOf } from './shape.ts'

export type NestedNumbers = number | readonly NestedNumbers[]

/**
 * A dense, contiguous, row-major Float32 tensor.
 *
 * Operations in this package never modify their inputs. The only mutators are `set`, `fill`
 * and the functions whose names end in `InPlace`. `reshape` returns a view that shares memory.
 */
export class Tensor {
	readonly shape: readonly number[]
	readonly data: Float32Array

	constructor(shape: Shape, data: Float32Array) {
		assertValidShape(shape)
		if (data.length !== sizeOf(shape)) throw new RangeError(`Shape [${shape.join(', ')}] needs ${sizeOf(shape)} values but received ${data.length}`)
		this.shape = [...shape]
		this.data = data
	}

	static zeros(shape: Shape): Tensor {
		assertValidShape(shape)
		return new Tensor(shape, new Float32Array(sizeOf(shape)))
	}

	static ones(shape: Shape): Tensor {
		return Tensor.full(shape, 1)
	}

	static full(shape: Shape, value: number): Tensor {
		return Tensor.zeros(shape).fill(value)
	}

	static scalar(value: number): Tensor {
		return new Tensor([], Float32Array.of(value))
	}

	/** Copies `values` (flat, row-major) into a tensor of the given shape. */
	static from(shape: Shape, values: ArrayLike<number>): Tensor {
		return new Tensor(shape, Float32Array.from(values))
	}

	/** Builds a tensor from rectangular nested arrays, inferring the shape. */
	static fromNested(nested: NestedNumbers): Tensor {
		const shape: number[] = []
		const values: number[] = []
		let leafDepth = -1
		const walk = (node: NestedNumbers, depth: number): void => {
			if (typeof node === 'number') {
				if (leafDepth === -1) leafDepth = depth
				if (depth !== leafDepth) throw new RangeError('Nested arrays are not rectangular')
				values.push(node)
				return
			}
			if (leafDepth !== -1 && depth >= leafDepth) throw new RangeError('Nested arrays are not rectangular')
			if (shape.length === depth) shape.push(node.length)
			else if (shape[depth] !== node.length) throw new RangeError('Nested arrays are not rectangular')
			for (const child of node) walk(child, depth + 1)
		}
		walk(nested, 0)
		return Tensor.from(shape, values)
	}

	/** The integers `0 .. count - 1` as a rank-1 tensor. */
	static arange(count: number): Tensor {
		return Tensor.from([count], Array.from({ length: count }, (_, index) => index))
	}

	get rank(): number {
		return this.shape.length
	}

	get size(): number {
		return this.data.length
	}

	get strides(): number[] {
		return stridesOf(this.shape)
	}

	/** Reads one element. Throws on a wrong number of indices or an index out of range. */
	at(...indices: number[]): number {
		return this.data[this.#offsetOf(indices)]!
	}

	/** Writes one element. */
	set(indices: readonly number[], value: number): void {
		this.data[this.#offsetOf(indices)] = value
	}

	/** The only element of a one-element tensor. */
	item(): number {
		if (this.size !== 1) throw new RangeError(`item() needs a one-element tensor, received shape [${this.shape.join(', ')}]`)
		return this.data[0]!
	}

	/** Sets every element to `value` and returns the tensor. */
	fill(value: number): this {
		this.data.fill(value)
		return this
	}

	/** Returns a tensor with a new shape over the same memory. One dimension may be `-1`. */
	reshape(shape: Shape): Tensor {
		return new Tensor(resolveReshape(this.size, shape), this.data)
	}

	/** Returns an independent copy. */
	clone(): Tensor {
		return new Tensor(this.shape, this.data.slice())
	}

	/** Nested arrays for inspection and tests. A scalar becomes a number. */
	toArray(): NestedNumbers {
		const build = (axis: number, offset: number): NestedNumbers => {
			if (axis === this.rank) return this.data[offset]!
			const stride = this.strides[axis]!
			return Array.from({ length: this.shape[axis]! }, (_, index) => build(axis + 1, offset + index * stride))
		}
		return build(0, 0)
	}

	toString(): string {
		const preview = Array.from(this.data.subarray(0, 8)).map((value) => Number(value.toPrecision(5)))
		return `Tensor(shape=[${this.shape.join(', ')}], data=[${preview.join(', ')}${this.size > 8 ? ', …' : ''}])`
	}

	#offsetOf(indices: readonly number[]): number {
		if (indices.length !== this.rank) throw new RangeError(`Expected ${this.rank} indices for shape [${this.shape.join(', ')}], received ${indices.length}`)
		const strides = this.strides
		let offset = 0
		indices.forEach((index, axis) => {
			if (!Number.isInteger(index) || index < 0 || index >= this.shape[axis]!) throw new RangeError(`Index ${index} is out of range for axis ${axis} of size ${this.shape[axis]}`)
			offset += index * strides[axis]!
		})
		return offset
	}
}
