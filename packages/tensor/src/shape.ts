export type Shape = readonly number[]

/**
 * Number of elements in a tensor of the given shape. A scalar (`[]`) has one.
 */
export function sizeOf(shape: Shape): number {
	let size = 1
	for (const dimension of shape) size *= dimension
	return size
}

/**
 * Throws when a shape contains a negative or non-integer dimension.
 */
export function assertValidShape(shape: Shape): void {
	for (const dimension of shape) { if (!Number.isInteger(dimension) || dimension < 0) throw new RangeError(`Invalid dimension ${dimension} in shape [${shape.join(', ')}]`) }
}

/**
 * Row-major strides, in elements.
 */
export function stridesOf(shape: Shape): number[] {
	const strides = new Array<number>(shape.length).fill(0)
	let stride = 1
	for (let axis = shape.length - 1; axis >= 0; axis -= 1) {
		strides[axis] = stride
		stride *= shape[axis]!
	}
	return strides
}

export function shapesEqual(a: Shape, b: Shape): boolean {
	return a.length === b.length && a.every((dimension, index) => dimension === b[index])
}

/**
 * Resolves a possibly negative axis against a rank and validates it.
 */
export function normalizeAxis(axis: number, rank: number): number {
	const resolved = axis < 0 ? axis + rank : axis
	if (!Number.isInteger(resolved) || resolved < 0 || resolved >= rank) throw new RangeError(`Axis ${axis} is out of range for rank ${rank}`)
	return resolved
}

/**
 * Computes the shape that results from broadcasting two shapes (NumPy rules:
 * align trailing dimensions; each pair must be equal or contain a 1).
 */
export function broadcastShapes(a: Shape, b: Shape): number[] {
	const rank = Math.max(a.length, b.length)
	const shape = new Array<number>(rank).fill(1)
	for (let offset = 0; offset < rank; offset += 1) {
		const left = a[a.length - 1 - offset] ?? 1
		const right = b[b.length - 1 - offset] ?? 1
		if (left !== right && left !== 1 && right !== 1) throw new RangeError(`Shapes [${a.join(', ')}] and [${b.join(', ')}] cannot be broadcast together`)
		shape[rank - 1 - offset] = left === 1 ? right : left
	}
	return shape
}

/**
 * Strides for reading a tensor of `shape` as if it had `target` shape:
 * broadcast and missing dimensions get stride 0.
 */
export function broadcastStrides(shape: Shape, target: Shape): number[] {
	const own = stridesOf(shape)
	const strides = new Array<number>(target.length).fill(0)
	const lead = target.length - shape.length
	for (let axis = 0; axis < shape.length; axis += 1) strides[lead + axis] = shape[axis] === 1 ? 0 : own[axis]!
	return strides
}

/**
 * Resolves a reshape request containing at most one `-1` placeholder.
 */
export function resolveReshape(size: number, shape: Shape): number[] {
	const unknown = shape.indexOf(-1)
	if (shape.lastIndexOf(-1) !== unknown) throw new RangeError('Only one dimension can be -1')
	const known = shape.reduce((product, dimension) => (dimension === -1 ? product : product * dimension), 1)
	if (unknown === -1) {
		if (known !== size) throw new RangeError(`Cannot reshape ${size} elements into [${shape.join(', ')}]`)
		return [...shape]
	}
	if (known === 0 || size % known !== 0) throw new RangeError(`Cannot reshape ${size} elements into [${shape.join(', ')}]`)
	const resolved = [...shape]
	resolved[unknown] = size / known
	return resolved
}
