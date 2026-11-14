import { normalizeAxis, sizeOf, stridesOf } from './shape.ts'
import type { Shape } from './shape.ts'
import { Tensor } from './tensor.ts'

function copyStrided(src: Float32Array, shape: Shape, strides: readonly number[], srcOffset: number, dst: Float32Array): void {
	if (dst.length === 0) return
	const rank = shape.length
	if (rank === 0) {
		dst[0] = src[srcOffset]!
		return
	}
	const inner = shape[rank - 1]!
	const innerStride = strides[rank - 1]!
	const outer = dst.length / inner
	const counter = new Array<number>(rank).fill(0)
	let offset = srcOffset
	let written = 0
	for (let row = 0; row < outer; row += 1) {
		for (let column = 0, position = offset; column < inner; column += 1, position += innerStride) dst[written++] = src[position]!
		for (let axis = rank - 2; axis >= 0; axis -= 1) {
			const next = counter[axis]! + 1
			offset += strides[axis]!
			if (next < shape[axis]!) {
				counter[axis] = next
				break
			}
			counter[axis] = 0
			offset -= strides[axis]! * shape[axis]!
		}
	}
}

/**
 * Reorders axes and returns a new contiguous tensor. `axes[i]` is the input axis that becomes output axis `i`.
 */
export function permute(t: Tensor, axes: readonly number[]): Tensor {
	if (axes.length !== t.rank) throw new RangeError(`permute needs ${t.rank} axes, received ${axes.length}`)
	const resolved = axes.map((axis) => normalizeAxis(axis, t.rank))
	if (new Set(resolved).size !== t.rank) throw new RangeError(`permute axes [${axes.join(', ')}] are not a permutation`)
	const own = stridesOf(t.shape)
	const shape = resolved.map((axis) => t.shape[axis]!)
	const out = new Float32Array(t.size)
	copyStrided(t.data, shape, resolved.map((axis) => own[axis]!), 0, out)
	return new Tensor(shape, out)
}

/**
 * Swaps two axes (the last two by default) and returns a new contiguous tensor.
 */
export function transpose(t: Tensor, axis0 = -2, axis1 = -1): Tensor {
	const first = normalizeAxis(axis0, t.rank)
	const second = normalizeAxis(axis1, t.rank)
	const axes = Array.from({ length: t.rank }, (_, index) => index)
	axes[first] = second
	axes[second] = first
	return permute(t, axes)
}

/**
 * Copies the range `[start, end)` along `axis`.
 */
export function sliceAxis(t: Tensor, axis: number, start: number, end: number): Tensor {
	const resolved = normalizeAxis(axis, t.rank)
	const length = t.shape[resolved]!
	if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end > length || start > end) throw new RangeError(`Slice [${start}, ${end}) is out of range for axis ${axis} of size ${length}`)
	const shape = t.shape.map((dimension, index) => (index === resolved ? end - start : dimension))
	const strides = stridesOf(t.shape)
	const out = new Float32Array(sizeOf(shape))
	copyStrided(t.data, shape, strides, start * strides[resolved]!, out)
	return new Tensor(shape, out)
}

/**
 * Joins tensors along `axis`. All other dimensions must match.
 */
export function concat(tensors: readonly Tensor[], axis: number): Tensor {
	const first = tensors[0]
	if (first === undefined) throw new RangeError('concat needs at least one tensor')
	const resolved = normalizeAxis(axis, first.rank)
	let total = 0
	for (const t of tensors) {
		const compatible = t.rank === first.rank && t.shape.every((dimension, index) => index === resolved || dimension === first.shape[index])
		if (!compatible) throw new RangeError(`concat shapes [${first.shape.join(', ')}] and [${t.shape.join(', ')}] differ outside axis ${axis}`)
		total += t.shape[resolved]!
	}
	let outer = 1
	for (let index = 0; index < resolved; index += 1) outer *= first.shape[index]!
	let inner = 1
	for (let index = resolved + 1; index < first.rank; index += 1) inner *= first.shape[index]!
	const shape = first.shape.map((dimension, index) => (index === resolved ? total : dimension))
	const out = new Float32Array(sizeOf(shape))
	let position = 0
	for (const t of tensors) {
		const chunk = t.shape[resolved]! * inner
		for (let o = 0; o < outer; o += 1) out.set(t.data.subarray(o * chunk, (o + 1) * chunk), o * total * inner + position * inner)
		position += t.shape[resolved]!
	}
	return new Tensor(shape, out)
}

function assertRows(table: Tensor, label: string): { rows: number; width: number } {
	if (table.rank !== 2) throw new RangeError(`${label} needs a rank-2 table, received shape [${table.shape.join(', ')}]`)
	return { rows: table.shape[0]!, width: table.shape[1]! }
}

/**
 * Selects rows of a `[rows, width]` table. This is an embedding lookup.
 */
export function gatherRows(table: Tensor, ids: ArrayLike<number>): Tensor {
	const { rows, width } = assertRows(table, 'gatherRows')
	const out = new Float32Array(ids.length * width)
	for (let index = 0; index < ids.length; index += 1) {
		const id = ids[index]!
		if (!Number.isInteger(id) || id < 0 || id >= rows) throw new RangeError(`Row id ${id} is out of range for ${rows} rows`)
		out.set(table.data.subarray(id * width, (id + 1) * width), index * width)
	}
	return new Tensor([ids.length, width], out)
}

/**
 * Adds each row of `rows` into `target[ids[i]]`, in place. This is the gradient of `gatherRows`.
 */
export function scatterAddRowsInPlace(target: Tensor, ids: ArrayLike<number>, rows: Tensor): Tensor {
	const { rows: count, width } = assertRows(target, 'scatterAddRowsInPlace')
	if (rows.rank !== 2 || rows.shape[0] !== ids.length || rows.shape[1] !== width) throw new RangeError(`Expected rows of shape [${ids.length}, ${width}], received [${rows.shape.join(', ')}]`)
	for (let index = 0; index < ids.length; index += 1) {
		const id = ids[index]!
		if (!Number.isInteger(id) || id < 0 || id >= count) throw new RangeError(`Row id ${id} is out of range for ${count} rows`)
		const to = id * width
		const from = index * width
		for (let column = 0; column < width; column += 1) target.data[to + column] = target.data[to + column]! + rows.data[from + column]!
	}
	return target
}
