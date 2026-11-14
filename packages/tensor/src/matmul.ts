import { KERNELS, accumulatorFor, kernelKind } from './kernels.ts'
import type { KernelKind } from './kernels.ts'
import { gemmSimd, simdActive } from './simd.ts'
import { broadcastShapes, broadcastStrides, sizeOf } from './shape.ts'
import type { Shape } from './shape.ts'
import { Tensor } from './tensor.ts'

export interface MatmulOptions {
	/** Treat `a` as stored transposed: its last two axes are `[k, m]`. */
	readonly transposeA?: boolean
	/** Treat `b` as stored transposed: its last two axes are `[n, k]`. */
	readonly transposeB?: boolean
}

/**
 * Everything needed to run a (possibly batched) matrix product: dimensions, the output shape and
 * where each matrix of the batch starts in the operands.
 */
export interface MatmulPlan {
	readonly kind: KernelKind
	readonly m: number
	readonly k: number
	readonly n: number
	readonly shape: number[]
	readonly count: number
	readonly offsetsA: number[]
	readonly offsetsB: number[]
}

function batchOffsets(batch: Shape, own: Shape, matrixSize: number): number[] {
	const count = sizeOf(batch)
	const strides = broadcastStrides(own, batch)
	const offsets = new Array<number>(count)
	for (let flat = 0; flat < count; flat += 1) {
		let rest = flat
		let offset = 0
		for (let axis = batch.length - 1; axis >= 0; axis -= 1) {
			const dimension = batch[axis]!
			offset += (rest % dimension) * strides[axis]!
			rest = Math.floor(rest / dimension)
		}
		offsets[flat] = offset * matrixSize
	}
	return offsets
}

/**
 * Validates the operands and describes the product. Throws `RangeError` on a shape mismatch.
 */
export function planMatmul(a: Tensor, b: Tensor, options: MatmulOptions = {}): MatmulPlan {
	if (a.rank < 2 || b.rank < 2) throw new RangeError('matmul needs tensors of rank 2 or more; reshape vectors first')
	const transposeA = options.transposeA === true
	const transposeB = options.transposeB === true
	const aRows = a.shape[a.rank - 2]!
	const aCols = a.shape[a.rank - 1]!
	const bRows = b.shape[b.rank - 2]!
	const bCols = b.shape[b.rank - 1]!
	const m = transposeA ? aCols : aRows
	const k = transposeA ? aRows : aCols
	const innerB = transposeB ? bCols : bRows
	const n = transposeB ? bRows : bCols
	if (k !== innerB) throw new RangeError(`Cannot multiply [${a.shape.join(', ')}] by [${b.shape.join(', ')}]: inner dimensions ${k} and ${innerB} differ`)
	const batchA = a.shape.slice(0, -2)
	const batchB = b.shape.slice(0, -2)
	const batch = broadcastShapes(batchA, batchB)
	return {
		kind: kernelKind(transposeA, transposeB),
		m,
		k,
		n,
		shape: [...batch, m, n],
		count: sizeOf(batch),
		offsetsA: batchOffsets(batch, batchA, aRows * aCols),
		offsetsB: batchOffsets(batch, batchB, bRows * bCols)
	}
}

/**
 * Matrix product over the last two axes with batch broadcasting.
 *
 * `a` is `[...batchA, m, k]` and `b` is `[...batchB, k, n]`; the result is `[...batch, m, n]`
 * where `batch` is the broadcast of `batchA` and `batchB`. A rank-2 `b` is therefore applied
 * to every matrix of a batched `a`, which is what a linear layer needs.
 * Large products use the SIMD kernel when the runtime supports it (float32 accumulation); small products and
 * single-row products use the scalar kernels (64-bit accumulation).
 */
export function matmul(a: Tensor, b: Tensor, options: MatmulOptions = {}): Tensor {
	const plan = planMatmul(a, b, options)
	const out = new Float32Array(plan.count * plan.m * plan.n)
	if (simdActive(plan.m, plan.k, plan.n)) {
		for (let index = 0; index < plan.count; index += 1) gemmSimd(a.data, plan.offsetsA[index]!, options.transposeA === true, b.data, plan.offsetsB[index]!, options.transposeB === true, out, index * plan.m * plan.n, plan.m, plan.k, plan.n)
		return new Tensor(plan.shape, out)
	}
	const kernel = KERNELS[plan.kind]
	const acc = accumulatorFor(plan.n)
	for (let index = 0; index < plan.count; index += 1) kernel(a.data, plan.offsetsA[index]!, b.data, plan.offsetsB[index]!, out, index * plan.m * plan.n, plan.m, plan.k, plan.n, acc, 0, plan.m)
	return new Tensor(plan.shape, out)
}
