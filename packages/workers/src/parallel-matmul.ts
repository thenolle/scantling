import { matmul, planMatmul, Tensor } from '@scantling/tensor'
import type { MatmulOptions } from '@scantling/tensor'
import type { MatmulJob, MatmulPiece, MatmulReply } from './jobs.ts'
import { WorkerPool } from './pool.ts'

export interface ParallelMatmulOptions {
	readonly workers: number
	/** Products with fewer floating-point operations run on the calling thread. Defaults to 2 million. */
	readonly minimumFlops?: number
	/** One more worker is used for every this-many operations, up to the pool size. Defaults to 1 million. */
	readonly minFlopsPerWorker?: number
}

const FLOAT_BYTES = Float32Array.BYTES_PER_ELEMENT

/**
 * Runs matrix products on a pool of worker threads. Operands are copied into shared memory,
 * output rows are split across workers, and the result is copied back. Calls are serialised
 * because they share the staging buffers. Results are bit-identical to `matmul`.
 */
export class ParallelMatmul {
	readonly size: number
	readonly #pool: WorkerPool
	readonly #minimumFlops: number
	readonly #minFlopsPerWorker: number
	#a = new SharedArrayBuffer(0)
	#b = new SharedArrayBuffer(0)
	#c = new SharedArrayBuffer(0)
	#tail: Promise<unknown> = Promise.resolve()

	constructor(options: ParallelMatmulOptions) {
		this.size = options.workers
		this.#minimumFlops = options.minimumFlops ?? 2_000_000
		this.#minFlopsPerWorker = options.minFlopsPerWorker ?? 1_000_000
		this.#pool = new WorkerPool(options.workers, new URL('./matmul-worker.ts', import.meta.url))
	}

	/** Same contract as `matmul`, resolved asynchronously. */
	matmul(a: Tensor, b: Tensor, options: MatmulOptions = {}): Promise<Tensor> {
		const result = this.#tail.then(() => this.#run(a, b, options))
		this.#tail = result.catch(() => undefined)
		return result
	}

	close(): Promise<void> {
		return this.#pool.close()
	}

	static #grow(buffer: SharedArrayBuffer, bytes: number): SharedArrayBuffer {
		return buffer.byteLength >= bytes ? buffer : new SharedArrayBuffer(Math.max(bytes, buffer.byteLength * 2))
	}

	async #run(a: Tensor, b: Tensor, options: MatmulOptions): Promise<Tensor> {
		const plan = planMatmul(a, b, options)
		const flops = 2 * plan.m * plan.k * plan.n * plan.count
		if (flops < this.#minimumFlops) return matmul(a, b, options)
		const outputSize = plan.count * plan.m * plan.n
		this.#a = ParallelMatmul.#grow(this.#a, a.size * FLOAT_BYTES)
		this.#b = ParallelMatmul.#grow(this.#b, b.size * FLOAT_BYTES)
		this.#c = ParallelMatmul.#grow(this.#c, outputSize * FLOAT_BYTES)
		new Float32Array(this.#a, 0, a.size).set(a.data)
		new Float32Array(this.#b, 0, b.size).set(b.data)
		const totalRows = plan.count * plan.m
		const useful = this.#minFlopsPerWorker <= 0 ? this.size : Math.min(this.size, Math.max(1, Math.floor(flops / this.#minFlopsPerWorker)))
		const parts = Math.min(totalRows, useful === this.size ? this.size * 2 : useful)
		const jobs: Promise<MatmulReply>[] = []
		for (let part = 0; part < parts; part += 1) {
			const start = Math.floor((part * totalRows) / parts)
			const end = Math.floor(((part + 1) * totalRows) / parts)
			const pieces: MatmulPiece[] = []
			for (let batch = Math.floor(start / plan.m); batch <= Math.floor((end - 1) / plan.m); batch += 1) {
				pieces.push({
					ao: plan.offsetsA[batch]!,
					bo: plan.offsetsB[batch]!,
					co: batch * plan.m * plan.n,
					i0: Math.max(start, batch * plan.m) - batch * plan.m,
					i1: Math.min(end, (batch + 1) * plan.m) - batch * plan.m
				})
			}
			const job: MatmulJob = { kind: plan.kind, a: this.#a, b: this.#b, c: this.#c, m: plan.m, k: plan.k, n: plan.n, pieces }
			jobs.push(this.#pool.run<MatmulReply>(job))
		}
		for (const reply of await Promise.all(jobs)) {
			if (!reply.ok) throw new Error(`A matmul worker failed: ${reply.error}`)
		}
		return new Tensor(plan.shape, new Float32Array(this.#c, 0, outputSize).slice())
	}
}
