import { Worker } from 'node:worker_threads'

interface Pending {
	readonly resolve: (value: unknown) => void
	readonly reject: (error: Error) => void
}

interface Job {
	readonly message: unknown
	readonly pending: Pending
}

/**
 * A fixed-size pool of worker threads that each run one job at a time. Jobs wait in a queue when
 * every worker is busy. A worker that crashes is replaced and its job is rejected.
 */
export class WorkerPool {
	readonly size: number
	readonly #script: URL
	readonly #all = new Set<Worker>()
	readonly #idle: Worker[] = []
	readonly #busy = new Map<Worker, Pending>()
	readonly #queue: Job[] = []
	#closed = false

	constructor(size: number, script: URL) {
		if (!Number.isInteger(size) || size < 1) throw new RangeError('A pool needs at least one worker')
		this.size = size
		this.#script = script
		for (let index = 0; index < size; index += 1) this.#idle.push(this.#spawn())
	}

	/** Sends `message` to the next free worker and resolves with its reply. */
	run<R>(message: unknown): Promise<R> {
		if (this.#closed) return Promise.reject(new Error('The worker pool is closed'))
		return new Promise<R>((resolve, reject) => {
			this.#queue.push({ message, pending: { resolve: resolve as (value: unknown) => void, reject } })
			this.#dispatch()
		})
	}

	/** Terminates every worker and rejects jobs that have not started. */
	async close(): Promise<void> {
		this.#closed = true
		for (const job of this.#queue.splice(0)) job.pending.reject(new Error('The worker pool was closed'))
		await Promise.all([...this.#all].map((worker) => worker.terminate()))
		this.#all.clear()
		this.#idle.length = 0
	}

	#spawn(): Worker {
		const worker = new Worker(this.#script)
		this.#all.add(worker)
		worker.on('message', (value: unknown) => {
			const pending = this.#busy.get(worker)
			this.#busy.delete(worker)
			if (!this.#closed) this.#idle.push(worker)
			pending?.resolve(value)
			this.#dispatch()
		})
		worker.on('error', (error: Error) => {
			const pending = this.#busy.get(worker)
			this.#busy.delete(worker)
			this.#all.delete(worker)
			const position = this.#idle.indexOf(worker)
			if (position !== -1) this.#idle.splice(position, 1)
			pending?.reject(error)
			if (!this.#closed) this.#idle.push(this.#spawn())
			this.#dispatch()
		})
		return worker
	}

	#dispatch(): void {
		while (this.#queue.length > 0 && this.#idle.length > 0) {
			const worker = this.#idle.pop()!
			const job = this.#queue.shift()!
			this.#busy.set(worker, job.pending)
			worker.postMessage(job.message)
		}
	}
}
