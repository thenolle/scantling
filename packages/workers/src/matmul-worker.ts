import { parentPort } from 'node:worker_threads'
import { KERNELS, accumulatorFor } from '@scantling/tensor'
import type { MatmulJob, MatmulReply } from './jobs.ts'

const port = parentPort
if (port === null) throw new Error('matmul-worker must run inside a worker thread')

port.on('message', (job: MatmulJob) => {
	let reply: MatmulReply
	try {
		const a = new Float32Array(job.a)
		const b = new Float32Array(job.b)
		const c = new Float32Array(job.c)
		const acc = accumulatorFor(job.n)
		const kernel = KERNELS[job.kind]
		for (const piece of job.pieces) kernel(a, piece.ao, b, piece.bo, c, piece.co, job.m, job.k, job.n, acc, piece.i0, piece.i1)
		reply = { ok: true }
	} catch (error) {
		reply = { ok: false, error: error instanceof Error ? error.message : String(error) }
	}
	port.postMessage(reply)
})
