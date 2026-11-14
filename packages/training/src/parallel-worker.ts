import { parentPort } from 'node:worker_threads'
import { backward, zeroGrad } from '@scantling/autodiff'
import { Transformer } from '@scantling/nn'
import { Tensor } from '@scantling/tensor'
import type { GradientJob, GradientReply } from './parallel-protocol.ts'

const port = parentPort
if (port === null) throw new Error('parallel-worker must run inside a worker thread')

let cached: { readonly key: string; readonly model: Transformer } | undefined

function modelFor(job: GradientJob): Transformer {
	const key = JSON.stringify(job.config)
	if (cached === undefined || cached.key !== key) cached = { key, model: new Transformer(job.config, 1) }
	return cached.model
}

port.on('message', (job: GradientJob) => {
	let reply: GradientReply
	try {
		const model = modelFor(job)
		const entries = model.namedParameters()
		const weights = new Float32Array(job.weights, 0, job.parameterCount)
		let offset = 0
		for (const { variable } of entries) {
			variable.value.data.set(weights.subarray(offset, offset + variable.value.size))
			offset += variable.value.size
		}
		zeroGrad(entries.map((entry) => entry.variable))
		const loss = model.loss(job.inputs, job.targets, job.batch, job.length)
		backward(loss, Tensor.scalar(job.scale))
		const slot = new Float32Array(job.grads, job.slot * job.parameterCount * 4, job.parameterCount)
		offset = 0
		for (const { variable } of entries) {
			if (variable.grad === null) slot.fill(0, offset, offset + variable.value.size)
			else slot.set(variable.grad.data, offset)
			offset += variable.value.size
		}
		reply = { ok: true, loss: loss.value.item() }
	} catch (error) {
		reply = { ok: false, error: error instanceof Error ? error.message : String(error) }
	}
	port.postMessage(reply)
})
