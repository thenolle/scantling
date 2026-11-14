import { noGrad } from '@scantling/autodiff'
import { presetConfig, Transformer } from '@scantling/nn'
import { InferenceSession } from './session.ts'

const model = new Transformer(presetConfig('tiny', 4096), 1)
const prompt = Array.from({ length: 64 }, (_, index) => (index * 31 + 7) % 4096)
const steps = 32
const next = (index: number): number => (index * 17 + 3) % 4096

const session = new InferenceSession(model)
let started = performance.now()
session.feed(prompt)
const prefill = performance.now() - started
started = performance.now()
for (let index = 0; index < steps; index += 1) session.feed([next(index)])
const cached = (performance.now() - started) / steps

const ids = [...prompt]
started = performance.now()
for (let index = 0; index < steps; index += 1) {
	ids.push(next(index))
	noGrad(() => model.forward(ids, 1, ids.length))
}
const recomputed = (performance.now() - started) / steps

console.log(`Tiny model, ${model.parameterCount()} parameters, prompt of ${prompt.length} tokens, ${steps} generated tokens`)
console.log(`Prefill (${prompt.length} tokens)        ${prefill.toFixed(1)} ms`)
console.log(`With the KV cache            ${cached.toFixed(1)} ms/token   ${(1000 / cached).toFixed(1)} tokens/s`)
console.log(`Recomputing every step       ${recomputed.toFixed(1)} ms/token   ${(1000 / recomputed).toFixed(1)} tokens/s`)
console.log(`Speedup from the cache       ${(recomputed / cached).toFixed(1)}x`)
