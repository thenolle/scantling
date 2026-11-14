import { presetConfig, Transformer } from '@scantling/nn'
import type { PresetName } from '@scantling/nn'
import { FastDecoder, InferenceSession } from './index.ts'

const warm = 16
const steps = 64
const token = (index: number): number => (index * 17 + 3) % 4096

function milliseconds(work: (index: number) => void): number {
	const started = performance.now()
	for (let index = 0; index < steps; index += 1) work(warm + index)
	return (performance.now() - started) / steps
}

for (const preset of ['tiny', 'small'] as PresetName[]) {
	const created = performance.now()
	const model = new Transformer(presetConfig(preset, 4096), 1)
	console.log(`\n${preset}: ${model.parameterCount()} parameters (${((model.parameterCount() * 4) / 1e6).toFixed(1)} MB as float32), built in ${((performance.now() - created) / 1000).toFixed(1)} s`)
	const prompt = Array.from({ length: warm }, (_, index) => token(index))
	const session = new InferenceSession(model)
	session.feed(prompt)
	const tensor = milliseconds((index) => session.feed([token(index)]))
	const float32 = new FastDecoder(model, 'float32')
	for (const id of prompt) float32.step(id)
	const dense = milliseconds((index) => float32.step(token(index)))
	const int8 = new FastDecoder(model, 'int8')
	for (const id of prompt) int8.step(id)
	const quantized = milliseconds((index) => int8.step(token(index)))
	const exact = new FastDecoder(model, 'float32')
	const approximate = new FastDecoder(model, 'int8')
	let worst = 0
	for (let index = 0; index < 24; index += 1) {
		const a = Float32Array.from(exact.step(token(index)))
		const b = approximate.step(token(index))
		const range = Math.max(...a.map(Math.abs))
		for (let position = 0; position < a.length; position += 1) worst = Math.max(worst, Math.abs(a[position]! - b[position]!) / range)
	}

	const row = (name: string, ms: number): void => console.log(`  ${name.padEnd(22)} ${ms.toFixed(2).padStart(7)} ms/token  ${(1000 / ms).toFixed(0).padStart(5)} tokens/s  ${(tensor / ms).toFixed(2)}x`)
	row('tensor path', tensor)
	row('fast decoder, float32', dense)
	row('fast decoder, int8', quantized)
	console.log(`  int8 largest logit error: ${(worst * 100).toFixed(2)}% of the logit range`)
}
