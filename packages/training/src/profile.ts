import { backward, profiler, zeroGrad } from '@scantling/autodiff'
import { presetConfig, Transformer } from '@scantling/nn'
import { Random, setSimdEnabled } from '@scantling/tensor'

const batch = 2
const length = 128
const steps = 3
const random = new Random(1)
const model = new Transformer(presetConfig('tiny', 4096), 1)
const inputs = Int32Array.from({ length: batch * length }, () => random.nextInt(4096))
const targets = Int32Array.from({ length: batch * length }, () => random.nextInt(4096))
const variables = model.parameters()
const step = (): void => {
	if (profiler.enabled) profiler.mark = performance.now()
	zeroGrad(variables)
	backward(model.loss(inputs, targets, batch, length))
}

setSimdEnabled(process.env['SCANTLING_SIMD'] !== '0')
for (let index = 0; index < 3; index += 1) step()
profiler.start()
const started = performance.now()
for (let index = 0; index < steps; index += 1) step()
const total = performance.now() - started
profiler.stop()

const labels = [...new Set([...profiler.forward.keys(), ...profiler.backward.keys()])]
const rows = labels.map((label) => ({ label, forward: (profiler.forward.get(label) ?? 0) / steps, backward: (profiler.backward.get(label) ?? 0) / steps })).sort((a, b) => b.forward + b.backward - (a.forward + a.backward))
console.log(`Tiny, ${batch} x ${length} tokens: ${(total / steps).toFixed(0)} ms per forward and backward step`)
console.log('operation'.padEnd(24) + 'forward ms'.padStart(12) + 'backward ms'.padStart(13) + 'share'.padStart(8))
for (const row of rows.slice(0, 14)) console.log(row.label.padEnd(24) + row.forward.toFixed(1).padStart(12) + row.backward.toFixed(1).padStart(13) + `${(((row.forward + row.backward) / (total / steps)) * 100).toFixed(1)}%`.padStart(8))
