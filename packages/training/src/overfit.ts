import { Transformer } from '@scantling/nn'
import { Random } from '@scantling/tensor'
import { RandomBatchSource, TokenWindows } from './batches.ts'
import { AdamW } from './optimizers.ts'
import { warmupCosine } from './schedule.ts'
import { Trainer } from './trainer.ts'

const steps = Number(Bun.argv[2] ?? 200)
const vocab = 32
const cycle = Array.from({ length: 16 }, (_, index) => (index * 7 + 3) % vocab)
const tokens = Int32Array.from({ length: 320 }, (_, index) => cycle[index % cycle.length]!)

const model = new Transformer({ vocabSize: vocab, dim: 32, layers: 2, heads: 2, maxSeqLen: 32 }, 1)
const windows = new TokenWindows(tokens, 16)
const trainer = new Trainer({ model, optimizer: new AdamW(), schedule: warmupCosine({ peak: 5e-3, warmupSteps: 10, totalSteps: steps }), maxGradNorm: 1 })

console.log(`Model        ${model.parameterCount()} parameters, vocabulary ${vocab}, random initialisation (seed 1)`)
console.log(`Data         one 16-token cycle repeated; the loss of a model that learns nothing is ln(${vocab}) = ${Math.log(vocab).toFixed(3)}`)
const started = performance.now()
const summary = await trainer.run({
	steps,
	source: new RandomBatchSource(windows, 8, new Random(2)),
	evalEvery: 25,
	evalBatches: () => windows.evaluationBatches(8, 4),
	onStep: (metrics) => {
		if (metrics.step % 10 === 0) console.log(`step ${String(metrics.step).padStart(4)}  loss ${metrics.loss.toFixed(4)}  grad ${metrics.gradNorm.toFixed(3)}  lr ${metrics.learningRate.toExponential(2)}  ${Math.round(metrics.tokensPerSecond)} tokens/s`)
	},
	onEval: (step, result) => console.log(`  eval @${step}  loss ${result.loss.toFixed(4)}  perplexity ${result.perplexity.toFixed(2)}`)
})
const first = summary.history[0]?.loss ?? Number.NaN
const last = summary.history[summary.history.length - 1]?.loss ?? Number.NaN
console.log(`Result       loss ${first.toFixed(4)} -> ${last.toFixed(4)} in ${((performance.now() - started) / 1000).toFixed(1)} s, ${trainer.skippedSteps} skipped steps`)
