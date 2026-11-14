import { presetConfig, Transformer } from '@scantling/nn'
import { Random } from '@scantling/tensor'
import { RandomBatchSource, TokenWindows } from './batches.ts'
import { AdamW } from './optimizers.ts'
import { ParallelTrainer } from './parallel.ts'
import { constantRate } from './schedule.ts'
import { Trainer } from './trainer.ts'

const length = 128
const microBatch = 2
const warmup = 2
const steps = 4
const config = presetConfig('tiny', 4096)
const noise = new Random(1)
const windows = new TokenWindows(Int32Array.from({ length: 50_000 }, () => noise.nextInt(4096)), length)
const source = (): RandomBatchSource => new RandomBatchSource(windows, microBatch, new Random(2))
const median = (values: number[]): number => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)]!

console.log(`Tiny model, ${new Transformer(config).parameterCount()} parameters; every worker processes ${microBatch} x ${length} tokens per step`)

const sequential = new Trainer({ model: new Transformer(config, 1), optimizer: new AdamW(), schedule: constantRate(1e-4) })
const sequentialSource = source()
for (let index = 0; index < warmup; index += 1) sequential.trainStep(sequentialSource)
const base = median(Array.from({ length: steps }, () => sequential.trainStep(sequentialSource).tokensPerSecond))
console.log(`sequential      ${base.toFixed(0).padStart(7)} tokens/s`)

for (const workers of [1, 2, 4, 8, 12]) {
	const trainer = new ParallelTrainer({ model: new Transformer(config, 1), optimizer: new AdamW(), schedule: constantRate(1e-4), workers })
	const input = source()
	for (let index = 0; index < warmup; index += 1) await trainer.trainStep(input)
	const rates: number[] = []
	for (let index = 0; index < steps; index += 1) rates.push((await trainer.trainStep(input)).tokensPerSecond)
	await trainer.close()
	const rate = median(rates)
	console.log(`${String(workers).padStart(2)} workers     ${rate.toFixed(0).padStart(7)} tokens/s   ${(rate / base).toFixed(2)}x of sequential`)
}
