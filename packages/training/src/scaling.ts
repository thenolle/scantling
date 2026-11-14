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

async function measureParallel(workers: number, rounds: number): Promise<void> {
	const trainer = new ParallelTrainer({ model: new Transformer(config, 1), optimizer: new AdamW(), schedule: constantRate(1e-4), workers, rounds })
	const input = source()
	for (let index = 0; index < warmup; index += 1) await trainer.trainStep(input)
	const rates: number[] = []
	for (let index = 0; index < steps; index += 1) rates.push((await trainer.trainStep(input)).tokensPerSecond)
	const t = trainer.lastTimings!
	await trainer.close()
	const rate = median(rates)
	const phases = `publish ${t.publishMs.toFixed(1)} ms, compute ${t.computeMs.toFixed(0)} ms, reduce ${t.reduceMs.toFixed(1)} ms, optimizer ${t.optimizerMs.toFixed(1)} ms`
	console.log(`${String(workers).padStart(2)} workers x ${rounds}  ${rate.toFixed(0).padStart(7)} tokens/s  ${(rate / base).toFixed(2)}x  | ${phases}`)
}

for (const workers of [1, 2, 4, 8, 12]) await measureParallel(workers, 1)
for (const rounds of [2, 4]) await measureParallel(8, rounds)
