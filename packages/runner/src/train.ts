import { CheckpointStore, applyWeights, checkpointId, weightsOf } from '@scantling/checkpoint'
import type { CheckpointState } from '@scantling/checkpoint'
import { FileBatchSource, TokenFile, fileEvaluationBatches } from '@scantling/datasets'
import { computeBudget, detectHardware, formatBytes } from '@scantling/hardware'
import { Transformer, presetConfig } from '@scantling/nn'
import type { PresetName } from '@scantling/nn'
import { createRuntimeLogger } from '@scantling/telemetry'
import { Random } from '@scantling/tensor'
import { Tokenizer } from '@scantling/tokenizer'
import { AdamW, ParallelTrainer, warmupCosine } from '@scantling/training'
import { bitsPerByte, smokeChecks, tokenBytes, unigramLoss } from './evaluation.ts'

const args = Bun.argv.slice(2)
const option = (name: string): string | undefined => (args.includes(name) ? args[args.indexOf(name) + 1] : undefined)
const value = (name: string, fallback: number): number => Number(option(name) ?? fallback)

if (args.includes('--help')) {
	console.log('bun run train -- [--dataset dir] [--tokenizer file] [--preset micro|tiny|small] [--steps N] [--length 128] [--micro-batch 2]\n  [--rounds 1] [--workers N] [--peak-lr 0.002] [--eval-every 100] [--checkpoint-every 250] [--log-every 10] [--name run] [--seed 1] [--resume]')
	process.exit(0)
}

const datasetDirectory = option('--dataset') ?? 'data/datasets/default'
const tokenizer = Tokenizer.fromJson(await Bun.file(option('--tokenizer') ?? 'data/tokenizer.json').text())
const preset = (option('--preset') ?? 'tiny') as PresetName
const steps = value('--steps', 1000)
const length = value('--length', 128)
const microBatch = value('--micro-batch', 2)
const rounds = value('--rounds', 1)
const seed = value('--seed', 1)
const name = option('--name') ?? `${preset}-run`
const evalEvery = value('--eval-every', 100)
const checkpointEvery = value('--checkpoint-every', 250)
const logEvery = value('--log-every', 10)

const { logger } = createRuntimeLogger({ level: 'info', directory: 'data/logs', toFile: true, maxFileBytes: 10 * 1024 * 1024, maxFiles: 5 })
const profile = await detectHardware({ storagePaths: [process.cwd()] })
const budget = computeBudget(profile)
const workers = value('--workers', budget.workerThreads)

const train = TokenFile.open(`${datasetDirectory}/train.tokens`)
const validation = TokenFile.open(`${datasetDirectory}/validation.tokens`)
if (train.vocabSize !== tokenizer.vocabSize) throw new Error(`The dataset was built with a vocabulary of ${train.vocabSize} but the tokenizer has ${tokenizer.vocabSize}`)

const model = new Transformer(presetConfig(preset, tokenizer.vocabSize), seed)
const optimizer = new AdamW()
const source = new FileBatchSource(train, length, microBatch, new Random(seed + 1))
const evalSet = fileEvaluationBatches(validation, length, 8, 16)
const evalTargets = evalSet.flatMap((batch) => Array.from(batch.targets))
const baseline = unigramLoss(train.read(0, Math.min(train.count, 20_000_000)), evalTargets, tokenizer.vocabSize)
const evalBytes = tokenBytes(tokenizer, evalTargets)

const trainer = new ParallelTrainer({
	model,
	optimizer,
	schedule: warmupCosine({ peak: value('--peak-lr', 2e-3), warmupSteps: Math.max(20, Math.floor(steps * 0.05)), totalSteps: steps }),
	workers,
	rounds,
	maxGradNorm: 1,
	maxMemoryBytes: budget.trainingMemoryBytes
})

const store = new CheckpointStore(`data/checkpoints/${name}`)
if (args.includes('--resume')) {
	const id = store.latest()
	if (id === undefined) throw new Error('--resume was given but there is no checkpoint to resume from')
	const state = store.load(id)
	applyWeights(model.namedParameters(), state.weights)
	optimizer.loadState({ step: state.optimizer.step, tensors: state.optimizer.tensors })
	source.random.restore(state.random['batches']!)
	trainer.resume(state.step, state.skippedSteps)
	logger.info('Resumed', { checkpoint: id, step: state.step })
}

const tokensPerStep = workers * rounds * microBatch * length
logger.info('Training plan', {
	parameters: model.parameterCount(),
	trainTokens: train.count,
	validationTokens: validation.count,
	workers,
	tokensPerStep,
	steps,
	epochs: Number(((steps * tokensPerStep) / train.count).toFixed(2)),
	memoryBudget: formatBytes(budget.trainingMemoryBytes),
	uniformLoss: Number(Math.log(tokenizer.vocabSize).toFixed(4)),
	unigramBaseline: Number(baseline.toFixed(4))
})

let lastEval = trainer.evaluate(evalSet)

function checkpoint(): void {
	lastEval = trainer.evaluate(evalSet)
	const id = checkpointId(trainer.step)
	if (store.list().some((meta) => meta.id === id)) return
	const optimizerState = optimizer.state()
	const state: CheckpointState = {
		step: trainer.step,
		skippedSteps: trainer.skippedSteps,
		config: model.config,
		weights: weightsOf(model.namedParameters()),
		optimizer: { kind: optimizer.kind, step: optimizerState.step, tensors: optimizerState.tensors },
		random: { batches: source.random.state() },
		metrics: { evalLoss: lastEval.loss, perplexity: lastEval.perplexity, bitsPerByte: bitsPerByte(lastEval.loss, evalTargets.length, evalBytes), unigramLoss: baseline }
	}
	store.save(state)
	const decision = store.promote(id, { evalLoss: lastEval.loss, regression: smokeChecks(model.namedParameters(), lastEval.loss, baseline) })
	logger.info('Checkpoint', { id, evalLoss: Number(lastEval.loss.toFixed(4)), promoted: decision.accepted, reasons: decision.reasons })
}

const controller = new AbortController()
process.on('SIGINT', () => {
	logger.warn('Interrupt received: finishing the current step, then saving')
	controller.abort()
})

const started = performance.now()
let rate = 0
const remaining = Math.max(0, steps - trainer.step)
const summary = await trainer.run({
	steps: remaining,
	source,
	evalEvery,
	evalBatches: () => evalSet,
	signal: controller.signal,
	onStep: (metrics) => {
		rate = rate === 0 ? metrics.tokensPerSecond : rate * 0.9 + metrics.tokensPerSecond * 0.1
		if (metrics.step % logEvery === 0) {
			const eta = ((steps - metrics.step - 1) * tokensPerStep) / rate
			logger.info(`step ${metrics.step + 1}/${steps}`, { loss: Number(metrics.loss.toFixed(4)), gradNorm: Number(metrics.gradNorm.toFixed(3)), lr: Number(metrics.learningRate.toExponential(2)), tokensPerSecond: Math.round(rate), etaMinutes: Number((eta / 60).toFixed(1)), skipped: metrics.skipped })
		}
		if ((metrics.step + 1) % checkpointEvery === 0) checkpoint()
	},
	onEval: (step, result) =>
		logger.info(`validation @${step}`, { loss: Number(result.loss.toFixed(4)), perplexity: Number(result.perplexity.toFixed(2)), bitsPerByte: Number(bitsPerByte(result.loss, evalTargets.length, evalBytes).toFixed(3)), unigramBaseline: Number(baseline.toFixed(4)) })
})
checkpoint()

const seconds = (performance.now() - started) / 1000
logger.info(summary.stopped ? 'Stopped early' : 'Finished', {
	steps: trainer.step,
	tokensSeen: trainer.step * tokensPerStep,
	epochs: Number(((trainer.step * tokensPerStep) / train.count).toFixed(2)),
	minutes: Number((seconds / 60).toFixed(1)),
	finalValidationLoss: Number(lastEval.loss.toFixed(4)),
	perplexity: Number(lastEval.perplexity.toFixed(2)),
	bitsPerByte: Number(bitsPerByte(lastEval.loss, evalTargets.length, evalBytes).toFixed(3)),
	stableCheckpoint: store.stable()?.id ?? 'none',
	skippedSteps: trainer.skippedSteps
})
await trainer.close()
train.close()
validation.close()
