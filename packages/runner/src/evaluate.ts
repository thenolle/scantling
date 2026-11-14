import { CheckpointStore, applyWeights } from '@scantling/checkpoint'
import { TokenFile, fileEvaluationBatches } from '@scantling/datasets'
import { noGrad } from '@scantling/autodiff'
import { Transformer } from '@scantling/nn'
import type { ModelConfigInput } from '@scantling/nn'
import { Tokenizer } from '@scantling/tokenizer'
import { bitsPerByte, tokenBytes, unigramLoss } from './evaluation.ts'
import { evaluateCompletions, nextTokenAccuracy } from './quality.ts'

const args = Bun.argv.slice(2)
const option = (name: string): string | undefined => (args.includes(name) ? args[args.indexOf(name) + 1] : undefined)

const dataset = option('--dataset') ?? 'data/datasets/web'
const tokenizer = Tokenizer.fromJson(await Bun.file(option('--tokenizer') ?? 'data/tokenizer.json').text())
const store = new CheckpointStore(option('--checkpoint') ?? 'data/checkpoints/tiny-web')
const id = option('--id') ?? store.stable()?.id ?? store.latest()
if (id === undefined) throw new Error('There are no checkpoints to evaluate')
const state = store.load(id)
const model = new Transformer(state.config as ModelConfigInput)
applyWeights(model.namedParameters(), state.weights)

const train = TokenFile.open(`${dataset}/train.tokens`)
const validation = TokenFile.open(`${dataset}/validation.tokens`)
const length = Math.min(128, model.config.maxSeqLen)
const batches = fileEvaluationBatches(validation, length, 8, 16)
const targets = batches.flatMap((batch) => Array.from(batch.targets))

let total = 0
for (const batch of batches) total += noGrad(() => model.loss(batch.inputs, batch.targets, batch.batch, batch.length)).value.item() * batch.batch * batch.length
const loss = total / targets.length
const baseline = unigramLoss(train.read(0, Math.min(train.count, 20_000_000)), targets, tokenizer.vocabSize)
const accuracy = nextTokenAccuracy(model, batches)
const completions = evaluateCompletions(model, tokenizer, validation, { prompts: Number(option('--prompts') ?? 50), promptTokens: 48, newTokens: 64, seed: 7 })
const percent = (value: number): string => `${(value * 100).toFixed(1)}%`

console.log(`Checkpoint        ${id} (step ${state.step}), ${model.parameterCount()} parameters`)
console.log(`Validation loss   ${loss.toFixed(4)}  perplexity ${Math.exp(loss).toFixed(2)}  ${bitsPerByte(loss, targets.length, tokenBytes(tokenizer, targets)).toFixed(3)} bits per byte  (unigram baseline ${baseline.toFixed(4)})`)
console.log(`Next token        top-1 ${percent(accuracy.top1)}   top-5 ${percent(accuracy.top5)}   over ${accuracy.tokens} tokens`)
console.log(`Completions       ${completions.meanMatchingPrefix.toFixed(2)} leading tokens match on average; first 8 exact in ${percent(completions.firstEightExact)}`)
console.log(`Brackets          mismatched in ${percent(completions.generatedMismatch)} of generated text, against ${percent(completions.realMismatch)} of the real code`)
for (const [index, sample] of completions.samples.entries()) console.log(`\n--- sample ${index + 1} ---\nprompt  ...${sample.prompt}\nreal    ${JSON.stringify(sample.expected.slice(0, 200))}\nmodel   ${JSON.stringify(sample.generated.slice(0, 200))}`)
train.close()
validation.close()
