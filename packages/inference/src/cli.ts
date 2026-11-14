import { CheckpointStore, applyWeights } from '@scantling/checkpoint'
import { presetConfig, Transformer } from '@scantling/nn'
import type { ModelConfigInput } from '@scantling/nn'
import { StreamDecoder, Tokenizer } from '@scantling/tokenizer'
import { generate } from './generate.ts'

const args = Bun.argv.slice(2)
const option = (name: string): string | undefined => (args.includes(name) ? args[args.indexOf(name) + 1] : undefined)
const number = (name: string): number | undefined => (option(name) === undefined ? undefined : Number(option(name)))

const tokenizer = Tokenizer.fromJson(await Bun.file(option('--tokenizer') ?? 'data/tokenizer.json').text())
let model: Transformer
if (args.includes('--untrained')) {
	model = new Transformer(presetConfig('tiny', tokenizer.vocabSize), number('--seed') ?? 1)
	console.error('Using random weights: the output is noise. Pass --checkpoint to load a trained model.')
} else {
	const store = new CheckpointStore(option('--checkpoint') ?? 'data/checkpoints')
	const id = option('--id') ?? store.stable()?.id ?? store.latest()
	if (id === undefined) throw new Error('There are no checkpoints. Train a model first, or pass --untrained.')
	const state = store.load(id)
	model = new Transformer(state.config as ModelConfigInput)
	applyWeights(model.namedParameters(), state.weights)
	console.error(`Loaded ${id}`)
}

const prompt = [tokenizer.idOf('<|file|>'), ...tokenizer.encode(option('--prompt') ?? '')]
const trace = args.includes('--trace') ? 5 : 0
const decoder = new StreamDecoder(tokenizer, { skipSpecial: true })
const started = performance.now()
let tokens = 0
for await (const event of generate(model, prompt, {
	engine: (option('--engine') ?? 'float32') as 'tensor' | 'float32' | 'int8',
	maxNewTokens: number('--max') ?? 100,
	seed: number('--seed') ?? 1,
	sampling: {
		temperature: number('--temperature') ?? 0.8,
		topK: number('--top-k') ?? 40,
		topP: number('--top-p') ?? 0.95,
		repetitionPenalty: number('--repetition') ?? 1.1,
		banned: [tokenizer.idOf('<|pad|>')]
	},
	stopTokens: [tokenizer.idOf('<|endfile|>'), tokenizer.idOf('<|eos|>')],
	text: decoder,
	trace
})) {
	if (event.type === 'token') {
		tokens += 1
		process.stdout.write(event.text)
		if (trace > 0) console.error(`\n  #${event.position} p=${event.probability.toFixed(3)} ${event.candidates.map((candidate) => `${JSON.stringify(tokenizer.decode([candidate.token]))}:${candidate.probability.toFixed(2)}`).join(' ')}`)
	} else {
		process.stdout.write(event.tail)
		console.error(`\n[${event.reason}: ${tokens} tokens in ${((performance.now() - started) / 1000).toFixed(1)} s]`)
	}
}
