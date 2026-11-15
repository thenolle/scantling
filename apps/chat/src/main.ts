import { serve } from '@gambrel/bun'
import { CheckpointStore, applyWeights } from '@scantling/checkpoint'
import { Transformer } from '@scantling/nn'
import { Tokenizer } from '@scantling/tokenizer'
import { createChatApp } from './app.ts'

const args = Bun.argv.slice(2)
const option = (name: string): string | undefined => (args.includes(name) ? args[args.indexOf(name) + 1] : undefined)

if (args.includes('--help')) {
	console.log('bun run chat -- [--checkpoint data/checkpoints/small-mix-gpu] [--id step-00006000] [--tokenizer data/tokenizer.json] [--port 3300] [--context 128]')
	process.exit(0)
}

const directory = option('--checkpoint') ?? 'data/checkpoints/small-mix-gpu'
const store = new CheckpointStore(directory)
const id = option('--id') ?? store.stable()?.id ?? store.latest()
if (id === undefined) throw new Error(`There are no checkpoints in ${directory}. Train a model first, or pass --checkpoint.`)
const tokenizer = Tokenizer.fromJson(await Bun.file(option('--tokenizer') ?? 'data/tokenizer.json').text())
const state = store.load(id)
const trainedLength = Number(option('--context') ?? 128)
const model = new Transformer({ ...state.config, maxSeqLen: Math.min(trainedLength, state.config.maxSeqLen) })
applyWeights(model.namedParameters(), state.weights)
if (model.config.vocabSize !== tokenizer.vocabSize) throw new Error(`The checkpoint has a vocabulary of ${model.config.vocabSize} but the tokenizer has ${tokenizer.vocabSize}`)

const port = Number(option('--port') ?? 3300)
const server = serve(createChatApp({ model, tokenizer, info: { name: `${directory.split(/[\\/]/).pop()}/${id}`, step: state.step } }), { port, hostname: '127.0.0.1' })
console.log(`Loaded ${id} (${model.parameterCount().toLocaleString()} parameters). Open ${server.url.href}`)
