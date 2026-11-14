import { CheckpointStore } from './store.ts'

const [command, ...rest] = Bun.argv.slice(2)
const option = (name: string): string | undefined => (rest.includes(name) ? rest[rest.indexOf(name) + 1] : undefined)
const store = new CheckpointStore(option('--dir') ?? 'data/checkpoints')

if (command === 'list') {
	const stable = store.stable()?.id
	for (const meta of store.list()) {
		const metrics = Object.entries(meta.metrics).map(([name, value]) => `${name} ${value.toFixed(4)}`).join(', ')
		console.log(`${meta.id}${meta.id === stable ? '  [stable]' : ''}  ${meta.createdAt}  ${metrics}`)
	}
} else if (command === 'verify') {
	const id = option('--id') ?? store.latest()
	if (id === undefined) throw new Error('There are no checkpoints')
	const state = store.load(id)
	console.log(`${id}: ${state.weights.size} weight tensors and ${state.optimizer.tensors.size} optimizer tensors verified`)
} else if (command === 'history') {
	for (const entry of store.history()) console.log(JSON.stringify(entry))
} else {
	console.error('Commands: list | verify [--id id] | history   (all accept --dir <directory>)')
	process.exit(1)
}
