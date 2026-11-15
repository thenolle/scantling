import { Tokenizer } from '@scantling/tokenizer'

interface Scope {
	onmessage: ((event: { data: unknown }) => void) | null
	postMessage(message: unknown, transfer?: ArrayBuffer[]): void
}

const scope = globalThis as unknown as Scope
let tokenizer: Tokenizer | undefined

scope.onmessage = (event) => {
	const message = event.data as { kind: 'init'; json: string } | { kind: 'job'; id: number; texts: string[] }
	if (message.kind === 'init') {
		tokenizer = Tokenizer.fromJson(message.json)
		scope.postMessage({ kind: 'ready' })
		return
	}
	const results = message.texts.map((text) => Int32Array.from(tokenizer!.encode(text)))
	scope.postMessage({ kind: 'done', id: message.id, results }, results.map((result) => result.buffer as ArrayBuffer))
}
