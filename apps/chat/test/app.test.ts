import { expect, test } from 'bun:test'
import { Transformer } from '@scantling/nn'
import { trainBpe } from '@scantling/tokenizer'
import { createChatApp } from '../src/app.ts'
import { parseCompletion } from '../src/request.ts'

const { tokenizer } = trainBpe(['export function add(a: number, b: number) { return a + b }', 'const value = items.map((item) => item.id)'], { vocabSize: 320 })
const model = new Transformer({ vocabSize: tokenizer.vocabSize, dim: 16, layers: 1, heads: 2, maxSeqLen: 32 }, 1)
const app = createChatApp({ model, tokenizer, info: { name: 'test', step: 7 } })
const post = (body: unknown, host = '127.0.0.1'): Promise<Response> => app.fetch(new Request(`http://${host}:3300/api/complete`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }))
const events = (text: string): Record<string, unknown>[] => text.split('\n\n').filter((line) => line.startsWith('data: ')).map((line) => JSON.parse(line.slice(6)) as Record<string, unknown>)

test('the page and model information are served', async () => {
	const page = await app.fetch(new Request('http://localhost:3300/'))
	expect(page.headers.get('content-type')).toContain('text/html')
	expect(await page.text()).toContain('<textarea id="prompt"')
	const info = (await (await app.fetch(new Request('http://127.0.0.1:3300/api/model'))).json()) as Record<string, unknown>
	expect(info['name']).toBe('test')
	expect(info['context']).toBe(32)
	expect(info['parameters']).toBe(model.parameterCount())
})

test('a completion streams events and ends with a reason', async () => {
	const response = await post({ prompt: 'const', maxTokens: 6, temperature: 0.7 })
	expect(response.status).toBe(200)
	expect(response.headers.get('content-type')).toContain('text/event-stream')
	const items = events(await response.text())
	const last = items[items.length - 1]!
	expect(typeof last['done']).toBe('string')
	expect(last['tokens'] as number).toBeLessThanOrEqual(6)
})

test('greedy decoding is repeatable and requests are validated', async () => {
	const a = events(await (await post({ prompt: 'const', maxTokens: 8, temperature: 0 })).text())
	const b = events(await (await post({ prompt: 'const', maxTokens: 8, temperature: 0 })).text())
	expect(a).toEqual(b)
	expect((await post({ prompt: 'x', maxTokens: 100000 })).status).toBe(400)
	expect((await post({ prompt: 1 })).status).toBe(400)
	expect((await post({ prompt: 'x'.repeat(30_000) })).status).toBe(400)
	expect('error' in parseCompletion({ prompt: 'x', topP: 5 })).toBe(true)
	expect(parseCompletion({ prompt: 'x' })).toMatchObject({ maxTokens: 128, temperature: 0.7, raw: false, stopOnLoop: true })
})

test('only loopback hosts and same-origin posts are accepted', async () => {
	expect((await post({ prompt: 'x' }, 'evil.example')).status).toBe(403)
	const crossOrigin = await app.fetch(new Request('http://127.0.0.1:3300/api/complete', { method: 'POST', headers: { origin: 'http://evil.example', 'content-type': 'application/json' }, body: JSON.stringify({ prompt: 'x' }) }))
	expect(crossOrigin.status).toBe(403)
})

test('one generation runs at a time and cancelling frees the model', async () => {
	const first = await post({ prompt: 'a', maxTokens: 500 })
	expect(first.status).toBe(200)
	expect((await post({ prompt: 'b' })).status).toBe(429)
	await first.body!.cancel()
	const next = await post({ prompt: 'c', maxTokens: 1 })
	expect(next.status).toBe(200)
	await next.text()
})
