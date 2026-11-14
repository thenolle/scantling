import { createApp } from '@gambrel/core'
import type { App } from '@gambrel/core'
import { generate } from '@scantling/inference'
import type { Transformer } from '@scantling/nn'
import { StreamDecoder } from '@scantling/tokenizer'
import type { Tokenizer } from '@scantling/tokenizer'
import { chatPage } from './page.ts'
import { parseCompletion } from './request.ts'

export interface ModelInfo {
	readonly name: string
	readonly step: number
}

export interface ChatOptions {
	readonly model: Transformer
	readonly tokenizer: Tokenizer
	readonly info: ModelInfo
}

const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]'])
const MAX_BODY_BYTES = 100_000
const encoder = new TextEncoder()

const message = (text: string, status: number): Response => new Response(text, { status, headers: { 'content-type': 'text/plain; charset=utf-8' } })
const event = (payload: unknown): Uint8Array => encoder.encode(`data: ${JSON.stringify(payload)}\n\n`)

/**
 * The completion playground. The server accepts only requests addressed to a loopback name (which defeats DNS
 * rebinding) and, for posts, from its own origin. One generation runs at a time; others receive 429.
 */
export function createChatApp(options: ChatOptions): App {
	const { model, tokenizer, info } = options
	const stopTokens = [tokenizer.idOf('<|endfile|>'), tokenizer.idOf('<|eos|>')]
	const banned = [tokenizer.idOf('<|pad|>')]
	let busy = false
	return createApp()
		.use((context, next) => {
			if (!LOCAL_HOSTS.has(context.url.hostname)) return message('Forbidden: this server only answers on localhost', 403)
			const origin = context.request.headers.get('origin')
			if (context.method !== 'GET' && origin !== null && new URL(origin).host !== context.url.host) return message('Forbidden: cross-origin request', 403)
			return next()
		})
		.get('/', () => new Response(chatPage(), { headers: { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store', 'content-security-policy': "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'" } }))
		.get('/api/model', () => Response.json({ name: info.name, step: info.step, parameters: model.parameterCount(), context: model.config.maxSeqLen, vocabulary: tokenizer.vocabSize }))
		.post('/api/complete', async (context) => {
			const length = Number(context.request.headers.get('content-length') ?? 0)
			if (length > MAX_BODY_BYTES) return message('The request is too large', 413)
			let body: unknown
			try {
				body = await context.request.json()
			} catch {
				return message('The body must be valid JSON', 400)
			}
			const request = parseCompletion(body)
			if ('error' in request) return message(request.error, 400)
			const ids = request.raw ? tokenizer.encode(request.prompt) : [tokenizer.idOf('<|file|>'), ...tokenizer.encode(request.prompt)]
			if (ids.length === 0) return message('prompt is empty', 400)
			if (busy) return message('The model is busy with another request', 429)
			busy = true
			let released = false
			const release = (): void => {
				if (!released) {
					released = true
					busy = false
				}
			}
			const abort = new AbortController()
			context.request.signal.addEventListener('abort', () => abort.abort())
			const sampling = request.temperature === 0 ? { repetitionPenalty: request.repetitionPenalty, banned } : { temperature: request.temperature, topK: request.topK, topP: request.topP, repetitionPenalty: request.repetitionPenalty, banned }
			const stream = new ReadableStream<Uint8Array>({
				async start(controller) {
					try {
						for await (const item of generate(model, ids, { engine: 'float32', maxNewTokens: request.maxTokens, seed: request.seed, sampling, stopTokens, text: new StreamDecoder(tokenizer, { skipSpecial: true }), signal: abort.signal })) {
							if (item.type === 'token') {
								if (item.text !== '') controller.enqueue(event({ text: item.text }))
							} else {
								controller.enqueue(event({ done: item.reason, tokens: item.tokens, text: item.tail }))
							}
						}
					} catch (error) {
						controller.enqueue(event({ error: error instanceof Error ? error.message : String(error) }))
					} finally {
						release()
						try {
							controller.close()
						} catch {
							// The client already went away.
						}
					}
				},
				cancel() {
					abort.abort()
					release()
				}
			})
			return new Response(stream, { headers: { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-store' } })
		})
}
