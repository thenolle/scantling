import { createApp } from '@gambrel/core'
import type { App } from '@gambrel/core'
import { Hub, frame } from './hub.ts'
import { monitorPage } from './page.ts'
import type { SystemSample } from './system.ts'
import type { RunTracker } from './tracker.ts'

export interface MonitorOptions {
	readonly tracker: RunTracker
	readonly hub: Hub
	readonly system: () => readonly SystemSample[]
}

const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]'])

/** Serves the page, a JSON snapshot and the event stream. Only loopback host names are answered. */
export function createMonitorApp({ tracker, hub, system }: MonitorOptions): App {
	const encoder = new TextEncoder()
	const snapshot = (): unknown => ({ ...tracker.snapshot(), system: system(), now: Date.now() })
	return createApp()
		.use((context, next) => (LOCAL_HOSTS.has(context.url.hostname) ? next() : new Response('Forbidden: this server only answers on localhost', { status: 403 })))
		.get('/', () => new Response(monitorPage(), { headers: { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store', 'content-security-policy': "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'" } }))
		.get('/api/snapshot', () => Response.json(snapshot()))
		.get('/events', (context) => {
			let remove: (() => void) | undefined
			const stream = new ReadableStream<Uint8Array>({
				start(controller) {
					const send = (chunk: string): void => {
						try {
							controller.enqueue(encoder.encode(chunk))
						} catch {
							remove?.()
						}
					}
					send(frame('snapshot', snapshot()))
					remove = hub.add(send)
					context.request.signal.addEventListener('abort', () => remove?.())
				},
				cancel() {
					remove?.()
				}
			})
			return new Response(stream, { headers: { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-store' } })
		})
}
