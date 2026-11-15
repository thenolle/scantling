export const frame = (event: string, data: unknown): string => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`

/** The connected browsers. Each client is a function that writes one chunk to its stream. */
export class Hub {
	readonly #clients = new Set<(chunk: string) => void>()

	get size(): number {
		return this.#clients.size
	}

	add(send: (chunk: string) => void): () => void {
		this.#clients.add(send)
		return () => {
			this.#clients.delete(send)
		}
	}

	broadcast(event: string, data: unknown): void {
		const chunk = frame(event, data)
		for (const send of [...this.#clients]) send(chunk)
	}
}
