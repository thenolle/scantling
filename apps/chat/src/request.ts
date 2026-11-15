export interface CompletionRequest {
	readonly prompt: string
	readonly maxTokens: number
	readonly temperature: number
	readonly topK: number
	readonly topP: number
	readonly repetitionPenalty: number
	readonly seed: number
	/** When true the prompt is encoded as given, without the start-of-file marker. */
	readonly raw: boolean
	readonly stopOnLoop: boolean
}

const MAX_PROMPT_CHARACTERS = 20_000

/** Validates a request body. Every field is bounded so that a request can never ask for unbounded work. */
export function parseCompletion(body: unknown): CompletionRequest | { readonly error: string } {
	if (typeof body !== 'object' || body === null || Array.isArray(body)) return { error: 'The body must be a JSON object' }
	const record = body as Record<string, unknown>
	const prompt = record['prompt']
	if (typeof prompt !== 'string') return { error: 'prompt must be a string' }
	if (prompt.length > MAX_PROMPT_CHARACTERS) return { error: `prompt is longer than ${MAX_PROMPT_CHARACTERS} characters` }
	const read = (key: string, fallback: number, min: number, max: number, whole = false): number => {
		const value = record[key]
		if (value === undefined) return fallback
		if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max || (whole && !Number.isInteger(value))) throw new RangeError(`${key} must be ${whole ? 'a whole number' : 'a number'} from ${min} to ${max}`)
		return value
	}
	try {
		const raw = record['raw']
		if (raw !== undefined && typeof raw !== 'boolean') throw new RangeError('raw must be true or false')
		const stopOnLoop = record['stopOnLoop']
		if (stopOnLoop !== undefined && typeof stopOnLoop !== 'boolean') throw new RangeError('stopOnLoop must be true or false')
		return {
			prompt,
			maxTokens: read('maxTokens', 128, 1, 512, true),
			temperature: read('temperature', 0.7, 0, 2),
			topK: read('topK', 40, 0, 1000, true),
			topP: read('topP', 0.95, 0.01, 1),
			repetitionPenalty: read('repetitionPenalty', 1.1, 1, 2),
			seed: read('seed', 1, 0, 2 ** 31, true),
			raw: raw ?? false,
			stopOnLoop: stopOnLoop ?? true
		}
	} catch (error) {
		return { error: error instanceof Error ? error.message : String(error) }
	}
}
