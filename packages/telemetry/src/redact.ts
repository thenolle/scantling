export const REDACTED = '[redacted]'

const MAX_DEPTH = 6
const MAX_ITEMS = 100
const SENSITIVE_KEY = /pass(word|wd)?|secret|token|api[-_]?key|authorization|cookie|credential|private[-_]?key/i

export interface SecretPattern {
	readonly name: string
	readonly pattern: RegExp
}

/**
 * Shapes of credentials that must never reach logs or training data.
 */
export const SECRET_PATTERNS: readonly SecretPattern[] = [
	{ name: 'private-key', pattern: /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/ },
	{ name: 'github-token', pattern: /\bgh[pousr]_[A-Za-z0-9]{36,}\b/ },
	{ name: 'github-fine-grained-token', pattern: /\bgithub_pat_[A-Za-z0-9_]{50,}\b/ },
	{ name: 'aws-access-key', pattern: /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/ },
	{ name: 'google-api-key', pattern: /\bAIza[0-9A-Za-z_-]{35}\b/ },
	{ name: 'slack-token', pattern: /\bxox[baprs]-[A-Za-z0-9-]{10,}\b/ },
	{ name: 'jwt', pattern: /\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b/ },
	{ name: 'bearer-token', pattern: /\bBearer\s+[A-Za-z0-9._~+/=-]{16,}/ },
	{ name: 'prefixed-secret-key', pattern: /\bsk-[A-Za-z0-9_-]{20,}\b/ },
	{ name: 'assigned-secret', pattern: /(?<=(?:password|passwd|secret|token|api[_-]?key)\s*[=:]\s*)["']?[^\s"',;]{6,}/i }
]

const GLOBAL_PATTERNS = SECRET_PATTERNS.map(({ name, pattern }) => ({ name, pattern: new RegExp(pattern.source, `${pattern.flags}g`) }))

/**
 * Replaces every secret-shaped substring.
 */
export function redactString(value: string): string {
	return GLOBAL_PATTERNS.reduce((text, { pattern }) => text.replace(pattern, REDACTED), value)
}

/**
 * Returns the names of the secret patterns found in `value`.
 */
export function findSecrets(value: string): string[] {
	return SECRET_PATTERNS.filter(({ pattern }) => pattern.test(value)).map(({ name }) => name)
}

/**
 * Patterns for scanning source code. Identical to `SECRET_PATTERNS` except that a name such as `password`
 * or `token` only counts when it is assigned a quoted literal that looks like a secret (16 or more characters
 * with letters and digits, no spaces) or a long unquoted token, so `password: string` and `token = sign(x)`
 * are not flagged.
 */
export const SOURCE_SECRET_PATTERNS: readonly SecretPattern[] = [
	...SECRET_PATTERNS.filter((entry) => entry.name !== 'assigned-secret'),
	{ name: 'assigned-literal', pattern: /(?<=(?:password|passwd|secret|token|api[_-]?key|private[_-]?key)["']?\s*[=:]\s*)(?:"(?=[^"\s]*\d)(?=[^"\s]*[A-Za-z])[^"\s]{16,}"|'(?=[^'\s]*\d)(?=[^'\s]*[A-Za-z])[^'\s]{16,}'|(?=[A-Za-z0-9+/_=-]*\d)(?=[A-Za-z0-9+/_=-]*[A-Za-z])[A-Za-z0-9+/_=-]{24,})/i }
]

/** Names of the source-code secret patterns found in `value`. */
export function findSourceSecrets(value: string): string[] {
	return SOURCE_SECRET_PATTERNS.filter(({ pattern }) => pattern.test(value)).map(({ name }) => name)
}

/**
 * Converts any value into a JSON-safe structure: secrets are redacted, sensitive keys are
 * masked, errors become plain objects, and cycles, deep nesting and huge collections are cut off.
 */
export function sanitize(value: unknown, depth = 0, seen: WeakSet<object> = new WeakSet()): unknown {
	if (typeof value === 'string') return redactString(value)
	if (typeof value === 'bigint') return value.toString()
	if (typeof value === 'function') return '[function]'
	if (typeof value === 'symbol') return value.toString()
	if (value === null || typeof value !== 'object') return value
	if (seen.has(value)) return '[circular]'
	if (depth >= MAX_DEPTH) return '[max depth]'
	if (value instanceof Date) return Number.isNaN(value.getTime()) ? 'invalid date' : value.toISOString()
	if (ArrayBuffer.isView(value)) return `[${value.constructor.name} ${value.byteLength} bytes]`
	seen.add(value)
	try {
		if (value instanceof Error) return { name: value.name, message: redactString(value.message), ...(value.stack === undefined ? {} : { stack: redactString(value.stack) }), ...(value.cause === undefined ? {} : { cause: sanitize(value.cause, depth + 1, seen) }) }
		if (value instanceof Map || value instanceof Set) return sanitize([...value], depth, seen)
		if (Array.isArray(value)) return value.slice(0, MAX_ITEMS).map((item) => sanitize(item, depth + 1, seen))
		const out: Record<string, unknown> = {}
		for (const [key, child] of Object.entries(value).slice(0, MAX_ITEMS)) out[key] = SENSITIVE_KEY.test(key) ? REDACTED : sanitize(child, depth + 1, seen)
		return out
	} finally {
		seen.delete(value)
	}
}
