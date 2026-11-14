import { findSourceSecrets } from '@scantling/telemetry'

const SENSITIVE_PATHS: readonly RegExp[] = [
	/(^|[\\/])\.env(\.|$)/i,
	/(^|[\\/])id_(rsa|dsa|ecdsa|ed25519)(\.pub)?$/i,
	/\.(pem|key|p12|pfx|jks|keystore)$/i,
	/(^|[\\/])\.(npmrc|netrc|pgpass)$/i,
	/(^|[\\/])(credentials?|secrets?)(\.[a-z]+)?$/i,
	/(^|[\\/])(secrets?|credentials?|private|\.ssh|\.aws|\.gnupg)[\\/]/i,
	/(^|[\\/])[^\\/]*(secret|private[_-]?key|backup[_-]?key)[^\\/]*$/i
]

const GENERATED_PATHS: readonly RegExp[] = [/(^|[\\/])generated([\\/]|$)/i, /\.min\.(js|css)$/i, /\.generated\./i]
const EMAIL = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g
const KEPT_EXTENSIONS = new Set(['.ts', '.tsx', '.mts', '.cts', '.js', '.jsx', '.mjs', '.cjs', '.css', '.scss', '.html', '.md', '.mdx', '.sql', '.prisma', '.sh', '.ps1', '.yml', '.yaml', '.toml'])
const KEPT_NAMES = new Set(['dockerfile', 'makefile'])
const JSON_LIMIT = 20_000
const LONE_STRING = /^\s*(['"`])[^'"`\\]{1,80}\1,?\s*$/

/** True for files that usually hold credentials or keys. They are never read into a dataset. */
export function isSensitivePath(path: string): boolean {
	return SENSITIVE_PATHS.some((pattern) => pattern.test(path))
}

/** True for machine-generated or minified files, which teach a model nothing about how people write code. */
export function isGeneratedPath(path: string): boolean {
	return GENERATED_PATHS.some((pattern) => pattern.test(path))
}

export interface ScanResult {
	/** Names of the credential patterns found. A non-empty list means the document must be excluded. */
	readonly secrets: readonly string[]
	/** Number of email addresses that were redacted. */
	readonly emails: number
	/** The text with email addresses replaced by `user@example.com`. */
	readonly text: string
}

/**
 * Looks for credentials and redacts personal email addresses. Credentials are never redacted
 * in place: the caller drops the whole document.
 */
export function scanDocument(text: string): ScanResult {
	let emails = 0
	const redacted = text.replace(EMAIL, () => {
		emails += 1
		return 'user@example.com'
	})
	return { secrets: findSourceSecrets(text), emails, text: redacted }
}

/** True for source, markup, documentation and small configuration files. Data files, images and lists are not kept. */
export function isCodeLikePath(path: string, length: number): boolean {
	const name = (path.split(/[\\/]/).pop() ?? '').toLowerCase()
	const dot = name.lastIndexOf('.')
	const extension = dot <= 0 ? '' : name.slice(dot)
	if (extension === '.json') return length <= JSON_LIMIT
	return KEPT_EXTENSIONS.has(extension) || KEPT_NAMES.has(name)
}

/** True for long files that are mostly one short quoted string per line: word lists and similar data. */
export function isDataList(text: string): boolean {
	const lines = text.split('\n').filter((line) => line.trim().length > 0)
	if (lines.length < 60) return false
	let strings = 0
	for (const line of lines) if (LONE_STRING.test(line)) strings += 1
	return strings / lines.length >= 0.6
}

/** True for text whose lines are so long that it is almost certainly minified or bundled. */
export function isMinifiedText(text: string): boolean {
	return text.length > 2000 && text.length / text.split('\n').length > 300
}
