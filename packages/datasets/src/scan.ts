import { findSecrets } from '@scantling/telemetry'

const SENSITIVE_PATHS: readonly RegExp[] = [
	/(^|[\\/])\.env(\.|$)/i,
	/(^|[\\/])id_(rsa|dsa|ecdsa|ed25519)(\.pub)?$/i,
	/\.(pem|key|p12|pfx|jks|keystore)$/i,
	/(^|[\\/])\.(npmrc|netrc|pgpass)$/i,
	/(^|[\\/])(credentials?|secrets?)(\.[a-z]+)?$/i
]

const EMAIL = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g

/**
 * True for files that usually hold credentials or keys. They are never read into a dataset.
 */
export function isSensitivePath(path: string): boolean {
	return SENSITIVE_PATHS.some((pattern) => pattern.test(path))
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
	return { secrets: findSecrets(text), emails, text: redacted }
}
