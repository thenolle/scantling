import { join } from 'node:path'

export interface AuditPolicy {
	readonly allowedDevDependencies: readonly string[]
	readonly allowedRuntimePatterns: readonly string[]
	readonly workspaceScopes: readonly string[]
	readonly forbiddenModelExtensions: readonly string[]
	readonly forbiddenHosts: readonly string[]
	readonly sourceRoots: readonly string[]
	readonly sourceExtensions: readonly string[]
	readonly ignoredDirectories: readonly string[]
}

const POLICY_KEYS: readonly (keyof AuditPolicy)[] = [
	'allowedDevDependencies',
	'allowedRuntimePatterns',
	'workspaceScopes',
	'forbiddenModelExtensions',
	'forbiddenHosts',
	'sourceRoots',
	'sourceExtensions',
	'ignoredDirectories'
]

function isStringArray(value: unknown): value is string[] {
	return Array.isArray(value) && value.every((item) => typeof item === 'string')
}

/**
 * Loads and validates `tools/audit.config.json`.
 * Throws when the file is missing or any policy field is not an array of strings.
 */
export async function loadPolicy(root: string): Promise<AuditPolicy> {
	const raw: unknown = await Bun.file(join(root, 'tools', 'audit.config.json')).json()
	if (typeof raw !== 'object' || raw === null) throw new Error('audit.config.json must contain an object')
	const record = raw as Record<string, unknown>
	const entries = POLICY_KEYS.map((key) => {
		const value = record[key]
		if (!isStringArray(value)) throw new Error(`audit.config.json: "${key}" must be an array of strings`)
		return [key, value] as const
	})
	return Object.fromEntries(entries) as unknown as AuditPolicy
}

/**
 * Matches a package name against exact names or `prefix*` patterns.
 */
export function matchesPattern(name: string, patterns: readonly string[]): boolean {
	return patterns.some((pattern) => pattern.endsWith('*') ? name.startsWith(pattern.slice(0, -1)) : name === pattern)
}
