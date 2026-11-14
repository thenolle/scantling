import type { AuditPolicy } from './policy.ts'
import { matchesPattern } from './policy.ts'
import type { Finding } from './types.ts'

const RUNTIME_FIELDS = ['dependencies', 'optionalDependencies', 'peerDependencies'] as const
const FORBIDDEN_FIELDS = ['bundledDependencies', 'bundleDependencies', 'overrides', 'resolutions', 'trustedDependencies'] as const
const LIFECYCLE_SCRIPTS = ['preinstall', 'install', 'postinstall'] as const

const IMPORT_PATTERN = /(?<![\w$.'"`])(?:(?:import|export)\b[^'";]*?\bfrom\s*['"]([^'"]+)['"]|import\s*['"]([^'"]+)['"]|import\(\s*['"]([^'"]+)['"]\s*\)|require\(\s*['"]([^'"]+)['"]\s*\))/g

function asRecord(value: unknown): Record<string, unknown> {
	return typeof value === 'object' && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : {}
}

function isWorkspaceName(name: string, policy: AuditPolicy): boolean {
	return policy.workspaceScopes.some((scope) => name.startsWith(scope))
}

function isAllowedSpecifier(specifier: string, policy: AuditPolicy): boolean {
	if (specifier.startsWith('.') || specifier.startsWith('/')) return true
	if (specifier === 'bun' || specifier.startsWith('bun:') || specifier.startsWith('node:')) return true
	return isWorkspaceName(specifier, policy) || matchesPattern(specifier, policy.allowedRuntimePatterns)
}

/**
 * Audits one `package.json`: dependency allowlist, workspace specs,
 * forbidden fields and install lifecycle scripts.
 */
export function checkManifest(file: string, text: string, policy: AuditPolicy): Finding[] {
	let manifest: Record<string, unknown>
	try {
		manifest = asRecord(JSON.parse(text))
	} catch (error) {
		const reason = error instanceof Error ? error.message : String(error)
		return [{ rule: 'dependency', file, message: `invalid JSON: ${reason}` }]
	}
	const findings: Finding[] = []
	const checkField = (field: string, allowDev: boolean): void => {
		for (const [name, spec] of Object.entries(asRecord(manifest[field]))) {
			const workspace = isWorkspaceName(name, policy)
			const allowed = workspace || matchesPattern(name, policy.allowedRuntimePatterns) || (allowDev && policy.allowedDevDependencies.includes(name))
			if (!allowed) findings.push({ rule: 'dependency', file, message: `"${name}" in ${field} is not on the allowlist` })
			else if (workspace && !(typeof spec === 'string' && spec.startsWith('workspace:'))) findings.push({ rule: 'spec', file, message: `"${name}" in ${field} must use a workspace: spec` })
		}
	}
	for (const field of RUNTIME_FIELDS) checkField(field, false)
	checkField('devDependencies', true)
	for (const field of FORBIDDEN_FIELDS) { if (field in manifest) findings.push({ rule: 'dependency', file, message: `"${field}" is not permitted` }) }
	const scripts = asRecord(manifest['scripts'])
	for (const script of LIFECYCLE_SCRIPTS) { if (script in scripts) findings.push({ rule: 'lifecycle', file, message: `"${script}" lifecycle script is not permitted` }) }
	return findings
}

/**
 * Audits one source file for unauthorized bare imports and external AI hosts.
 * The import scan is lexical, so a comment that resembles an import can be flagged.
 */
export function checkSource(file: string, text: string, policy: AuditPolicy): Finding[] {
	const findings: Finding[] = []
	const lineAt = (index: number): number => text.slice(0, index).split('\n').length
	for (const match of text.matchAll(IMPORT_PATTERN)) {
		const specifier = match[1] ?? match[2] ?? match[3] ?? match[4]
		if (specifier === undefined || isAllowedSpecifier(specifier, policy)) continue
		findings.push({ rule: 'import', file, line: lineAt(match.index ?? 0), message: `import of "${specifier}" is not permitted` })
	}
	const lower = text.toLowerCase()
	for (const host of policy.forbiddenHosts) {
		const index = lower.indexOf(host)
		if (index !== -1) findings.push({ rule: 'endpoint', file, line: lineAt(index), message: `references external AI host "${host}"` })
	}
	return findings
}

/**
 * Flags files whose extension belongs to a pretrained-weight format.
 */
export function checkModelFile(file: string, policy: AuditPolicy): Finding[] {
	const lower = file.toLowerCase()
	const extension = policy.forbiddenModelExtensions.find((ext) => lower.endsWith(ext))
	return extension === undefined ? [] : [{ rule: 'model-file', file, message: `pretrained-model format "${extension}" is not permitted` }]
}

/**
 * Extracts unique package names from the text of a `bun.lock` file,
 * ignoring workspace members.
 */
export function parseLockfile(text: string): string[] {
	const parsed: unknown = JSON.parse(text.replace(/,(\s*[}\]])/g, '$1'))
	const packages = asRecord(asRecord(parsed)['packages'])
	const names = new Set<string>()
	for (const value of Object.values(packages)) {
		const id = Array.isArray(value) ? value[0] : undefined
		if (typeof id !== 'string' || id.includes('@workspace:')) continue
		const at = id.lastIndexOf('@')
		names.add(at > 0 ? id.slice(0, at) : id)
	}
	return [...names].sort()
}

/**
 * Reports lockfile packages that are absent from the reviewed baseline.
 */
export function checkLockedPackages(locked: readonly string[], baseline: ReadonlySet<string>): Finding[] {
	return locked.filter((name) => !baseline.has(name)).map((name) => ({ rule: 'lockfile' as const, file: 'bun.lock', message: `"${name}" is locked but not in tools/audit.baseline.json` }))
}
