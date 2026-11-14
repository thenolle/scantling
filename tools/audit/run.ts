import { basename, extname, join } from 'node:path'
import { checkLockedPackages, checkManifest, checkModelFile, checkSource, parseLockfile } from './checks.ts'
import { toPosix, walk } from './fs.ts'
import type { AuditPolicy } from './policy.ts'
import type { Finding } from './types.ts'

const BASELINE_PATH = ['tools', 'audit.baseline.json'] as const

function isSourceFile(file: string, policy: AuditPolicy): boolean {
	return (policy.sourceExtensions.includes(extname(file)) && policy.sourceRoots.some((root) => file.startsWith(`${root}/`)))
}

async function readBaseline(root: string): Promise<ReadonlySet<string> | undefined> {
	const file = Bun.file(join(root, ...BASELINE_PATH))
	if (!(await file.exists())) return undefined
	const raw: unknown = await file.json()
	const packages = typeof raw === 'object' && raw !== null ? (raw as { packages?: unknown }).packages : undefined
	if (!Array.isArray(packages) || !packages.every((item) => typeof item === 'string')) throw new Error('audit.baseline.json must contain a "packages" array of strings')
	return new Set<string>(packages)
}

async function auditLockfile(root: string): Promise<Finding[]> {
	const lockfile = Bun.file(join(root, 'bun.lock'))
	if (!(await lockfile.exists())) return [{ rule: 'lockfile', file: 'bun.lock', message: 'bun.lock is missing; commit the lockfile' }]
	const baseline = await readBaseline(root)
	if (baseline === undefined) return [{ rule: 'lockfile', file: 'tools/audit.baseline.json', message: 'baseline is missing; run "bun tools/audit-deps.ts --write-baseline" and review it' }]
	return checkLockedPackages(parseLockfile(await lockfile.text()), baseline)
}

/**
 * Runs every audit check against the repository at `root`.
 * Returns findings sorted by rule, then file; an empty array means the audit passed.
 */
export async function runAudit(root: string, policy: AuditPolicy): Promise<Finding[]> {
	const findings: Finding[] = []
	const ignored = new Set(policy.ignoredDirectories)
	for await (const path of walk(root, ignored)) {
		const file = toPosix(root, path)
		findings.push(...checkModelFile(file, policy))
		if (basename(file) === 'package.json') findings.push(...checkManifest(file, await Bun.file(path).text(), policy))
		else if (isSourceFile(file, policy)) findings.push(...checkSource(file, await Bun.file(path).text(), policy))
	}
	findings.push(...(await auditLockfile(root)))
	return findings.sort((a, b) => a.rule.localeCompare(b.rule) || a.file.localeCompare(b.file))
}

/**
 * Records the packages currently in `bun.lock` as the reviewed baseline.
 * Returns the number of packages written.
 */
export async function writeBaseline(root: string): Promise<number> {
	const names = parseLockfile(await Bun.file(join(root, 'bun.lock')).text())
	await Bun.write(join(root, ...BASELINE_PATH), `${JSON.stringify({ packages: names }, null, 2)}\n`)
	return names.length
}
