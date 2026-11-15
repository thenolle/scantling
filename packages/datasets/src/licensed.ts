import { readdir } from 'node:fs/promises'
import { join, relative } from 'node:path'

const ALLOWED = new Set(['zlib', 'mit', 'isc', 'bsd-2-clause', 'bsd-3-clause', 'apache-2.0', '0bsd', 'unlicense', 'cc0-1.0', 'blueoak-1.0.0'])
const SOURCE = /\.(ts|tsx|mts|cts|js|jsx|mjs|cjs)$/i
const LICENSE_FILE = /^(licen[sc]e|copying|notice)(\.[a-z]+)?$/i
const MAX_FILE_BYTES = 200_000

/**
 * True when an SPDX licence expression can be satisfied with permissive licences only: an `OR` needs one
 * allowed option and an `AND` needs all of its parts to be allowed. Unknown, missing or custom licences are refused.
 */
export function isPermissive(expression: string | undefined): boolean {
	if (expression === undefined) return false
	const alternatives = expression.replace(/[()]/g, '').split(/\s+OR\s+/i)
	return alternatives.some((alternative) => alternative.split(/\s+AND\s+/i).every((id) => ALLOWED.has(id.trim().toLowerCase())))
}

function licenseOf(pkg: Record<string, unknown>): string | undefined {
	const value = pkg['license']
	if (typeof value === 'string') return value
	if (typeof value === 'object' && value !== null && typeof (value as { type?: unknown }).type === 'string') return (value as { type: string }).type
	const list = pkg['licenses']
	if (Array.isArray(list)) {
		const types = list.map((entry) => (typeof entry === 'object' && entry !== null ? (entry as { type?: unknown }).type : undefined)).filter((type): type is string => typeof type === 'string')
		return types.length > 0 ? types.join(' OR ') : undefined
	}
	return undefined
}

async function entriesOf(directory: string) {
	return (await readdir(directory, { withFileTypes: true }).catch(() => [])).sort((a, b) => a.name.localeCompare(b.name))
}

async function findModules(root: string, depth = 0, found: string[] = []): Promise<string[]> {
	for (const entry of await entriesOf(root)) {
		if (!entry.isDirectory() || entry.name.startsWith('.')) continue
		if (entry.name === 'node_modules') found.push(join(root, entry.name))
		else if (depth < 6) await findModules(join(root, entry.name), depth + 1, found)
	}
	return found
}

async function packageDirectories(modules: string): Promise<string[]> {
	const found: string[] = []
	for (const entry of await entriesOf(modules)) {
		if (entry.name.startsWith('.') || !entry.isDirectory()) continue
		if (entry.name.startsWith('@')) {
			for (const inner of await entriesOf(join(modules, entry.name))) if (inner.isDirectory()) found.push(join(modules, entry.name, inner.name))
		} else {
			found.push(join(modules, entry.name))
		}
	}
	return found
}

async function* sourceFiles(directory: string): AsyncGenerator<string> {
	for (const entry of await entriesOf(directory)) {
		if (entry.name === 'node_modules' || entry.name === '.git') continue
		const path = join(directory, entry.name)
		if (entry.isDirectory()) yield* sourceFiles(path)
		else if (entry.isFile() && SOURCE.test(entry.name) && !/\.min\./i.test(entry.name) && Bun.file(path).size <= MAX_FILE_BYTES) yield path
	}
}

export interface StageOptions {
	readonly from: string
	readonly to: string
	readonly maxBytes: number
	/** Bun global cache folders: each holds `name@version@@@1` package folders, scoped ones under an `@scope` folder. */
	readonly caches?: readonly string[]
}

export interface StagedPackage {
	readonly package: string
	readonly version: string
	readonly license: string
	readonly files: number
	readonly bytes: number
}

export interface StageReport {
	readonly staged: readonly StagedPackage[]
	/** Packages refused, counted once each, by their licence text. */
	readonly rejected: ReadonlyMap<string, number>
	readonly bytes: number
}

async function cachePackageDirectories(cache: string): Promise<string[]> {
	const found: string[] = []
	for (const entry of await entriesOf(cache)) {
		if (!entry.isDirectory() || entry.name.startsWith('.')) continue
		if (entry.name.startsWith('@') && !entry.name.includes('@', 1)) {
			for (const inner of await entriesOf(join(cache, entry.name))) if (inner.isDirectory()) found.push(join(cache, entry.name, inner.name))
		} else {
			found.push(join(cache, entry.name))
		}
	}
	return found
}

/**
 * Copies the source of permissively licensed packages found in `node_modules` folders under `from` into `to`,
 * once per name and version, together with their licence files, and records what it copied in `licenses.json`.
 */
export async function stageLicensed(options: StageOptions): Promise<StageReport> {
	const candidates = new Map<string, { directory: string; name: string; version: string; license: string }>()
	const rejected = new Map<string, number>()
	const seen = new Set<string>()
	const consider = async (directory: string): Promise<void> => {
		const pkg = (await Bun.file(join(directory, 'package.json')).json().catch(() => undefined)) as Record<string, unknown> | undefined
		if (pkg === undefined || typeof pkg['name'] !== 'string') return
		const version = typeof pkg['version'] === 'string' ? pkg['version'] : '0.0.0'
		const key = `${pkg['name']}@${version}`
		if (seen.has(key)) return
		seen.add(key)
		const license = licenseOf(pkg)
		if (isPermissive(license)) candidates.set(key, { directory, name: pkg['name'], version, license: license! })
		else rejected.set(license ?? 'none declared', (rejected.get(license ?? 'none declared') ?? 0) + 1)
	}
	for (const modules of await findModules(options.from)) for (const directory of await packageDirectories(modules)) await consider(directory)
	for (const cache of options.caches ?? []) for (const directory of await cachePackageDirectories(cache)) await consider(directory)
	const ordered = [...candidates.entries()].sort((a, b) => String(Bun.hash(a[0])).localeCompare(String(Bun.hash(b[0]))))
	const staged: StagedPackage[] = []
	let total = 0
	for (const [key, candidate] of ordered) {
		if (total >= options.maxBytes) break
		const folder = join(options.to, key.replace('/', '__'))
		let files = 0
		let bytes = 0
		for await (const path of sourceFiles(candidate.directory)) {
			await Bun.write(join(folder, relative(candidate.directory, path)), Bun.file(path))
			files += 1
			bytes += Bun.file(path).size
		}
		if (files === 0) continue
		for (const entry of await entriesOf(candidate.directory)) {
			if (entry.isFile() && LICENSE_FILE.test(entry.name) && Bun.file(join(candidate.directory, entry.name)).size <= 50_000) {
				await Bun.write(join(folder, entry.name), Bun.file(join(candidate.directory, entry.name)))
			}
		}
		staged.push({ package: candidate.name, version: candidate.version, license: candidate.license, files, bytes })
		total += bytes
	}
	await Bun.write(join(options.to, 'licenses.json'), JSON.stringify({ createdAt: new Date().toISOString(), packages: staged }, null, 2))
	return { staged, rejected, bytes: total }
}

if (import.meta.main) {
	const args = Bun.argv.slice(2)
	const option = (name: string): string | undefined => (args.includes(name) ? args[args.indexOf(name) + 1] : undefined)
	const from = option('--from')
	const to = option('--to')
	const cache = option('--cache')
	if (from === undefined || to === undefined) throw new Error('Usage: corpus --from <folder with projects> --to <output folder> [--max-mb 200]')
	const report = await stageLicensed({ from, to, maxBytes: Number(option('--max-mb') ?? 200) * 1_048_576, ...(cache === undefined ? {} : { caches: [cache] }) })
	console.log(`Staged ${report.staged.length} packages, ${(report.bytes / 1_048_576).toFixed(1)} MiB, into ${to}`)
	const refused = [...report.rejected].sort((a, b) => b[1] - a[1]).slice(0, 8)
	console.log(`Refused ${[...report.rejected.values()].reduce((sum, count) => sum + count, 0)} packages by licence: ${refused.map(([license, count]) => `${license} x${count}`).join(', ')}`)
}
