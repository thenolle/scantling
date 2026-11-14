import { dirname, isAbsolute, join, resolve } from 'node:path'
import type { ScantlingConfig } from './config.ts'
import { configSchema, validateConfig } from './config.ts'
import type { Issue } from './schema.ts'
import { leaves } from './schema.ts'

const DEFAULT_FILE = 'scantling.config.json'
const ENV_PREFIX = 'SCANTLING_'

export class ConfigError extends Error {
	readonly issues: readonly Issue[]

	constructor(issues: readonly Issue[]) {
		super(`Invalid configuration:\n${issues.map((issue) => `  ${issue.path}: ${issue.message}`).join('\n')}`)
		this.name = 'ConfigError'
		this.issues = issues
	}
}

export interface LoadOptions {
	/** Directory used to find the default file and resolve relative paths. Defaults to the working directory. */
	readonly cwd?: string
	/** Explicit configuration file. It must exist. Falls back to `SCANTLING_CONFIG`. */
	readonly file?: string
	/** Environment to read overrides from. Defaults to `process.env`. */
	readonly env?: Readonly<Record<string, string | undefined>>
}

export interface LoadedConfig {
	readonly config: ScantlingConfig
	/** The file that was read, when one existed. */
	readonly file: string | undefined
	/** Environment variables that overrode a setting. */
	readonly envOverrides: readonly string[]
	/** Directory against which relative paths were resolved. */
	readonly rootDir: string
}

type PlainObject = Record<string, unknown>

function isPlain(value: unknown): value is PlainObject {
	return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function merge(base: PlainObject, over: PlainObject): PlainObject {
	const out: PlainObject = { ...base }
	for (const [key, value] of Object.entries(over)) {
		if (key === '__proto__') continue
		const current = out[key]
		out[key] = isPlain(current) && isPlain(value) ? merge(current, value) : value
	}
	return out
}

function setNested(target: PlainObject, path: readonly string[], value: unknown): void {
	let cursor = target
	path.slice(0, -1).forEach((key) => {
		const next = cursor[key]
		const child: PlainObject = isPlain(next) ? next : {}
		cursor[key] = child
		cursor = child
	})
	const last = path[path.length - 1]
	if (last !== undefined) cursor[last] = value
}

function envName(path: readonly string[]): string {
	return ENV_PREFIX + path.map((key) => key.replace(/([a-z0-9])([A-Z])/g, '$1_$2').toUpperCase()).join('_')
}

function readEnvironment(env: Readonly<Record<string, string | undefined>>): { overrides: PlainObject, applied: string[], issues: Issue[] } {
	const overrides: PlainObject = {}
	const applied: string[] = []
	const issues: Issue[] = []
	for (const { path, schema } of leaves(configSchema)) {
		const name = envName(path)
		const raw = env[name]
		if (raw === undefined) continue
		if (schema.fromText === undefined) {
			issues.push({ path: name, message: 'cannot be set from the environment' })
			continue
		}
		setNested(overrides, path, schema.fromText(raw))
		applied.push(name)
	}
	return { overrides, applied, issues }
}

async function readJsonObject(file: string): Promise<PlainObject> {
	let parsed: unknown
	try {
		parsed = JSON.parse(await Bun.file(file).text())
	} catch (error) {
		const reason = error instanceof Error ? error.message : String(error)
		throw new ConfigError([{ path: file, message: `is not valid JSON: ${reason}` }])
	}
	if (!isPlain(parsed)) throw new ConfigError([{ path: file, message: 'must contain a JSON object' }])
	const { $schema: _ignored, ...rest } = parsed
	return rest
}

/**
 * Loads configuration in three layers: built-in defaults, the JSON file, then
 * `SCANTLING_<SECTION>_<SETTING>` environment variables (for example `SCANTLING_SERVER_PORT`).
 * Relative paths are resolved against the configuration file's directory, or the working directory
 * when there is no file. Throws `ConfigError` listing every invalid setting.
 */
export async function loadConfig(options: LoadOptions = {}): Promise<LoadedConfig> {
	const cwd = options.cwd ?? process.cwd()
	const env = options.env ?? process.env
	const explicit = options.file ?? env['SCANTLING_CONFIG']
	const file = resolve(cwd, explicit ?? DEFAULT_FILE)
	const present = await Bun.file(file).exists()
	if (!present && explicit !== undefined) throw new ConfigError([{ path: file, message: 'configuration file does not exist' }])
	const fromFile = present ? await readJsonObject(file) : {}
	const { overrides, applied, issues: envIssues } = readEnvironment(env)
	const result = configSchema.parse(merge(fromFile, overrides), '')
	const issues = [...envIssues, ...(result.ok ? validateConfig(result.value) : result.issues)]
	if (!result.ok || issues.length > 0) throw new ConfigError(issues)
	const rootDir = present ? dirname(file) : cwd
	const paths = Object.fromEntries(Object.entries(result.value.paths).map(([key, value]) => [key, isAbsolute(value) ? value : join(rootDir, value)])) as ScantlingConfig['paths']
	return {
		config: { ...result.value, paths },
		file: present ? file : undefined,
		envOverrides: applied,
		rootDir
	}
}
