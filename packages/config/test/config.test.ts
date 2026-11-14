import { afterEach, expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { ConfigError, loadConfig } from '../src/index.ts'

const roots: string[] = []

afterEach(async () => {
	await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })))
})

async function workspace(file?: unknown): Promise<string> {
	const root = await mkdtemp(join(tmpdir(), 'scantling-config-'))
	roots.push(root)
	if (file !== undefined) await Bun.write(join(root, 'scantling.config.json'), typeof file === 'string' ? file : JSON.stringify(file))
	return root
}

async function issuesOf(promise: Promise<unknown>): Promise<string[]> {
	try {
		await promise
	} catch (error) {
		if (error instanceof ConfigError) return error.issues.map((issue) => issue.path)
		throw error
	}
	return []
}

test('defaults are valid, local-only and keep training off', async () => {
	const cwd = await workspace()
	const { config, file } = await loadConfig({ cwd, env: {} })
	expect(file).toBeUndefined()
	expect(config.server.host).toBe('127.0.0.1')
	expect(config.privacy.trainingAllowed).toBe(false)
	expect(config.paths.dataDir).toBe(join(cwd, 'data'))
})

test('the file overrides defaults', async () => {
	const cwd = await workspace({ server: { port: 5000 } })
	const { config } = await loadConfig({ cwd, env: {} })
	expect(config.server.port).toBe(5000)
	expect(config.server.host).toBe('127.0.0.1')
})

test('environment variables override the file and are coerced', async () => {
	const cwd = await workspace({ server: { port: 5000 } })
	const loaded = await loadConfig({
		cwd,
		env: { SCANTLING_SERVER_PORT: '6000', SCANTLING_PRIVACY_TRAINING_ALLOWED: 'true', SCANTLING_PATHS_DATA_DIR: 'elsewhere' }
	})
	expect(loaded.config.server.port).toBe(6000)
	expect(loaded.config.privacy.trainingAllowed).toBe(true)
	expect(loaded.config.paths.dataDir).toBe(join(cwd, 'elsewhere'))
	expect(loaded.envOverrides).toContain('SCANTLING_SERVER_PORT')
})

test('unknown keys and wrong types are reported with their paths', async () => {
	const cwd = await workspace({ server: { port: 'x', prot: 1 }, logging: { level: 'loud' } })
	expect(await issuesOf(loadConfig({ cwd, env: {} }))).toEqual(['server.port', 'server.prot', 'logging.level'])
})

test('refuses a non-loopback host unless remote access is allowed', async () => {
	const cwd = await workspace({ server: { host: '0.0.0.0' } })
	expect(await issuesOf(loadConfig({ cwd, env: {} }))).toEqual(['server.host'])
	const allowed = await workspace({ server: { host: '0.0.0.0', allowRemote: true } })
	expect((await loadConfig({ cwd: allowed, env: {} })).config.server.host).toBe('0.0.0.0')
})

test('reports invalid JSON and missing explicit files', async () => {
	const broken = await workspace('{ nope')
	expect(loadConfig({ cwd: broken, env: {} })).rejects.toThrow(ConfigError)
	const empty = await workspace()
	expect(loadConfig({ cwd: empty, file: 'missing.json', env: {} })).rejects.toThrow(ConfigError)
})

test('rejects an invalid environment value', async () => {
	const cwd = await workspace()
	expect(await issuesOf(loadConfig({ cwd, env: { SCANTLING_SERVER_PORT: 'abc' } }))).toEqual(['server.port'])
})
