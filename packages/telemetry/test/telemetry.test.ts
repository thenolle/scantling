import { afterEach, expect, test } from 'bun:test'
import { existsSync, readFileSync } from 'node:fs'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createLogger, FileSink, findSecrets, MemorySink, REDACTED, redactString, sanitize, throughput, timed } from '../src/index.ts'
import type { LogEntry, LogSink } from '../src/index.ts'

const githubToken = `ghp_${'a'.repeat(36)}`
const awsKey = `AKIA${'ABCDEFGHIJKLMNOP'}`
const jwt = ['eyJhbGciOiJIUzI1NiJ9', 'eyJzdWIiOiIxMjM0NTY3ODkw', 'c2lnbmF0dXJlc2lnbmF0dXJl'].join('.')

const roots: string[] = []

afterEach(async () => { await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true }))) })

test('redacts credential-shaped strings', () => {
	const text = redactString(`token ${githubToken} key ${awsKey} jwt ${jwt} password=hunter2hunter2`)
	expect(text).not.toContain(githubToken)
	expect(text).not.toContain(awsKey)
	expect(text).not.toContain(jwt)
	expect(text).not.toContain('hunter2hunter2')
	expect(findSecrets(`Bearer ${'x'.repeat(24)}`)).toEqual(['bearer-token'])
	expect(findSecrets('nothing sensitive here')).toEqual([])
})

test('masks sensitive keys and survives cycles', () => {
	const value: Record<string, unknown> = { apiKey: 'abc', nested: { password: 'p', ok: 1 } }
	value['self'] = value
	const result = sanitize(value) as Record<string, unknown>
	expect(result['apiKey']).toBe(REDACTED)
	expect((result['nested'] as Record<string, unknown>)['password']).toBe(REDACTED)
	expect((result['nested'] as Record<string, unknown>)['ok']).toBe(1)
	expect(result['self']).toBe('[circular]')
})

test('serialises errors without leaking secrets', () => {
	const result = sanitize(new Error(`failed with ${githubToken}`)) as Record<string, unknown>
	expect(result['name']).toBe('Error')
	expect(String(result['message'])).not.toContain(githubToken)
})

test('filters by level and joins child scopes', () => {
	const memory = new MemorySink()
	const logger = createLogger({ sinks: [memory], level: 'warn' }).child('training').child('worker')
	logger.info('hidden')
	logger.warn('shown', { step: 3 })
	const [entry] = memory.entries()
	expect(memory.entries()).toHaveLength(1)
	expect(entry?.scope).toBe('training.worker')
	expect(entry?.fields).toEqual({ step: 3 })
})

test('a failing sink never throws into the caller', () => {
	const broken: LogSink = { write: (_entry: LogEntry) => { throw new Error('disk full') } }
	const memory = new MemorySink()
	const logger = createLogger({ sinks: [broken, memory] })
	expect(() => logger.error('still logged')).not.toThrow()
	expect(logger.droppedEntries).toBe(1)
	expect(memory.entries()).toHaveLength(1)
})

test('the file sink writes JSON lines and rotates within its limits', async () => {
	const directory = await mkdtemp(join(tmpdir(), 'scantling-log-'))
	roots.push(directory)
	const logger = createLogger({ sinks: [new FileSink({ directory, maxBytes: 300, maxFiles: 2 })] })
	for (let index = 0; index < 20; index += 1) logger.info('entry', { index })
	const current = join(directory, 'scantling.log')
	expect(existsSync(current)).toBe(true)
	expect(existsSync(`${current}.1`)).toBe(true)
	expect(existsSync(`${current}.2`)).toBe(true)
	expect(existsSync(`${current}.3`)).toBe(false)
	const first = readFileSync(current, 'utf8').trim().split('\n')[0] ?? ''
	expect(JSON.parse(first).message).toBe('entry')
})

test('timing helpers report plausible values', async () => {
	const result = await timed(() => Bun.sleep(15))
	expect(result.milliseconds).toBeGreaterThanOrEqual(10)
	expect(throughput(500, 250)).toBe(2000)
	expect(throughput(1, 0)).toBe(0)
})

test('masks keys that end in token but not counters that mention tokens', () => {
	const result = sanitize({ accessToken: 'abc', token: 'abc', tokensPerSecond: 5, trainTokens: 9 }) as Record<string, unknown>
	expect(result['accessToken']).toBe(REDACTED)
	expect(result['token']).toBe(REDACTED)
	expect(result['tokensPerSecond']).toBe(5)
	expect(result['trainTokens']).toBe(9)
})
