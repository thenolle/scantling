import { appendFileSync, writeFileSync } from 'node:fs'
import { mkdtemp } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { expect, test } from 'bun:test'
import { Hub } from '../src/hub.ts'
import { createMonitorApp } from '../src/server.ts'
import { parseGpuCsv } from '../src/system.ts'
import { LogTail } from '../src/tail.ts'
import { RunTracker } from '../src/tracker.ts'
import type { LogRecord } from '../src/types.ts'

const entry = (message: string, fields: Record<string, unknown> = {}, second = 0): LogRecord => ({ time: `2026-11-15T12:40:${String(second).padStart(2, '0')}.000Z`, level: 'info', scope: '', message, fields })
const line = (message: string): string => `${JSON.stringify(entry(message))}\n`

test('a run is followed from plan to finish and the next plan archives it', () => {
	const tracker = new RunTracker()
	expect(tracker.ingest(entry('Training plan', { steps: 100 })).map((delta) => delta.type)).toEqual(['log', 'plan'])
	tracker.ingest(entry('step 11/100', { loss: 7.5, gradNorm: 1.2, lr: 0.001, tokensPerSecond: 20000, etaMinutes: 3, skipped: false, gpuMs: 270, optimizerMs: 80, transferMs: 36 }, 1))
	tracker.ingest(entry('validation @10', { loss: 6.1, perplexity: 450, bitsPerByte: 2.5 }, 2))
	tracker.ingest(entry('Checkpoint', { id: 'step-00000010', evalLoss: 6.1, promoted: true }, 3))
	const running = tracker.snapshot()
	expect(running.status).toBe('running')
	expect(running.steps[0]).toMatchObject({ step: 11, total: 100, loss: 7.5, gpuMs: 270 })
	expect(running.evals[0]?.loss).toBe(6.1)
	expect(running.checkpoints[0]?.promoted).toBe(true)
	tracker.ingest(entry('Finished', { minutes: 5 }, 4))
	expect(tracker.snapshot().status).toBe('finished')
	tracker.ingest(entry('Training plan', { steps: 50 }, 5))
	const next = tracker.snapshot()
	expect(next.steps).toEqual([])
	expect(next.past).toEqual([{ startedAt: '2026-11-15T12:40:00.000Z', status: 'finished', lastStep: 11, total: 100 }])
})

test('a step without a plan starts an implicit run and dataset progress is tracked', () => {
	const tracker = new RunTracker()
	tracker.ingest(entry('step 1/10', { loss: 8 }))
	expect(tracker.snapshot().status).toBe('running')
	tracker.ingest(entry('Dataset build started'))
	tracker.ingest(entry('Dataset progress', { phase: 'encode', done: 5, total: 10, cached: 2 }, 1))
	expect(tracker.snapshot().dataset).toMatchObject({ phase: 'encode', done: 5, total: 10, cached: 2, finished: false })
})

test('new lines are read once, partial lines wait and a rotated file restarts', async () => {
	const path = join(await mkdtemp(join(tmpdir(), 'tail-')), 'a.log')
	const tail = new LogTail(path)
	expect(await tail.read()).toEqual([])
	writeFileSync(path, line('one') + line('two').slice(0, 20))
	expect((await tail.read()).map((record) => record.message)).toEqual(['one'])
	appendFileSync(path, line('two').slice(20) + line('three'))
	expect((await tail.read()).map((record) => record.message)).toEqual(['two', 'three'])
	expect(await tail.read()).toEqual([])
	writeFileSync(path, line('four'))
	expect((await tail.read()).map((record) => record.message)).toEqual(['four'])
})

test('nvidia-smi rows are parsed and unavailable fields are left out', () => {
	expect(parseGpuCsv('NVIDIA GeForce RTX 4060 Laptop GPU, 97, 1241, 8188, 63, [N/A], 2370\n')).toEqual({ name: 'NVIDIA GeForce RTX 4060 Laptop GPU', utilization: 97, memoryUsedMiB: 1241, memoryTotalMiB: 8188, temperatureC: 63, clockMHz: 2370 })
	expect(parseGpuCsv('')).toBeUndefined()
})

test('the server answers only on localhost and streams a snapshot, then live events', async () => {
	const tracker = new RunTracker()
	const hub = new Hub()
	const app = createMonitorApp({ tracker, hub, system: () => [] })
	expect((await app.fetch(new Request('http://evil.example:3400/'))).status).toBe(403)
	const page = await app.fetch(new Request('http://127.0.0.1:3400/'))
	expect(await page.text()).toContain('id="lossChart"')
	expect(((await (await app.fetch(new Request('http://localhost:3400/api/snapshot'))).json()) as { status: string }).status).toBe('idle')
	const response = await app.fetch(new Request('http://127.0.0.1:3400/events'))
	const reader = response.body!.getReader()
	const decoder = new TextDecoder()
	expect(decoder.decode((await reader.read()).value)).toContain('event: snapshot')
	hub.broadcast('tick', { now: 1, lastEntryAt: 2 })
	expect(decoder.decode((await reader.read()).value)).toContain('event: tick')
	await reader.cancel()
	expect(hub.size).toBe(0)
})
