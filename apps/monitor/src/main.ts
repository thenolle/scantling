import { serve } from '@gambrel/bun'
import { Hub } from './hub.ts'
import { createMonitorApp } from './server.ts'
import { CpuMeter, sampleSystem } from './system.ts'
import type { SystemSample } from './system.ts'
import { LogTail } from './tail.ts'
import { RunTracker } from './tracker.ts'

const args = Bun.argv.slice(2)
const option = (name: string): string | undefined => (args.includes(name) ? args[args.indexOf(name) + 1] : undefined)
const logPath = option('--log') ?? 'data/logs/scantling.log'
const port = Number(option('--port') ?? 3400)
const interval = Number(option('--system-interval') ?? 3000)

const tracker = new RunTracker()
const hub = new Hub()
const tail = new LogTail(logPath)
const history: SystemSample[] = []
const meter = new CpuMeter()

const pump = async (): Promise<void> => {
	for (const record of await tail.read()) for (const delta of tracker.ingest(record)) hub.broadcast('delta', delta)
}
await pump()
setInterval(() => void pump().catch(() => undefined), 500)
setInterval(() => hub.broadcast('tick', { now: Date.now(), lastEntryAt: tracker.lastEntryAt }), 4000)
setInterval(() => {
	if (hub.size === 0) return
	void sampleSystem(meter).then((sample) => {
		history.push(sample)
		if (history.length > 150) history.shift()
		hub.broadcast('system', sample)
	})
}, interval)

const server = serve(createMonitorApp({ tracker, hub, system: () => history }), { port, hostname: '127.0.0.1' })
console.log(`Watching ${logPath}. Open ${server.url.href}`)
