import { closeSync, existsSync, mkdirSync, openSync, readdirSync, statSync, writeSync } from 'node:fs'
import { join } from 'node:path'
import { keepAwake } from '@scantling/hardware'
import { createRuntimeLogger } from '@scantling/telemetry'
import { extractRange } from './extract.ts'

const [command, ...rest] = Bun.argv.slice(2)
const option = (name: string): string | undefined => (rest.includes(name) ? rest[rest.indexOf(name) + 1] : undefined)
const wiki = `${option('--lang') ?? 'simple'}wiki`
const date = option('--date') ?? 'latest'
const fileName = `${wiki}-${date}-pages-articles-multistream.xml.bz2`
const dumpsDirectory = option('--dir') ?? 'F:/corpus/wikipedia/dumps'
const proseDirectory = option('--out') ?? 'F:/corpus/wikipedia/prose'
const dumpPath = option('--dump') ?? join(dumpsDirectory, fileName)
const USER_AGENT = 'scantling-dataset/0.1 (local research use; https://github.com/thenolle/scantling)'

async function downloadOnce(url: string, destination: string, report: (done: number, total: number) => void): Promise<void> {
	const head = await fetch(url, { method: 'HEAD', headers: { 'user-agent': USER_AGENT } })
	if (!head.ok) throw new Error(`The server answered ${head.status} for ${url}`)
	const total = Number(head.headers.get('content-length') ?? 0)
	let have = existsSync(destination) ? statSync(destination).size : 0
	if (total > 0 && have === total) return
	if (total > 0 && have > total) throw new Error(`${destination} is larger than the remote file; delete it and run again`)
	const response = await fetch(url, { headers: { 'user-agent': USER_AGENT, ...(have > 0 ? { range: `bytes=${have}-` } : {}) } })
	if (response.body === null || (response.status !== 200 && response.status !== 206)) throw new Error(`The server answered ${response.status}`)
	if (response.status === 200) have = 0
	const fd = openSync(destination, have === 0 ? 'w' : 'a')
	try {
		const reader = response.body.getReader()
		for (; ;) {
			const { done, value } = await reader.read()
			if (done) break
			writeSync(fd, value)
			have += value.length
			report(have, total)
		}
	} finally {
		closeSync(fd)
	}
	if (total > 0 && have !== total) throw new Error('The download ended early')
}

async function download(url: string, destination: string, report: (done: number, total: number) => void): Promise<void> {
	for (let attempt = 1; ; attempt += 1) {
		try {
			await downloadOnce(url, destination, report)
			return
		} catch (error) {
			if (attempt >= 10) throw error
			console.error(`\nAttempt ${attempt} failed (${error instanceof Error ? error.message : String(error)}); resuming in ${attempt * 5} s`)
			await Bun.sleep(attempt * 5000)
		}
	}
}

const sizeOf = (directory: string): number => (existsSync(directory) ? readdirSync(directory).reduce((sum, name) => sum + statSync(join(directory, name)).size, 0) : 0)

if (command === 'download' || (command === 'extract' && option('--shard') === undefined)) {
	const { logger } = createRuntimeLogger({ level: 'info', directory: 'data/logs', toFile: true, maxFileBytes: 10 * 1024 * 1024, maxFiles: 5, console: false })
	const awake = keepAwake({ reason: `Scantling is working on ${fileName}` })
	logger.info('Dataset build started', { job: command, file: fileName })
	try {
		if (command === 'download') {
			mkdirSync(dumpsDirectory, { recursive: true })
			let lastShown = 0
			await download(`https://dumps.wikimedia.org/${wiki}/${date}/${fileName}`, dumpPath, (done, total) => {
				if (performance.now() - lastShown < 1000) return
				lastShown = performance.now()
				process.stderr.write(`\rdownloaded ${(done / 1048576).toFixed(0)} of ${(total / 1048576).toFixed(0)} MiB   `)
				logger.info('Dataset progress', { phase: 'download', done: Math.round(done / 1048576), total: Math.round(total / 1048576) })
			})
			process.stderr.write('\n')
			logger.info('Dataset finished', { file: dumpPath })
		} else {
			const shards = Number(option('--shards') ?? 64)
			const jobs = Number(option('--jobs') ?? 8)
			mkdirSync(proseDirectory, { recursive: true })
			const started = performance.now()
			const timer = setInterval(() => logger.info('Dataset progress', { phase: 'extract', done: Math.round(sizeOf(proseDirectory) / 1048576), total: 0 }), 2000)
			let next = 0
			await Promise.all(Array.from({ length: jobs }, async () => {
				for (; ;) {
					const shard = next++
					if (shard >= shards) return
					const child = Bun.spawn([process.execPath, import.meta.path, 'extract', '--dump', dumpPath, '--out', proseDirectory, '--shard', String(shard), '--shards', String(shards)], { stdout: 'inherit', stderr: 'inherit' })
					await child.exited
				}
			}))
			clearInterval(timer)
			console.log(`Took ${((performance.now() - started) / 1000).toFixed(0)} s`)
		}
	} finally {
		awake.release()
	}
} else if (command === 'extract') {
	const shard = Number(option('--shard'))
	const stats = extractRange({
		dump: dumpPath, shard, shards: Number(option('--shards')), out: proseDirectory
	})
} else {
	console.error('Commands: download [--lang simple|en] [--date latest|YYYYMMDD] [--dir folder] | extract [--lang ..] [--dump file] [--out folder] [--shards 8]')
	process.exit(1)
}
