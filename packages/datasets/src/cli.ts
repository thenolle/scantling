import { Tokenizer, collectTextFiles, trainBpe, DEFAULT_SPECIAL_TOKENS } from '@scantling/tokenizer'
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { buildDataset } from './pipeline.ts'
import type { DatasetManifest } from './pipeline.ts'
import { isGeneratedPath, isSensitivePath, scanDocument, isCodeLikePath, isMinifiedText, isDataList } from './scan.ts'
import { SEPARATOR } from './wikipedia/extract.ts'
import { createRuntimeLogger } from '@scantling/telemetry'
import { keepAwake } from '@scantling/hardware'
import type { BuildProgress } from './pipeline'

const [command, ...rest] = Bun.argv.slice(2)
const option = (name: string): string | undefined => (rest.includes(name) ? rest[rest.indexOf(name) + 1] : undefined)
const inputs = rest.flatMap((value, index) => (rest[index - 1] === '--input' ? [value] : []))
const prose = rest.flatMap((value, index) => (rest[index - 1] === '--prose' ? [value] : []))
const weights = Object.fromEntries(rest.flatMap((value, index) => (rest[index - 1] === '--weight' ? [[value.slice(0, value.lastIndexOf('=')), Number(value.slice(value.lastIndexOf('=') + 1))] as const] : [])))
for (const count of Object.values(weights)) if (!Number.isInteger(count) || count < 1) throw new Error('--weight needs a path and a whole number of at least 1, as in F:/web=3')

function summarize(manifest: DatasetManifest): void {
	const { totals } = manifest
	console.log(`Included     ${totals.documents} documents, ${totals.bytes} bytes`)
	const distinct = manifest.files.reduce((total, file) => total + (file.status === 'included' ? (file.tokens ?? 0) : 0), 0)
	console.log(`Tokens       ${totals.trainTokens} train (repeats counted), ${totals.validationTokens} validation; ${distinct} distinct tokens, ${(totals.bytes / Math.max(1, distinct)).toFixed(2)} bytes per token`)
	const reasons = new Map<string, number>()
	for (const file of manifest.files) { if (file.status === 'excluded') reasons.set((file.reason ?? 'unknown').split(':')[0]!, (reasons.get((file.reason ?? 'unknown').split(':')[0]!) ?? 0) + 1) }
	const byExtension = new Map<string, number>()
	for (const file of manifest.files) {
		if (file.status !== 'included') continue
		const name = file.path.split(/[\\/]/).pop() ?? ''
		const extension = name.lastIndexOf('.') > 0 ? name.slice(name.lastIndexOf('.')) : '(none)'
		byExtension.set(extension, (byExtension.get(extension) ?? 0) + (file.tokens ?? 0))
	}
	const top = [...byExtension].sort((a, b) => b[1] - a[1]).slice(0, 10).map(([extension, tokens]) => `${extension} ${(tokens / 1e6).toFixed(2)}M`)
	console.log(`Top types    ${top.join('  ')}`)
	for (const [reason, count] of reasons) console.log(`Excluded     ${count} x ${reason}`)
}

if (command === 'build') {
	const tokenizerPath = option('--tokenizer') ?? 'data/tokenizer.json'
	if (inputs.length === 0 && prose.length === 0) throw new Error('Usage: build --input <path>... [--tokenizer file] [--out dir] [--validation 5] [--approve] [--weight <path>=<n>] [--workers N] [--no-cache]')
	const approved = rest.includes('--approve')
	const workers = option('--workers') === undefined ? undefined : Number(option('--workers'))
	const { logger } = createRuntimeLogger({ level: 'info', directory: 'data/logs', toFile: true, maxFileBytes: 10 * 1024 * 1024, maxFiles: 5, console: false })
	const awake = keepAwake({ reason: 'Scantling is building a dataset' })
	let lastShown = 0
	const progress = (event: BuildProgress): void => {
		const now = performance.now()
		if (now - lastShown < 1000 && event.done !== event.total) return
		lastShown = now
		process.stderr.write(`\r${event.phase.padEnd(7)} ${event.done}${event.total > 0 ? `/${event.total}` : ''}${event.cached === undefined ? '' : `  (${event.cached} from cache)`}        `)
		logger.info('Dataset progress', { phase: event.phase, done: event.done, total: event.total, ...(event.cached === undefined ? {} : { cached: event.cached }) })
	}
	logger.info('Dataset build started', { inputs: inputs.length, approved })
	try {
		const report = await buildDataset({
			inputs, tokenizer: Tokenizer.fromJson(await Bun.file(tokenizerPath).text()), outputDirectory: option('--out') ?? 'data/datasets/default',
			validationPercent: Number(option('--validation') ?? 5), approved, weights, onProgress: progress,
			...(prose.length === 0 ? {} : { prose }),
			...(workers === undefined ? {} : { workers }),
			...(rest.includes('--no-cache') ? {} : { cacheDirectory: option('--cache-dir') ?? 'data/cache/tokens' })
		})
		process.stderr.write('\n')
		summarize(report.manifest)
		const { timings } = report
		console.log(`Timing       scan ${(timings.scanMs / 1000).toFixed(1)} s, encode ${(timings.encodeMs / 1000).toFixed(1)} s (${timings.cachedDocuments} of ${report.manifest.totals.documents} documents from cache), write ${(timings.writeMs / 1000).toFixed(1)} s`)
		logger.info('Dataset finished', { documents: report.manifest.totals.documents, trainTokens: report.manifest.totals.trainTokens, scanSeconds: Math.round(timings.scanMs / 1000), encodeSeconds: Math.round(timings.encodeMs / 1000), writeSeconds: Math.round(timings.writeMs / 1000), cached: timings.cachedDocuments })
		if (rest.includes('--verbose')) { for (const file of report.manifest.files) if (file.status === 'excluded') console.log(`  ${file.reason}  ${file.path}`) }
		if (report.manifestPath === undefined) console.log('\nDry run: nothing was written. Review the exclusions, then repeat the command with --approve to write the dataset.')
		else console.log(`\nWritten to ${report.manifestPath}`)
	} finally {
		awake.release()
	}
} else if (command === 'inspect') {
	const manifest = JSON.parse(await Bun.file(`${option('--dir') ?? 'data/datasets/default'}/manifest.json`).text()) as DatasetManifest
	console.log(`Created      ${manifest.createdAt}`)
	summarize(manifest)
} else if (command === 'tokenizer') {
	const texts: string[] = []
	const seen = new Set<string>()
	let skipped = 0
	for await (const file of collectTextFiles(inputs)) {
		const unwanted = isSensitivePath(file.path) || isGeneratedPath(file.path) || !isCodeLikePath(file.path, file.text.length) || isMinifiedText(file.text) || isDataList(file.text)
		const scan = unwanted ? undefined : scanDocument(file.text)
		if (scan === undefined || scan.secrets.length > 0) {
			skipped += 1
			continue
		}
		const key = String(Bun.hash(scan.text))
		if (seen.has(key)) continue
		seen.add(key)
		texts.push(scan.text)
	}
	const proseDirectories = rest.flatMap((value, index) => (rest[index - 1] === '--prose' ? [value] : []))
	if (proseDirectories.length > 0) {
		const shardFiles = proseDirectories.flatMap((directory) => readdirSync(directory).filter((name) => name.endsWith('.prose')).sort().map((name) => join(directory, name)))
		const total = shardFiles.reduce((sum, path) => sum + statSync(path).size, 0)
		const keep = Math.min(1, (Number(option('--prose-mb') ?? 100) * 1_048_576) / Math.max(1, total))
		for (const path of shardFiles) {
			for (const article of readFileSync(path, 'utf8').split(SEPARATOR)) {
				if (article !== '' && Number(BigInt(Bun.hash(article)) % 10_000n) < keep * 10_000) texts.push(article)
			}
		}
		console.log(`Prose        sampled ${(keep * 100).toFixed(1)}% of ${(total / 1_048_576).toFixed(0)} MiB`)
	}
	const extraSpecials = (option('--specials') ?? '').split(',').filter((token) => token !== '')
	const result = trainBpe(texts, { vocabSize: Number(option('--vocab') ?? 4096), specialTokens: [...DEFAULT_SPECIAL_TOKENS, ...extraSpecials] })
	const output = option('--out') ?? 'data/tokenizer.json'
	await Bun.write(output, result.tokenizer.toJson())
	console.log(`Trained ${result.merges} merges from ${texts.length} documents (${skipped} excluded as sensitive or secret-bearing); saved ${output}`)
} else {
	console.error('Commands: build --input <path>... [--tokenizer file] [--out dir] [--validation 5] [--approve] | inspect [--dir dir] | tokenizer --input <path>... [--vocab 4096] [--out file]')
	process.exit(1)
}
