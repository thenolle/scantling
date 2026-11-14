import { Tokenizer, collectTextFiles, trainBpe } from '@scantling/tokenizer'
import { buildDataset } from './pipeline.ts'
import type { DatasetManifest } from './pipeline.ts'
import { isSensitivePath, scanDocument } from './scan.ts'

const [command, ...rest] = Bun.argv.slice(2)
const option = (name: string): string | undefined => (rest.includes(name) ? rest[rest.indexOf(name) + 1] : undefined)
const inputs = rest.flatMap((value, index) => (rest[index - 1] === '--input' ? [value] : []))

function summarize(manifest: DatasetManifest): void {
	const { totals } = manifest
	console.log(`Included     ${totals.documents} documents, ${totals.bytes} bytes`)
	console.log(`Tokens       ${totals.trainTokens} train, ${totals.validationTokens} validation (${(totals.bytes / Math.max(1, totals.trainTokens + totals.validationTokens)).toFixed(2)} bytes per token)`)
	const reasons = new Map<string, number>()
	for (const file of manifest.files) { if (file.status === 'excluded') reasons.set((file.reason ?? 'unknown').split(':')[0]!, (reasons.get((file.reason ?? 'unknown').split(':')[0]!) ?? 0) + 1) }
	for (const [reason, count] of reasons) console.log(`Excluded     ${count} x ${reason}`)
}

if (command === 'build') {
	const tokenizerPath = option('--tokenizer') ?? 'data/tokenizer.json'
	if (inputs.length === 0) throw new Error('Usage: build --input <path>... [--tokenizer file] [--out dir] [--validation 5] [--approve]')
	const approved = rest.includes('--approve')
	const report = await buildDataset({ inputs, tokenizer: Tokenizer.fromJson(await Bun.file(tokenizerPath).text()), outputDirectory: option('--out') ?? 'data/datasets/default', validationPercent: Number(option('--validation') ?? 5), approved })
	summarize(report.manifest)
	if (rest.includes('--verbose')) { for (const file of report.manifest.files) if (file.status === 'excluded') console.log(`  ${file.reason}  ${file.path}`) }
	if (report.manifestPath === undefined) console.log('\nDry run: nothing was written. Review the exclusions, then repeat the command with --approve to write the dataset.')
	else console.log(`\nWritten to ${report.manifestPath}`)
} else if (command === 'inspect') {
	const manifest = JSON.parse(await Bun.file(`${option('--dir') ?? 'data/datasets/default'}/manifest.json`).text()) as DatasetManifest
	console.log(`Created      ${manifest.createdAt}`)
	summarize(manifest)
} else if (command === 'tokenizer') {
	const texts: string[] = []
	const seen = new Set<string>()
	let skipped = 0
	for await (const file of collectTextFiles(inputs)) {
		const scan = isSensitivePath(file.path) ? undefined : scanDocument(file.text)
		if (scan === undefined || scan.secrets.length > 0) {
			skipped += 1
			continue
		}
		const key = String(Bun.hash(scan.text))
		if (seen.has(key)) continue
		seen.add(key)
		texts.push(scan.text)
	}
	const result = trainBpe(texts, { vocabSize: Number(option('--vocab') ?? 4096) })
	const output = option('--out') ?? 'data/tokenizer.json'
	await Bun.write(output, result.tokenizer.toJson())
	console.log(`Trained ${result.merges} merges from ${texts.length} documents (${skipped} excluded as sensitive or secret-bearing); saved ${output}`)
} else {
	console.error('Commands: build --input <path>... [--tokenizer file] [--out dir] [--validation 5] [--approve] | inspect [--dir dir] | tokenizer --input <path>... [--vocab 4096] [--out file]')
	process.exit(1)
}
