import { dirname } from 'node:path'
import { mkdir } from 'node:fs/promises'
import { collectTextFiles } from './files.ts'
import { Tokenizer } from './tokenizer.ts'
import { trainBpe } from './train.ts'

const [command, ...rest] = Bun.argv.slice(2)

function option(name: string): string | undefined {
	const index = rest.indexOf(name)
	return index === -1 ? undefined : rest[index + 1]
}

function inputs(): string[] {
	return rest.flatMap((value, index) => (rest[index - 1] === '--input' ? [value] : []))
}

async function load(paths: readonly string[]): Promise<string[]> {
	const texts: string[] = []
	for await (const file of collectTextFiles(paths)) texts.push(file.text)
	return texts
}

function describe(tokenizer: Tokenizer, texts: readonly string[]): void {
	let bytes = 0
	let tokens = 0
	const started = performance.now()
	const encoder = new TextEncoder()
	for (const text of texts) {
		const ids = tokenizer.encode(text)
		tokens += ids.length
		bytes += encoder.encode(text).length
		if (tokenizer.decode(ids) !== text) throw new Error('Round trip failed: decoded text differs from the input')
	}
	const seconds = (performance.now() - started) / 1000
	console.log(`Round trip   exact for all ${texts.length} files`)
	console.log(`Compression  ${(bytes / Math.max(1, tokens)).toFixed(2)} bytes per token (${tokens} tokens for ${bytes} bytes)`)
	console.log(`Speed        ${Math.round(bytes / 1024 / seconds)} KiB/s encode and decode together`)
}

if (command === 'train') {
	const paths = inputs()
	const vocab = Number(option('--vocab') ?? 4096)
	const output = option('--out') ?? 'data/tokenizer.json'
	if (paths.length === 0) throw new Error('Usage: train --input <path> [--input <path>...] [--vocab 4096] [--out data/tokenizer.json]')
	const texts = await load(paths)
	const started = performance.now()
	const result = trainBpe(texts, { vocabSize: vocab })
	console.log(`Trained      ${result.merges} merges, vocabulary ${result.tokenizer.vocabSize}, in ${((performance.now() - started) / 1000).toFixed(1)} s`)
	console.log(`Corpus       ${texts.length} files, ${result.corpusBytes} bytes, ${result.distinctPieces} distinct pieces`)
	describe(result.tokenizer, texts)
	await mkdir(dirname(output), { recursive: true })
	await Bun.write(output, result.tokenizer.toJson())
	console.log(`Saved        ${output}`)
} else if (command === 'inspect') {
	const tokenizer = Tokenizer.fromJson(await Bun.file(option('--tokenizer') ?? 'data/tokenizer.json').text())
	console.log(`Vocabulary   ${tokenizer.vocabSize} (${tokenizer.specialTokens.length} special, 256 bytes, ${tokenizer.merges.length} merges)`)
	describe(tokenizer, await load(inputs()))
} else {
	console.error('Commands: train --input <path>... [--vocab N] [--out file] | inspect [--tokenizer file] --input <path>...')
	process.exit(1)
}
