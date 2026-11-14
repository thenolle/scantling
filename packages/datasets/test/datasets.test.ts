import { afterEach, expect, test } from 'bun:test'
import { mkdir, mkdtemp, readdir, rm, stat, truncate } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { Random } from '@scantling/tensor'
import { DEFAULT_SPECIAL_TOKENS, Tokenizer } from '@scantling/tokenizer'
import { FileBatchSource, TokenFile, TokenWriter, buildDataset, fileEvaluationBatches, isSensitivePath, lineHashes, overlapRatio, scanDocument, splitOf } from '../src/index.ts'

const roots: string[] = []
afterEach(async () => {
	await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })))
})

async function workspace(files: Record<string, string> = {}): Promise<string> {
	const root = await mkdtemp(join(tmpdir(), 'scantling-data-'))
	roots.push(root)
	for (const [name, content] of Object.entries(files)) {
		await mkdir(dirname(join(root, name)), { recursive: true })
		await Bun.write(join(root, name), content)
	}
	return root
}

const githubToken = `ghp_${'b'.repeat(36)}`
const tokenizer = new Tokenizer(DEFAULT_SPECIAL_TOKENS)

test('sensitive paths and secrets are recognised, emails are redacted', () => {
	expect(isSensitivePath('project/.env')).toBe(true)
	expect(isSensitivePath('project/.env.local')).toBe(true)
	expect(isSensitivePath('keys/server.pem')).toBe(true)
	expect(isSensitivePath('src/index.ts')).toBe(false)
	const scan = scanDocument(`contact: jane.doe@example.org\nconst token = '${githubToken}'`)
	expect(scan.secrets).toContain('github-token')
	expect(scan.emails).toBe(1)
	expect(scan.text).not.toContain('jane.doe')
})

test('a token file round-trips beyond one buffer and rejects bad ids', async () => {
	const root = await workspace()
	const path = join(root, 'a.tokens')
	const writer = new TokenWriter(path, 300)
	const ids = Array.from({ length: 70000 }, (_, index) => index % 300)
	writer.append(ids)
	expect(writer.close()).toBe(70000)
	const file = TokenFile.open(path)
	expect(file.count).toBe(70000)
	expect(file.vocabSize).toBe(300)
	expect(Array.from(file.read(33000, 5))).toEqual(ids.slice(33000, 33005))
	expect(file.readAll()).toEqual(Int32Array.from(ids))
	expect(() => file.read(69999, 5)).toThrow(RangeError)
	file.close()
	expect(() => new TokenWriter(join(root, 'b.tokens'), 100).append([100])).toThrow(RangeError)
	expect(() => new TokenWriter(join(root, 'c.tokens'), 70000)).toThrow(RangeError)
})

test('a truncated token file is detected', async () => {
	const root = await workspace()
	const path = join(root, 'a.tokens')
	const writer = new TokenWriter(path, 10)
	writer.append([1, 2, 3, 4])
	writer.close()
	await truncate(path, (await stat(path)).size - 1)
	expect(() => TokenFile.open(path)).toThrow()
})

test('file batches align targets one token ahead and evaluation batches are fixed', async () => {
	const root = await workspace()
	const path = join(root, 'a.tokens')
	const writer = new TokenWriter(path, 1000)
	writer.append(Array.from({ length: 101 }, (_, index) => index))
	writer.close()
	const file = TokenFile.open(path)
	const batch = new FileBatchSource(file, 8, 4, new Random(1)).next()
	expect(batch.batch).toBe(4)
	expect(Array.from(batch.targets).every((value, index) => value === (batch.inputs[index] as number) + 1)).toBe(true)
	const evaluation = fileEvaluationBatches(file, 10, 4, 10)
	expect(evaluation.map((item) => item.batch)).toEqual([4, 4, 2])
	expect(Array.from(Array.from(evaluation[0]!.inputs).slice(0, 10))).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	expect(evaluation[0]!.targets[9]).toBe(10)
	expect(() => new FileBatchSource(file, 200, 1, new Random(1))).toThrow(RangeError)
	file.close()
})

test('overlap and split helpers', () => {
	const train = new Set(lineHashes('this is a reasonably long line of source code\nanother sufficiently long line of source'))
	expect(overlapRatio(lineHashes('this is a reasonably long line of source code\nsomething else entirely that is long enough'), train)).toBe(0.5)
	expect(overlapRatio(lineHashes('short'), train)).toBe(0)
	expect(splitOf('a/b.ts', 100)).toBe('validation')
	expect(splitOf('a/b.ts', 0)).toBe('train')
	expect(splitOf('a/b.ts', 50)).toBe(splitOf('a\\b.ts', 50))
})

const sources = {
	'src/a.ts': 'export const alpha = 1\nexport function one() {\n\treturn alpha + 1\n}\n',
	'src/copy.ts': 'export const alpha = 1\nexport function one() {\n\treturn alpha + 1\n}\n',
	'src/b.ts': 'export const beta = 2\n// contact: someone@example.org\n',
	'src/leak.ts': `export const key = '${githubToken}'\n`,
	'.env': 'SECRET=value\n',
	'bin.dat': 'a\u0000b'
}

test('a dry run reports exclusions and writes nothing', async () => {
	const root = await workspace(sources)
	const output = join(root, 'out')
	const report = await buildDataset({ inputs: [root], tokenizer, outputDirectory: output, validationPercent: 0, approved: false })
	const excluded = Object.fromEntries(report.manifest.files.filter((file) => file.status === 'excluded').map((file) => [file.path.replace(root, '').replace(/\\/g, '/'), file.reason]))
	expect(excluded['/.env']).toBe('sensitive-path')
	expect(excluded['/src/leak.ts']).toMatch(/^secret:github-token/)
	expect(Object.values(excluded)).toContain('duplicate')
	expect(report.manifest.totals.documents).toBe(2)
	expect(report.manifest.totals.trainTokens).toBeGreaterThan(0)
	expect(report.manifestPath).toBeUndefined()
	expect(await readdir(root)).not.toContain('out')
})

test('an approved build writes token files and a manifest that agree', async () => {
	const root = await workspace(sources)
	const output = join(root, 'out')
	const report = await buildDataset({ inputs: [root], tokenizer, outputDirectory: output, validationPercent: 0, approved: true })
	const train = TokenFile.open(join(output, 'train.tokens'))
	expect(train.count).toBe(report.manifest.totals.trainTokens)
	const tokens = train.readAll()
	expect(tokens[0]).toBe(tokenizer.idOf('<|file|>'))
	expect(tokens.includes(tokenizer.idOf('<|endfile|>'))).toBe(true)
	expect(tokenizer.decode(tokens, { skipSpecial: true })).not.toContain('someone@example.org')
	expect(tokenizer.decode(tokens, { skipSpecial: true })).not.toContain(githubToken)
	train.close()
	expect(TokenFile.open(join(output, 'validation.tokens')).count).toBe(0)
	const saved = JSON.parse(await Bun.file(join(output, 'manifest.json')).text()) as { tokenizer: { vocabSize: number } }
	expect(saved.tokenizer.vocabSize).toBe(tokenizer.vocabSize)
})

test('the same inputs always produce identical token files', async () => {
	const root = await workspace(sources)
	await buildDataset({ inputs: [root], tokenizer, outputDirectory: join(root, 'one'), validationPercent: 30, approved: true })
	await buildDataset({ inputs: [root], tokenizer, outputDirectory: join(root, 'two'), validationPercent: 30, approved: true })
	for (const name of ['train.tokens', 'validation.tokens']) expect(new Uint8Array(await Bun.file(join(root, 'one', name)).arrayBuffer())).toEqual(new Uint8Array(await Bun.file(join(root, 'two', name)).arrayBuffer()))
})
