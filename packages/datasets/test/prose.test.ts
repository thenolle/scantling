import { mkdir, mkdtemp, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { expect, test } from 'bun:test'
import { trainBpe } from '@scantling/tokenizer'
import { buildDataset } from '../src/pipeline.ts'
import { TokenFile } from '../src/store.ts'
import { SEPARATOR } from '../src/wikipedia/extract.ts'

test('prose shards become documents wrapped in begin and end tokens', async () => {
	const root = await mkdtemp(join(tmpdir(), 'prose-'))
	const prose = join(root, 'prose')
	await mkdir(prose)
	const article = (title: string): string => `${title}\n\n${'The river runs past the old town and the mountain behind it. '.repeat(30)}`
	await writeFile(join(prose, 'wiki-00-0001.prose'), ['Alpha', 'Beta', 'Gamma', 'Delta'].map(article).join(SEPARATOR) + SEPARATOR)
	const { tokenizer } = trainBpe([article('Alpha')], { vocabSize: 320 })
	const report = await buildDataset({ inputs: [], prose: [prose], tokenizer, outputDirectory: join(root, 'out'), validationPercent: 50, approved: true })
	const { totals } = report.manifest
	expect(totals.documents).toBe(4)
	const train = TokenFile.open(join(root, 'out', 'train.tokens'))
	const validation = TokenFile.open(join(root, 'out', 'validation.tokens'))
	expect(train.count).toBe(totals.trainTokens)
	expect(validation.count).toBe(totals.validationTokens)
	const file = train.count > 0 ? train : validation
	expect(file.read(0, 1)[0]).toBe(tokenizer.idOf('<|bos|>'))
	expect(file.read(file.count - 1, 1)[0]).toBe(tokenizer.idOf('<|eos|>'))
	train.close()
	validation.close()
})
