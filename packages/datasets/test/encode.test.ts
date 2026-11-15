import { mkdtemp } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { expect, test } from 'bun:test'
import { trainBpe } from '@scantling/tokenizer'
import { encodeAll } from '../src/encode.ts'

test('workers, the cache and direct encoding all give the same ids', async () => {
	const texts = Array.from({ length: 40 }, (_, index) => `export function item${index}(a: number) { return a + ${index} }\n`.repeat(2000))
	const { tokenizer } = trainBpe(texts.slice(0, 3), { vocabSize: 320 })
	const direct = texts.map((text) => Int32Array.from(tokenizer.encode(text)))
	const cache = await mkdtemp(join(tmpdir(), 'encode-'))
	const first = await encodeAll(texts, tokenizer, { workers: 2, cacheDirectory: cache })
	expect(first.cached).toBe(0)
	expect(first.ids).toEqual(direct)
	const second = await encodeAll(texts, tokenizer, { workers: 2, cacheDirectory: cache })
	expect(second.cached).toBe(texts.length)
	expect(second.ids).toEqual(direct)
	expect((await encodeAll(texts, tokenizer, { workers: 0 })).ids).toEqual(direct)
})
