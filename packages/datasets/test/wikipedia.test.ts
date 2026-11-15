import { mkdtemp, readFile, readdir, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { expect, test } from 'bun:test'
import { bunzip2 } from '../src/wikipedia/bunzip2.ts'
import { SEPARATOR, extractRange } from '../src/wikipedia/extract.ts'
import { cleanWikitext } from '../src/wikipedia/wikitext.ts'

const bytes = (base64: string): Uint8Array => new Uint8Array(Buffer.from(base64, 'base64'))
const text = (data: Uint8Array): string => new TextDecoder().decode(data)

const QUICK = 'QlpoOTFBWSZTWfiS1VUAANuTgEABBAA////wMADYBQADQAAFAANAAAFKlNTTRowmhtTbUmBOQmRPoTInwJqJgTUToJ7ifYmBOwmBOwTYTcJ3E8CdRNhPQT8EwJuEyJ3E0E5CaieBNBOon8JsJoJkTInQVeRPIn+LuSKcKEh8SWqqgA=='
const RUNS = 'QlpoOTFBWSZTWeeelH8AAALBgeAAPgAAcACAAAggADEDQNAZIYMpNMqgC8OiYsxZzU0BXxdyRThQkOeelH8='
const DUMP = 'QlpoOTFBWSZTWc66VpgAASpfgAAQWIH4Bw4gVEo/79/gMAF5IASQpBoAyGmhmkAADAA0aGjCMmg0aaAwKVIpiaeVNGI9QMjTQ2keptT2GfgTwYJi6UnD0egmBNHX7cUYo3CeYmBPETmJ/cxPF9vl2aGmkEBjEIp2pKZOhIqTiugrnnqvpkwjMdWNCpWVOoUbmJlArC4toxMxK2VACJcR9dbndHou4TMT34E7529wmH7nEVb+NixYvL+Y/wmgmS1XITVZBPMTlx6hPXaJ910drVb60E15TOtgnHAnUJskxijHWtBNl5XqJuE3LsE3pNltvgJsyzsW0TN0CcFkuHNZibbdboTFlbazSaiZCdK1zSZLmJ/xdyRThQkM66VpgEJaaDkxQVkmU1kMWDYsAAGw24AAEECB5xVEAD/v3+AwAVgAYwmJoMEYhkYTDGExNBgjEMjCYJqoiaBMoNqaB6mEZPB2pdrFLFili2B6PShihwdevWlgrJW+h/KGKH3ofFDNDycJw4cOjWcQdho+cqu/j2xrNXLjy1eLCuj2fR0ZaNGjM67seWM3MHnPacaH61qHjOVDvPOkXm4sMP9Q9aH7nOf2hymah4UN9DSh9Z0cZtOFDb1mk3UN+KHvQ5xYwVjBWOc1od3s96G+hvnvQ4g3Nn4oZZbUNmtDtMz4maG5s2qGGW6aA50M0N8yDE+KH/F3JFOFCQDFg2LAQlpoOTFBWSZTWQeC9UEAAbHbgAAQQIHA5UQAP+/f4DABWABhgmBMBDRk0wMMEwJgIaMmmAmqkRo0FHqeo9QNkam1OFdaXWxSxYpYtAeHihihvc+XPBWCsit1D2oYodqHqhlQ/DfN+/fzbJqDrVZukqunVnM2uuuxowrj/MXw6OTJmzZspz0xjGOIO8+JvoZ0NcUOk1oa/2d6Rd3Bhh7/6hwofucp5ocplUPeh59uFD50odp1cprONDl5mc20POKGtDbFjBWMFY+k4UPDR80N1DdPtQ1BtaO1DJk0oaNlD8zKeplQ2tGlQwybZmDjQyobpxzBlPVD/i7kinChIA8F6oI='

test('bzip2 streams decode, including long runs and several streams in a row', () => {
	expect(text(bunzip2(bytes(QUICK)))).toBe('The quick brown fox jumps over the lazy dog. '.repeat(40))
	expect(text(bunzip2(bytes(RUNS)))).toBe('a'.repeat(1000) + 'b'.repeat(3) + 'c'.repeat(256) + 'd'.repeat(4) + 'e'.repeat(5) + 'xyz')
})

test('a damaged block is detected and non-bzip2 data is refused', () => {
	const bad = bytes(QUICK)
	bad[60] = bad[60]! ^ 0x10
	expect(() => bunzip2(bad)).toThrow()
	expect(() => bunzip2(new TextEncoder().encode('plain text'))).toThrow('Not bzip2 data')
})

test('wikitext becomes plain prose without templates, references, tables or categories', () => {
	const filler = 'The company grew quickly and opened a second site. '.repeat(14)
	const raw = `{{Infobox company|name=X}}\n'''Example''' is a [[manufacturing]] firm in [[London|the capital]].<ref>{{cite web|url=http://x}}</ref> ${filler}\n[[File:A.jpg|thumb|A [[factory]]]]\n== History ==\n* First ''item''\n{| class="wikitable"\n| 1 || 2\n|}\n== References ==\n* A book\n[[Category:Companies]]`
	const cleaned = cleanWikitext(raw, 'Example')!
	expect(cleaned.startsWith('Example\n\nExample is a manufacturing firm in the capital.')).toBe(true)
	for (const gone of ['{{', 'cite web', 'File:', 'wikitable', 'A book', 'Category', "'''"]) expect(cleaned).not.toContain(gone)
	expect(cleaned).toContain('History')
	expect(cleaned).toContain('- First item')
	expect(cleanWikitext('#REDIRECT [[Example]]', 'Old')).toBeUndefined()
	expect(cleanWikitext('Too short.', 'Stub')).toBeUndefined()
})

test('shards of a multistream dump cover every article exactly once', async () => {
	const directory = await mkdtemp(join(tmpdir(), 'wiki-'))
	const dump = join(directory, 'dump.xml.bz2')
	await writeFile(dump, bytes(DUMP))
	for (let shard = 0; shard < 2; shard += 1) extractRange({ dump, shard, shards: 2, out: join(directory, 'prose'), windowBytes: 300 })
	const titles: string[] = []
	for (const name of await readdir(join(directory, 'prose'))) {
		for (const article of (await readFile(join(directory, 'prose', name), 'utf8')).split(SEPARATOR)) if (article !== '') titles.push(article.split('\n')[0]!)
	}
	expect(titles.sort()).toEqual(['Place 1', 'Place 2', 'Place 4', 'Place 6', 'Place 7', 'Place 8', 'Place 9'])
})
