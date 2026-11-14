import { expect, test } from 'bun:test'
import { mkdtemp, readdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { isPermissive, stageLicensed } from '../src/licensed.ts'

test('only permissive licence expressions are accepted', () => {
	for (const accepted of ['MIT', 'ISC', 'Apache-2.0', '(MIT OR GPL-3.0)', 'MIT AND BSD-3-Clause', 'bsd-2-clause']) expect(isPermissive(accepted)).toBe(true)
	for (const refused of ['GPL-3.0', 'AGPL-3.0-only', 'MIT AND GPL-2.0', 'UNLICENSED', 'SEE LICENSE IN LICENSE', '', undefined]) expect(isPermissive(refused)).toBe(false)
})

test('staging copies source from permissive packages once and records them', async () => {
	const root = await mkdtemp(join(tmpdir(), 'stage-'))
	const write = (path: string, text: string) => Bun.write(join(root, path), text)
	const good = JSON.stringify({ name: 'good', version: '1.0.0', license: 'MIT' })
	await write('a/node_modules/good/package.json', good)
	await write('a/node_modules/good/index.js', 'export const x = 1\n')
	await write('a/node_modules/good/LICENSE', 'MIT text')
	await write('a/node_modules/good/index.min.js', 'x')
	await write('a/node_modules/@scope/copyleft/package.json', JSON.stringify({ name: '@scope/copyleft', version: '2.0.0', license: 'GPL-3.0' }))
	await write('a/node_modules/@scope/copyleft/index.js', 'copyleft source')
	await write('b/node_modules/good/package.json', good)
	await write('b/node_modules/good/index.js', 'export const x = 1\n')
	const report = await stageLicensed({ from: root, to: join(root, 'out'), maxBytes: 1_000_000 })
	expect(report.staged.map((entry) => `${entry.package}@${entry.version}`)).toEqual(['good@1.0.0'])
	expect(report.rejected.get('GPL-3.0')).toBe(1)
	expect((await readdir(join(root, 'out', 'good@1.0.0'))).sort()).toEqual(['LICENSE', 'index.js'])
	expect(await Bun.file(join(root, 'out', 'licenses.json')).exists()).toBe(true)
})
