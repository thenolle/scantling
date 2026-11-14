import { afterEach, expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { buildNav } from '../scripts/docs.ts'

const roots: string[] = []

afterEach(async () => {
	await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })))
})

async function docs(files: Record<string, string>): Promise<string> {
	const root = await mkdtemp(join(tmpdir(), 'scantling-docs-'))
	roots.push(root)
	for (const [name, content] of Object.entries(files)) {
		const target = join(root, 'docs', name)
		await mkdir(dirname(target), { recursive: true })
		await Bun.write(target, content)
	}
	return join(root, 'docs')
}

test('builds ordered sections from front matter and folders', async () => {
	const root = await docs({
		'site.json': JSON.stringify({ title: 'Test', sectionOrder: ['Overview', 'Guide'] }),
		'index.md': '---\ntitle: Home\nsection: Overview\norder: 1\n---\n# Home',
		'guide/b.md': '---\ntitle: B\norder: 2\n---\n# B',
		'guide/a.md': '---\ntitle: A\norder: 1\n---\n# A',
		'_hidden.md': '# Hidden',
		'guide/draft.md': '---\ndraft: true\n---\n# Draft'
	})
	const nav = await buildNav(root)
	expect(nav.sections.map((section) => section.title)).toEqual(['Overview', 'Guide'])
	expect(nav.sections[1]?.pages.map((page) => page.path)).toEqual(['guide/a', 'guide/b'])
})

test('falls back to the first heading and the folder name', async () => {
	const nav = await buildNav(await docs({ 'reference/api-notes.md': '# API notes\n\nText' }))
	expect(nav.sections[0]?.title).toBe('Reference')
	expect(nav.sections[0]?.pages[0]?.title).toBe('API notes')
})
