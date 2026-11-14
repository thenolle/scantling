import { afterEach, expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { buildNav } from '../scripts/docs.ts'
import { buildManifest, buildRobots, buildSitemap, renderPageHtml } from '../scripts/prerender.ts'

const roots: string[] = []

afterEach(async () => {
	await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })))
})

async function docs(): Promise<string> {
	const root = await mkdtemp(join(tmpdir(), 'scantling-prerender-'))
	roots.push(root)
	const files: Record<string, string> = {
		'site.json': JSON.stringify({ title: 'Demo', descriptor: 'demo tool', tagline: 'Demo is a demo tool.', author: 'Someone' }),
		'index.md': '---\ntitle: Demo\norder: 1\n---\n# Demo\n\nWelcome.',
		'guide/start.md': '---\ntitle: Start here\nsection: Guide\n---\n# Start here\n\nRun the **demo**.'
	}
	for (const [name, content] of Object.entries(files)) {
		await mkdir(dirname(join(root, 'docs', name)), { recursive: true })
		await Bun.write(join(root, 'docs', name), content)
	}
	return join(root, 'docs')
}

test('prerenders a crawlable page with metadata', async () => {
	const docsRoot = await docs()
	const nav = await buildNav(docsRoot, { url: 'https://example.com/demo/' })
	const html = await renderPageHtml({ nav, docsRoot, base: '/demo/' }, 'guide/start')
	expect(html).toContain('<link rel="canonical" href="https://example.com/demo/guide/start/">')
	expect(html).toContain('<meta property="og:image" content="https://example.com/demo/og/guide-start.png">')
	expect(html).toContain('<h1 id="start-here">Start here')
	expect(html).toContain('<strong>demo</strong>')
	expect(html).toContain('href="/demo/styles.css"')
	expect(await renderPageHtml({ nav, docsRoot, base: '/demo/' }, 'missing')).toBeUndefined()
})

test('generates the sitemap, robots file and manifest', async () => {
	const docsRoot = await docs()
	const nav = await buildNav(docsRoot, { url: 'https://example.com/demo/' })
	const sitemap = buildSitemap(nav, new Map([['index', '2026-11-14T10:00:00+01:00']]))
	expect(sitemap).toContain('<loc>https://example.com/demo/</loc><lastmod>2026-11-14</lastmod>')
	expect(sitemap).toContain('<loc>https://example.com/demo/guide/start/</loc>')
	expect(buildRobots(nav)).toContain('Sitemap: https://example.com/demo/sitemap.xml')
	expect(JSON.parse(buildManifest(nav, '/demo/')).start_url).toBe('/demo/')
})
