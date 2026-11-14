import { cp, mkdir, rm } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { flatten } from '../src/nav.ts'
import { DESCRIPTION_RANGE, TITLE_RANGE, pageMeta } from '../src/seo.ts'
import { basePathOf, ogName } from '../src/urls.ts'
import { bundleApp, docsRoot, siteRoot } from './bundle.ts'
import { buildNav } from './docs.ts'
import { renderIcon, renderOgImage } from './png.ts'
import { buildManifest, buildRobots, buildSitemap, lastModified, renderNotFoundHtml, renderPageHtml } from './prerender.ts'

const dist = join(siteRoot, 'dist')
const override = process.env['DOCS_SITE_URL']

async function write(path: string, content: string | Uint8Array): Promise<void> {
	await mkdir(dirname(path), { recursive: true })
	await Bun.write(path, content)
}

await rm(dist, { recursive: true, force: true })
await mkdir(dist, { recursive: true })
await cp(join(siteRoot, 'public'), dist, { recursive: true })
await cp(docsRoot, join(dist, 'docs'), { recursive: true })
await bundleApp(dist, true)

const nav = await buildNav(docsRoot, override === undefined ? {} : { url: override })
const base = basePathOf(nav.url)
const context = { nav, docsRoot, base }
const modified = new Map<string, string | undefined>()
const warnings: string[] = []

await write(join(dist, 'docs', 'nav.json'), JSON.stringify(nav))
for (const page of flatten(nav)) {
	const html = await renderPageHtml(context, page.path)
	if (html === undefined) throw new Error(`Could not render ${page.path}`)
	await write(join(dist, page.path === 'index' ? 'index.html' : join(page.path, 'index.html')), html)
	modified.set(page.path, await lastModified(docsRoot, page.path))
	const meta = pageMeta(nav, page)
	if (meta.title.length < TITLE_RANGE[0] || meta.title.length > TITLE_RANGE[1]) warnings.push(`${page.path}: title is ${meta.title.length} characters`)
	if (meta.description.length < DESCRIPTION_RANGE[0] || meta.description.length > DESCRIPTION_RANGE[1]) warnings.push(`${page.path}: description is ${meta.description.length} characters`)
	const image = renderOgImage({ brand: nav.title, kicker: page.path === 'index' ? 'Documentation' : page.section, title: page.path === 'index' ? nav.descriptor : page.title, footer: `${new URL(nav.url).host}${base.replace(/\/$/, '')}` })
	await write(join(dist, 'og', `${ogName(page.path)}.png`), image)
}

await write(join(dist, 'favicon-32.png'), renderIcon(32, true))
await write(join(dist, 'apple-touch-icon.png'), renderIcon(180, false))
await write(join(dist, 'icon-192.png'), renderIcon(192, true))
await write(join(dist, 'icon-512.png'), renderIcon(512, true))
await write(join(dist, 'site.webmanifest'), buildManifest(nav, base))
await write(join(dist, 'sitemap.xml'), buildSitemap(nav, modified))
await write(join(dist, 'robots.txt'), buildRobots(nav))
await write(join(dist, '404.html'), renderNotFoundHtml(context))
await write(join(dist, '.nojekyll'), '')

for (const warning of warnings) console.warn(`SEO warning: ${warning}`)
console.log(`Documentation built in ${dist}: ${flatten(nav).length} pages under ${nav.url}`)
