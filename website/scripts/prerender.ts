import { stat } from 'node:fs/promises'
import { join } from 'node:path'
import { escapeHtml } from '../src/escape.ts'
import { renderMarkdown } from '../src/markdown.ts'
import type { DocNav } from '../src/nav.ts'
import { flatten } from '../src/nav.ts'
import { buildHead, pageMeta } from '../src/seo.ts'
import { absoluteUrl } from '../src/urls.ts'
import { articleHtml, documentHtml, notFoundHtml, sidebarHtml, tocHtml } from '../src/view.ts'

export interface RenderContext {
	readonly nav: DocNav
	readonly docsRoot: string
	readonly base: string
}

async function gitDate(docsRoot: string, file: string): Promise<string | undefined> {
	try {
		const subprocess = Bun.spawn(['git', 'log', '-1', '--format=%cI', '--', file], { cwd: docsRoot, stdin: 'ignore', stdout: 'pipe', stderr: 'ignore' })
		const output = (await new Response(subprocess.stdout).text()).trim()
		await subprocess.exited
		return output === '' ? undefined : output
	} catch {
		return undefined
	}
}

/**
 * The date a page was last changed: its last git commit, or the file's modification time.
 */
export async function lastModified(docsRoot: string, path: string): Promise<string | undefined> {
	const file = `${path}.md`
	return (await gitDate(docsRoot, file)) ?? (await stat(join(docsRoot, file)).then((info) => info.mtime.toISOString(), () => undefined))
}

/**
 * Renders a complete, crawlable HTML document for a documentation page, or `undefined` for an unknown path.
 */
export async function renderPageHtml(context: RenderContext, path: string): Promise<string | undefined> {
	const { nav, docsRoot, base } = context
	const pages = flatten(nav)
	const index = pages.findIndex((page) => page.path === path)
	const page = pages[index]
	if (page === undefined) return undefined
	const doc = renderMarkdown(await Bun.file(join(docsRoot, `${path}.md`)).text(), { docPath: path, base })
	const meta = pageMeta(nav, page)
	const head = buildHead(nav, {
		path,
		heading: page.title,
		title: meta.title,
		description: meta.description,
		section: page.section,
		modified: await lastModified(docsRoot, path),
		base
	})
	return documentHtml({
		nav,
		base,
		head,
		path,
		sidebar: sidebarHtml(nav, base, path),
		main: articleHtml(nav, page, doc, base, { previous: pages[index - 1], next: pages[index + 1] }),
		toc: tocHtml(doc)
	})
}

/**
 * The 404 document. It is marked `noindex`.
 */
export function renderNotFoundHtml(context: RenderContext): string {
	const { nav, base } = context
	const head = [
		'<meta charset="utf-8">',
		'<meta name="viewport" content="width=device-width, initial-scale=1">',
		`<title>Page not found | ${escapeHtml(nav.title)}</title>`,
		'<meta name="robots" content="noindex">',
		`<link rel="icon" href="${escapeHtml(base)}favicon.svg" type="image/svg+xml">`
	].join('\n')
	return documentHtml({ nav, base, head, path: '', sidebar: sidebarHtml(nav, base, ''), main: notFoundHtml(base), toc: '' })
}

export function buildSitemap(nav: DocNav, modified: ReadonlyMap<string, string | undefined>): string {
	const entries = flatten(nav).map((page) => {
		const date = modified.get(page.path)
		const lastmod = date === undefined ? '' : `<lastmod>${date.slice(0, 10)}</lastmod>`
		return `<url><loc>${escapeHtml(absoluteUrl(nav.url, page.path))}</loc>${lastmod}</url>`
	})
	return `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${entries.join('\n')}\n</urlset>\n`
}

export function buildRobots(nav: DocNav): string {
	return `User-agent: *\nAllow: /\n\nSitemap: ${new URL('sitemap.xml', nav.url).href}\n`
}

export function buildManifest(nav: DocNav, base: string): string {
	const icon = (size: number) => ({ src: `${base}icon-${size}.png`, sizes: `${size}x${size}`, type: 'image/png' })
	return JSON.stringify({ name: `${nav.title} documentation`, short_name: nav.title, description: nav.description, lang: nav.language, start_url: base, scope: base, display: 'standalone', background_color: '#09090b', theme_color: '#09090b', icons: [icon(192), icon(512)] }, null, 2)
}
