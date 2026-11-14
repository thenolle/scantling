import { join } from 'node:path'
import { serve, staticFiles } from '@gambrel/bun'
import { createApp, type App } from '@gambrel/core'
import { flatten } from '../src/nav.ts'
import { ogName, pathFromPathname } from '../src/urls.ts'
import { bundleApp, docsRoot, siteRoot } from './bundle.ts'
import { buildNav } from './docs.ts'
import { renderIcon, renderOgImage } from './png.ts'
import { renderNotFoundHtml, renderPageHtml } from './prerender.ts'

const publicDir = join(siteRoot, 'public')
const cacheDir = join(siteRoot, '.cache')
const ICONS: Readonly<Record<string, [number, boolean]>> = {
	'/favicon-32.png': [32, true],
	'/apple-touch-icon.png': [180, false],
	'/icon-192.png': [192, true],
	'/icon-512.png': [512, true]
}

const html = (body: string | null, status = 200): Response => new Response(body, { status, headers: { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-cache' } })
const png = (body: Uint8Array | null): Response => {
	const buffer = body === null ? null : new ArrayBuffer(body.byteLength)
	if (body !== null && buffer !== null) new Uint8Array(buffer).set(body)
	return new Response(buffer, { headers: { 'content-type': 'image/png', 'cache-control': 'no-cache' } })
}

/**
 * The development server. Pages are rendered on every request from the Markdown sources, with the
 * same templates and metadata as the production build. `staticFiles` answers every request under
 * its prefix, so the bundle, images and pages are routed explicitly before it.
 */
export function createSiteApp(): App {
	return createApp()
		.use(async (context, next) => {
			if (context.method !== 'GET' && context.method !== 'HEAD') return next()
			const head = context.method === 'HEAD'
			const { pathname } = context.url
			if (pathname === '/app.js') return new Response(head ? null : Bun.file(join(cacheDir, 'app.js')), { headers: { 'content-type': 'text/javascript; charset=utf-8', 'cache-control': 'no-cache' } })
			const nav = await buildNav(docsRoot)
			if (pathname === '/docs/nav.json') return Response.json(nav, { headers: { 'cache-control': 'no-cache' } })
			const icon = ICONS[pathname]
			if (icon !== undefined) return png(head ? null : renderIcon(icon[0], icon[1]))
			if (pathname.startsWith('/og/') && pathname.endsWith('.png')) {
				const page = flatten(nav).find((candidate) => `/og/${ogName(candidate.path)}.png` === pathname)
				if (page === undefined) return html('Not Found', 404)
				const image = renderOgImage({ brand: nav.title, kicker: page.section, title: page.path === 'index' ? nav.descriptor : page.title, footer: 'localhost' })
				return png(head ? null : image)
			}
			if (!/\.[a-z0-9]+$/i.test(pathname)) {
				const context = { nav, docsRoot, base: '/' }
				const page = await renderPageHtml(context, pathFromPathname('/', pathname))
				return page === undefined ? html(head ? null : renderNotFoundHtml(context), 404) : html(head ? null : page)
			}
			return next()
		})
		.use(staticFiles({ directory: docsRoot, prefix: '/docs' }))
		.use(staticFiles({ directory: publicDir, prefix: '/' }))
}

function isAddressInUse(error: unknown): boolean {
	return typeof error === 'object' && error !== null && 'code' in error && error.code === 'EADDRINUSE'
}

if (import.meta.main) {
	await bundleApp(cacheDir, false)
	const port = Number(process.env['DOCS_PORT'] ?? 4000)
	try {
		const server = serve(createSiteApp(), { port, development: true })
		console.log(`Documentation at ${server.url.href}`)
	} catch (error) {
		if (!isAddressInUse(error)) throw error
		console.error(`Port ${port} is already in use. Stop the other process or set DOCS_PORT.`)
		process.exit(1)
	}
}
