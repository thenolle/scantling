import { join } from 'node:path'
import { serve, staticFiles } from '@gambrel/bun'
import { createApp, type App } from '@gambrel/core'
import { bundleApp, docsRoot, siteRoot } from './bundle.ts'
import { buildNav } from './docs.ts'

const publicDir = join(siteRoot, 'public')
const cacheDir = join(siteRoot, '.cache')

/**
 * The development server: the HTML shell at `/`, the bundled application at `/app.js`,
 * a live `nav.json` generated from `docs/`, the Markdown sources under `/docs`,
 * and the static assets in `public/`.
 *
 * `staticFiles` answers every request under its prefix, including missing files,
 * so each directory needs its own distinct prefix or an explicit route.
 */
export function createSiteApp(): App {
	return createApp()
		.use(async (context, next) => {
			if (context.method !== 'GET' && context.method !== 'HEAD') return next()
			const head = context.method === 'HEAD'
			if (context.url.pathname === '/') return new Response(head ? null : Bun.file(join(publicDir, 'index.html')), { headers: { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-cache' } })
			if (context.url.pathname === '/app.js') return new Response(head ? null : Bun.file(join(cacheDir, 'app.js')), { headers: { 'content-type': 'text/javascript; charset=utf-8', 'cache-control': 'no-cache' } })
			if (context.url.pathname === '/docs/nav.json') return Response.json(await buildNav(docsRoot), { headers: { 'cache-control': 'no-cache' } })
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
