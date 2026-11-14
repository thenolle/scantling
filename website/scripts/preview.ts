import { join } from 'node:path'
import { siteRoot } from './bundle.ts'

const dist = join(siteRoot, 'dist')
const port = Number(process.env['DOCS_PORT'] ?? 4200)

const server = Bun.serve({
	hostname: '127.0.0.1',
	port,
	async fetch(request) {
		const pathname = decodeURIComponent(new URL(request.url).pathname)
		if (pathname.includes('..')) return new Response('Bad Request', { status: 400 })
		const file = Bun.file(join(dist, pathname.endsWith('/') ? `${pathname}index.html` : pathname))
		if (await file.exists()) return new Response(file)
		return new Response(Bun.file(join(dist, '404.html')), { status: 404 })
	}
})

console.log(`Previewing ${dist} at ${server.url.href}`)
