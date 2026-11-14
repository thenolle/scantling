import { readdir } from 'node:fs/promises'
import { join } from 'node:path'
import { parseFrontMatter } from '../src/frontmatter.ts'
import type { DocNav, NavPage } from '../src/nav.ts'

export interface NavOptions {
	/** Overrides `url` from `site.json`, for example to preview a build locally. */
	readonly url?: string
}

interface Entry extends NavPage {
	readonly order: number
}

const prettify = (segment: string): string => segment.replace(/[-_]+/g, ' ').replace(/^./, (char) => char.toUpperCase())

async function readJson(path: string): Promise<Record<string, unknown>> {
	const file = Bun.file(path)
	const raw: unknown = (await file.exists()) ? await file.json() : {}
	return typeof raw === 'object' && raw !== null && !Array.isArray(raw) ? (raw as Record<string, unknown>) : {}
}

function summarize(body: string): string {
	const block = body.split(/\n\s*\n/).map((part) => part.trim()).find((part) => part !== '' && !/^(#|```|~~~|>|[-*+]\s|\d+[.)]\s|\||---)/.test(part))
	return (block ?? '').replace(/\[([^\]]*)\]\([^)]*\)/g, '$1').replace(/[`*_~]/g, '').replace(/\s+/g, ' ').trim()
}

/**
 * Builds the navigation tree and site settings from `docs/site.json` and every Markdown file under `docsRoot`.
 * Front matter keys: `title`, `seoTitle`, `section`, `order`, `description`, `draft`.
 * Files or folders starting with `_` or `.` are skipped.
 */
export async function buildNav(docsRoot: string, options: NavOptions = {}): Promise<DocNav> {
	const site = await readJson(join(docsRoot, 'site.json'))
	const text = (key: string, fallback: string): string => {
		const value = site[key]
		return typeof value === 'string' ? value : fallback
	}
	const order = site['sectionOrder']
	const sectionOrder = Array.isArray(order) ? order.filter((item): item is string => typeof item === 'string') : []
	const url = new URL(options.url ?? text('url', 'http://localhost:4000/')).href.replace(/\/?$/, '/')
	const files = (await readdir(docsRoot, { recursive: true })).map((file) => file.replace(/\\/g, '/')).filter((file) => file.endsWith('.md') && file.split('/').every((part) => !/^[_.]/.test(part)))
	const entries: Entry[] = []
	for (const file of files.sort()) {
		const { meta, body } = parseFrontMatter(await Bun.file(join(docsRoot, file)).text())
		if (meta['draft'] === 'true') continue
		const path = file.slice(0, -3)
		const folder = path.includes('/') ? (path.split('/')[0] ?? '') : ''
		const position = Number(meta['order'])
		entries.push({
			path,
			title: meta['title'] ?? /^#\s+(.+)$/m.exec(body)?.[1] ?? path,
			seoTitle: meta['seoTitle'] ?? '',
			section: meta['section'] ?? (folder === '' ? 'Overview' : prettify(folder)),
			description: meta['description'] ?? summarize(body),
			order: Number.isFinite(position) ? position : 999
		})
	}
	const grouped = new Map<string, Entry[]>()
	for (const entry of entries) grouped.set(entry.section, [...(grouped.get(entry.section) ?? []), entry])
	const rank = (title: string): number => {
		const index = sectionOrder.indexOf(title)
		return index === -1 ? Number.MAX_SAFE_INTEGER : index
	}
	const manifest = await readJson(join(docsRoot, '..', 'package.json'))
	const version = manifest['version']
	return {
		title: text('title', 'Documentation'),
		tagline: text('tagline', ''),
		descriptor: text('descriptor', 'documentation'),
		description: text('description', ''),
		repo: text('repo', ''),
		version: typeof version === 'string' ? version : '0.0.0',
		url,
		language: text('language', 'en'),
		author: text('author', ''),
		twitter: text('twitter', ''),
		sections: [...grouped.entries()].sort(([a], [b]) => rank(a) - rank(b) || a.localeCompare(b)).map(([title, list]) => ({ title, pages: list.sort((a, b) => a.order - b.order || a.title.localeCompare(b.title)).map(({ path, title: pageTitle, seoTitle, section, description }) => ({ path, title: pageTitle, seoTitle, section, description })) }))
	}
}
