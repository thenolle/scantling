import { readdir } from 'node:fs/promises'
import { join } from 'node:path'
import { parseFrontMatter } from '../src/frontmatter.ts'
import type { DocNav, NavPage } from '../src/nav.ts'

interface Site {
	readonly title: string
	readonly description: string
	readonly repo: string
	readonly sectionOrder: readonly string[]
}

interface Entry extends NavPage {
	readonly order: number
}

function prettify(segment: string): string {
	return segment.replace(/[-_]+/g, ' ').replace(/^./, (char) => char.toUpperCase())
}

async function readJson(path: string): Promise<Record<string, unknown>> {
	const file = Bun.file(path)
	const raw: unknown = (await file.exists()) ? await file.json() : {}
	return typeof raw === 'object' && raw !== null && !Array.isArray(raw) ? (raw as Record<string, unknown>) : {}
}

async function readSite(docsRoot: string): Promise<Site> {
	const record = await readJson(join(docsRoot, 'site.json'))
	const text = (key: string, fallback: string): string => {
		const value = record[key]
		return typeof value === 'string' ? value : fallback
	}
	const order = record['sectionOrder']
	return {
		title: text('title', 'Documentation'),
		description: text('description', ''),
		repo: text('repo', ''),
		sectionOrder: Array.isArray(order) ? order.filter((item): item is string => typeof item === 'string') : []
	}
}

async function readVersion(docsRoot: string): Promise<string> {
	const version = (await readJson(join(docsRoot, '..', 'package.json')))['version']
	return typeof version === 'string' ? version : '0.0.0'
}

/**
 * Builds the navigation tree from every Markdown file under `docsRoot`.
 * Front matter keys: `title`, `section`, `order`, `description`, `draft`.
 * Files or folders starting with `_` or `.` are skipped.
 */
export async function buildNav(docsRoot: string): Promise<DocNav> {
	const site = await readSite(docsRoot)
	const files = (await readdir(docsRoot, { recursive: true })).map((file) => file.replace(/\\/g, '/')).filter((file) => file.endsWith('.md') && file.split('/').every((part) => !/^[_.]/.test(part)))
	const entries: Entry[] = []
	for (const file of files.sort()) {
		const { meta, body } = parseFrontMatter(await Bun.file(join(docsRoot, file)).text())
		if (meta['draft'] === 'true') continue
		const path = file.slice(0, -3)
		const folder = path.includes('/') ? (path.split('/')[0] ?? '') : ''
		const order = Number(meta['order'])
		entries.push({
			path,
			title: meta['title'] ?? /^#\s+(.+)$/m.exec(body)?.[1] ?? path,
			section: meta['section'] ?? (folder === '' ? 'Overview' : prettify(folder)),
			description: meta['description'] ?? '',
			order: Number.isFinite(order) ? order : 999
		})
	}
	const grouped = new Map<string, Entry[]>()
	for (const entry of entries) grouped.set(entry.section, [...(grouped.get(entry.section) ?? []), entry])
	const rank = (title: string): number => {
		const index = site.sectionOrder.indexOf(title)
		return index === -1 ? Number.MAX_SAFE_INTEGER : index
	}
	const sections = [...grouped.entries()].sort(([a], [b]) => rank(a) - rank(b) || a.localeCompare(b)).map(([title, list]) => ({ title, pages: list.sort((a, b) => a.order - b.order || a.title.localeCompare(b.title)).map(({ path, title: pageTitle, section, description }) => ({ path, title: pageTitle, section, description })) }))
	return {
		title: site.title,
		description: site.description,
		repo: site.repo,
		version: await readVersion(docsRoot),
		sections
	}
}
