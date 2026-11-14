import { parseFrontMatter } from './frontmatter.ts'
import type { NavPage } from './nav.ts'

export interface SearchHit {
	readonly page: NavPage
	readonly snippet: string
	readonly score: number
}

interface Indexed {
	readonly page: NavPage
	readonly text: string
	readonly lower: string
}

function plain(markdown: string): string {
	return markdown.replace(/\[([^\]]*)\]\([^)]*\)/g, '$1').replace(/[#*_>`|~]/g, ' ').replace(/\s+/g, ' ').trim()
}

function snippet(text: string, term: string): string {
	const index = text.toLowerCase().indexOf(term)
	if (index === -1) return text.slice(0, 100)
	const start = Math.max(0, index - 40)
	const end = Math.min(text.length, index + 100)
	return `${start > 0 ? '…' : ''}${text.slice(start, end)}${end < text.length ? '…' : ''}`
}

/**
 * Creates a full-text search over every documentation page.
 * Pages are loaded and indexed lazily on the first query.
 */
export function createSearch(pages: readonly NavPage[], load: (path: string) => Promise<string>): (query: string) => Promise<SearchHit[]> {
	let index: Promise<Indexed[]> | undefined
	const build = (): Promise<Indexed[]> => {
		index ??= Promise.all(
			pages.map(async (page): Promise<Indexed> => {
				let text = ''
				try { text = plain(parseFrontMatter(await load(page.path)).body) } catch { text = '' }
				return { page, text, lower: `${page.title}\n${text}`.toLowerCase() }
			})
		)
		return index
	}
	return async (query) => {
		const terms = query.toLowerCase().split(/\s+/).filter((term) => term.length > 0)
		if (terms.length === 0) return []
		const hits: SearchHit[] = []
		for (const entry of await build()) {
			if (!terms.every((term) => entry.lower.includes(term))) continue
			const title = entry.page.title.toLowerCase()
			const score = terms.reduce((sum, term) => sum + (title.includes(term) ? 20 : 0) + Math.min(10, entry.lower.split(term).length - 1), 0)
			hits.push({ page: entry.page, snippet: snippet(entry.text, terms[0] ?? ''), score })
		}
		return hits.sort((a, b) => b.score - a.score).slice(0, 8)
	}
}
