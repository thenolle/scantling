import { byId, fetchText, isTyping } from './dom.ts'
import { escapeHtml } from './escape.ts'
import { renderMarkdown } from './markdown.ts'
import type { Rendered } from './markdown.ts'
import { flatten, parseNav } from './nav.ts'
import type { NavPage } from './nav.ts'
import { createSearch } from './search.ts'
import { currentTheme, toggleTheme } from './theme.ts'

const SKELETON = '<div class="doc"><div class="skeleton title"></div><div class="skeleton"></div><div class="skeleton short"></div></div>'

interface Route {
	readonly path: string
	readonly anchor: string | undefined
}

function readRoute(): Route {
	const [path = '', anchor] = location.hash.replace(/^#\/?/, '').split('@')
	try { return { path: decodeURIComponent(path), anchor: anchor === undefined ? undefined : decodeURIComponent(anchor) } } catch { return { path, anchor } }
}

function readingMinutes(html: string): number {
	const words = html.replace(/<[^>]+>/g, ' ').split(/\s+/).filter((word) => word.length > 0).length
	return Math.max(1, Math.round(words / 200))
}

function pagerCard(page: NavPage | undefined, kind: 'prev' | 'next'): string {
	if (page === undefined) return '<span></span>'
	const label = kind === 'prev' ? '← Previous' : 'Next →'
	return (`<a class="pager-card ${kind}" href="#/${escapeHtml(page.path)}"><span class="pager-label">${label}</span><span class="pager-title">${escapeHtml(page.title)}</span></a>`)
}

function showError(title: string, error: unknown): string {
	const message = error instanceof Error ? error.message : String(error)
	return `<div class="doc"><h1>${escapeHtml(title)}</h1><aside class="alert warning"><p>${escapeHtml(message)}</p></aside></div>`
}

async function main(): Promise<void> {
	const nav = parseNav(JSON.parse(await fetchText('docs/nav.json')))
	const pages = flatten(nav)
	const byPath = new Map(pages.map((page) => [page.path, page] as const))
	const home = byPath.has('index') ? 'index' : (pages[0]?.path ?? '')
	const content = byId('content')
	const sidebar = byId('sidebar')
	const toc = byId('toc')
	const overlay = byId('overlay')
	const menu = byId('menu')
	const searchInput = byId<HTMLInputElement>('search')
	const results = byId('results')
	const themeButton = byId('theme')
	document.title = `${nav.title} documentation`
	byId('brand-name').textContent = nav.title
	byId('version').textContent = `v${nav.version}`
	byId<HTMLAnchorElement>('repo').href = nav.repo
	sidebar.innerHTML = nav.sections.map((section) => `<section class="nav-section"><h2>${escapeHtml(section.title)}</h2><ul>` + section.pages.map((page) => `<li><a href="#/${escapeHtml(page.path)}" data-path="${escapeHtml(page.path)}">${escapeHtml(page.title)}</a></li>`).join('') + '</ul></section>').join('')
	const syncTheme = (): void => {
		themeButton.textContent = currentTheme() === 'dark' ? 'Light' : 'Dark'
		themeButton.setAttribute('aria-label', `Switch to ${currentTheme() === 'dark' ? 'light' : 'dark'} theme`)
	}
	syncTheme()
	themeButton.addEventListener('click', () => {
		toggleTheme()
		syncTheme()
	})
	const closeNav = (): void => {
		document.body.classList.remove('nav-open')
		overlay.hidden = true
		menu.setAttribute('aria-expanded', 'false')
	}
	menu.addEventListener('click', () => {
		const open = document.body.classList.toggle('nav-open')
		overlay.hidden = !open
		menu.setAttribute('aria-expanded', String(open))
	})
	overlay.addEventListener('click', closeNav)
	sidebar.addEventListener('click', (event) => {
		if (event.target instanceof Element && event.target.closest('a') !== null) closeNav()
	})
	const sources = new Map<string, Promise<string>>()
	const loadSource = (path: string): Promise<string> => {
		let pending = sources.get(path)
		if (pending === undefined) {
			pending = fetchText(`docs/${path}.md`)
			sources.set(path, pending)
			pending.catch(() => sources.delete(path))
		}
		return pending
	}
	const rendered = new Map<string, Rendered>()
	const renderPage = async (path: string): Promise<Rendered> => {
		const cached = rendered.get(path)
		if (cached !== undefined) return cached
		const doc = renderMarkdown(await loadSource(path), { docPath: path })
		rendered.set(path, doc)
		return doc
	}
	const search = createSearch(pages, loadSource)
	let searchToken = 0
	const showResults = async (): Promise<void> => {
		const token = ++searchToken
		const query = searchInput.value.trim()
		if (query === '') {
			results.hidden = true
			return
		}
		const hits = await search(query)
		if (token !== searchToken) return
		results.innerHTML = hits.length === 0 ? '<p class="empty">No matches</p>' : hits.map((hit) => `<a href="#/${escapeHtml(hit.page.path)}"><strong>${escapeHtml(hit.page.title)}</strong><span>${escapeHtml(hit.snippet)}</span></a>`).join('')
		results.hidden = false
	}
	const clearSearch = (): void => {
		searchToken += 1
		results.hidden = true
		searchInput.value = ''
	}
	searchInput.addEventListener('input', () => void showResults())
	searchInput.addEventListener('keydown', (event) => {
		if (event.key === 'Enter') results.querySelector<HTMLAnchorElement>('a')?.click()
		if (event.key === 'Escape') clearSearch()
	})
	results.addEventListener('click', clearSearch)
	const setActive = (page: NavPage | undefined): void => {
		sidebar.querySelectorAll<HTMLAnchorElement>('a').forEach((link) => {
			const active = page !== undefined && link.dataset['path'] === page.path
			link.classList.toggle('active', active)
			if (active) link.setAttribute('aria-current', 'page')
			else link.removeAttribute('aria-current')
		})
		sidebar.querySelector('a.active')?.scrollIntoView({ block: 'nearest' })
	}
	let observer: IntersectionObserver | undefined
	const buildToc = (path: string, doc: Rendered): void => {
		observer?.disconnect()
		const entries = doc.headings.filter((heading) => heading.level === 2 || heading.level === 3)
		toc.innerHTML = entries.length === 0 ? '' : '<h2>On this page</h2><ul>' + entries.map((heading) => `<li><a href="#/${escapeHtml(path)}@${heading.id}" data-id="${heading.id}" data-level="${heading.level}">${escapeHtml(heading.text)}</a></li>`).join('') + '</ul>'
		observer = new IntersectionObserver((observed) => {
			for (const entry of observed) {
				if (!entry.isIntersecting) continue
				toc.querySelectorAll<HTMLAnchorElement>('a').forEach((link) => { link.classList.toggle('active', link.dataset['id'] === entry.target.id) })
			}
		}, { rootMargin: '-72px 0px -70% 0px' })
		content.querySelectorAll<HTMLElement>('h2[id], h3[id]').forEach((heading) => observer?.observe(heading))
	}
	const scrollToAnchor = (anchor: string | undefined): void => {
		if (anchor === undefined) {
			window.scrollTo(0, 0)
			return
		}
		content.querySelector(`[id="${CSS.escape(anchor)}"]`)?.scrollIntoView()
	}
	let currentPath: string | undefined
	let navigation = 0
	const show = async (path: string, anchor: string | undefined): Promise<void> => {
		const token = ++navigation
		const page = byPath.get(path)
		if (page === undefined) {
			currentPath = undefined
			content.innerHTML = `<div class="doc"><h1>Page not found</h1><p>There is no page at <code>${escapeHtml(path)}</code>.</p><p><a href="#/">Back to the start</a></p></div>`
			toc.innerHTML = ''
			document.title = `Not found · ${nav.title}`
			setActive(undefined)
			return
		}
		setActive(page)
		if (path === currentPath) {
			scrollToAnchor(anchor)
			return
		}
		content.innerHTML = SKELETON
		try {
			const doc = await renderPage(path)
			if (token !== navigation) return
			currentPath = path
			const position = pages.findIndex((candidate) => candidate.path === path)
			content.innerHTML =
				'<div class="page-head">' +
				`<nav class="breadcrumb" aria-label="Breadcrumb"><span>${escapeHtml(page.section)}</span><span class="sep">/</span><span class="current">${escapeHtml(page.title)}</span></nav>` +
				`<span class="badge">${readingMinutes(doc.html)} min read</span></div>` +
				`<article class="doc">${doc.html}</article>` +
				`<div class="pager">${pagerCard(pages[position - 1], 'prev')}${pagerCard(pages[position + 1], 'next')}</div>` +
				`<p class="page-foot"><a href="${escapeHtml(nav.repo)}/blob/master/docs/${escapeHtml(path)}.md" target="_blank" rel="noopener noreferrer">Edit this page on GitHub</a></p>`
			document.title = `${doc.title === '' ? page.title : doc.title} · ${nav.title}`
			buildToc(path, doc)
			content.focus({ preventScroll: true })
			scrollToAnchor(anchor)
		} catch (error) {
			if (token === navigation) content.innerHTML = showError('Could not load this page', error)
		}
	}
	content.addEventListener('click', (event) => {
		if (!(event.target instanceof Element)) return
		const button = event.target.closest<HTMLButtonElement>('button.copy')
		if (button === null) return
		const code = button.closest('figure')?.querySelector('pre code')?.textContent ?? ''
		void navigator.clipboard.writeText(code).then(() => {
			button.textContent = 'Copied'
			setTimeout(() => { button.textContent = 'Copy' }, 1500)
		})
	})
	const step = (delta: number): void => {
		const index = pages.findIndex((page) => page.path === currentPath)
		const target = pages[index + delta]
		if (index >= 0 && target !== undefined) location.hash = `#/${target.path}`
	}
	document.addEventListener('keydown', (event) => {
		if (event.key === 'Escape') {
			closeNav()
			return
		}
		if (isTyping(event.target) || event.ctrlKey || event.metaKey || event.altKey) return
		if (event.key === '/') {
			event.preventDefault()
			searchInput.focus()
		} else if (event.key === '[') step(-1)
		else if (event.key === ']') step(1)
	})
	const route = (): void => {
		const { path, anchor } = readRoute()
		void show(path === '' ? home : path, anchor)
	}
	window.addEventListener('hashchange', route)
	route()
}

main().catch((error: unknown) => {
	const content = document.getElementById('content')
	if (content !== null) content.innerHTML = showError('Documentation failed to load', error)
})
