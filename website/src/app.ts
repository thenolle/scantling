import { byId, fetchText, isTyping } from './dom.ts'
import { escapeHtml } from './escape.ts'
import { icon } from './icons.ts'
import { renderMarkdown } from './markdown.ts'
import type { Rendered } from './markdown.ts'
import { flatten, parseNav } from './nav.ts'
import type { NavPage } from './nav.ts'
import { createSearch } from './search.ts'
import type { SearchHit } from './search.ts'
import { pageMeta } from './seo.ts'
import { currentTheme, toggleTheme } from './theme.ts'
import { absoluteUrl, pathFromPathname, routeHref } from './urls.ts'
import { articleHtml, notFoundHtml, tocHtml } from './view.ts'

const SKELETON = '<div class="doc"><div class="skeleton title"></div><div class="skeleton"></div><div class="skeleton short"></div></div>'

function setAttribute(selector: string, attribute: string, value: string): void {
	document.head.querySelector(selector)?.setAttribute(attribute, value)
}

async function main(): Promise<void> {
	const base = document.documentElement.dataset['base'] ?? '/'
	const nav = parseNav(JSON.parse(await fetchText(`${base}docs/nav.json`)))
	const pages = flatten(nav)
	const byPath = new Map(pages.map((page) => [page.path, page] as const))
	const content = byId('content')
	const sidebar = byId('sidebar')
	const toc = byId('toc')
	const overlay = byId('overlay')
	const menu = byId('menu')
	const themeButton = byId('theme')
	const dialog = byId<HTMLDialogElement>('search-dialog')
	const input = byId<HTMLInputElement>('search-input')
	const results = byId('results')
	const syncTheme = (): void => { themeButton.innerHTML = icon(currentTheme() === 'dark' ? 'sun' : 'moon', 18) }
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
	const sources = new Map<string, Promise<string>>()
	const loadSource = (path: string): Promise<string> => {
		let pending = sources.get(path)
		if (pending === undefined) {
			pending = fetchText(`${base}docs/${path}.md`)
			sources.set(path, pending)
			pending.catch(() => sources.delete(path))
		}
		return pending
	}
	const rendered = new Map<string, Rendered>()
	const renderPage = async (path: string): Promise<Rendered> => {
		const cached = rendered.get(path)
		if (cached !== undefined) return cached
		const doc = renderMarkdown(await loadSource(path), { docPath: path, base })
		rendered.set(path, doc)
		return doc
	}
	const setActive = (path: string): void => {
		sidebar.querySelectorAll<HTMLAnchorElement>('a').forEach((link) => {
			const active = link.dataset['path'] === path
			link.classList.toggle('active', active)
			if (active) link.setAttribute('aria-current', 'page')
			else link.removeAttribute('aria-current')
		})
	}
	const updateHead = (page: NavPage): void => {
		const meta = pageMeta(nav, page)
		const url = absoluteUrl(nav.url, page.path)
		document.title = meta.title
		setAttribute('meta[name="description"]', 'content', meta.description)
		setAttribute('link[rel="canonical"]', 'href', url)
		setAttribute('meta[property="og:title"]', 'content', meta.title)
		setAttribute('meta[property="og:description"]', 'content', meta.description)
		setAttribute('meta[property="og:url"]', 'content', url)
		setAttribute('meta[name="twitter:title"]', 'content', meta.title)
		setAttribute('meta[name="twitter:description"]', 'content', meta.description)
	}
	let observer: IntersectionObserver | undefined
	const observeHeadings = (): void => {
		observer?.disconnect()
		observer = new IntersectionObserver((entries) => {
			for (const entry of entries) {
				if (!entry.isIntersecting) continue
				toc.querySelectorAll<HTMLAnchorElement>('a').forEach((link) => { link.classList.toggle('active', link.getAttribute('href') === `#${entry.target.id}`) })
			}
		}, { rootMargin: '-72px 0px -70% 0px' })
		content.querySelectorAll<HTMLElement>('h2[id], h3[id]').forEach((heading) => observer?.observe(heading))
	}
	const scrollToHash = (hash: string, reset: boolean): void => {
		if (hash.length > 1) document.getElementById(decodeURIComponent(hash.slice(1)))?.scrollIntoView()
		else if (reset) window.scrollTo(0, 0)
	}
	let current = ''
	let navigation = 0
	const show = async (path: string, hash: string, prerendered: boolean): Promise<void> => {
		const token = ++navigation
		const page = byPath.get(path)
		if (page === undefined) {
			current = ''
			content.innerHTML = notFoundHtml(base)
			toc.innerHTML = ''
			document.title = `Page not found | ${nav.title}`
			setActive('')
			return
		}
		setActive(path)
		if (!prerendered) {
			content.innerHTML = SKELETON
			try {
				const doc = await renderPage(path)
				if (token !== navigation) return
				const index = pages.indexOf(page)
				content.innerHTML = articleHtml(nav, page, doc, base, { previous: pages[index - 1], next: pages[index + 1] })
				toc.innerHTML = tocHtml(doc)
				updateHead(page)
			} catch (error) {
				if (token === navigation) {
					const message = error instanceof Error ? error.message : String(error)
					content.innerHTML = `<div class="doc"><h1>Could not load this page</h1><div class="alert warning"><p>${escapeHtml(message)}</p></div></div>`
				}
				return
			}
			content.focus({ preventScroll: true })
		}
		current = path
		observeHeadings()
		scrollToHash(hash, !prerendered)
	}
	document.addEventListener('click', (event) => {
		if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
		const link = event.target instanceof Element ? event.target.closest<HTMLAnchorElement>('a[href]') : null
		if (link === null || link.target === '_blank' || link.origin !== location.origin || !link.pathname.startsWith(base)) return
		const path = pathFromPathname(base, link.pathname)
		if (!byPath.has(path)) return
		closeNav()
		if (path === current) {
			if (link.hash === '') {
				event.preventDefault()
				window.scrollTo(0, 0)
			}
			return
		}
		event.preventDefault()
		history.pushState(null, '', link.href)
		void show(path, link.hash, false)
	})
	window.addEventListener('popstate', () => {
		const path = pathFromPathname(base, location.pathname)
		if (path === current) scrollToHash(location.hash, false)
		else void show(path, location.hash, false)
	})
	content.addEventListener('click', (event) => {
		const button = event.target instanceof Element ? event.target.closest<HTMLButtonElement>('button.copy') : null
		if (button === null) return
		const code = button.closest('figure')?.querySelector('pre code')?.textContent ?? ''
		void navigator.clipboard.writeText(code).then(() => {
			button.innerHTML = icon('check', 14)
			setTimeout(() => { button.innerHTML = icon('copy', 14) }, 1500)
		})
	})
	content.querySelectorAll<HTMLButtonElement>('button.copy').forEach((button) => {
		button.innerHTML = icon('copy', 14)
	})
	const search = createSearch(pages, loadSource)
	let hits: SearchHit[] = []
	let selected = 0
	let searchToken = 0
	const drawHits = (): void => { results.innerHTML = hits.length === 0 ? `<p class="empty">${input.value.trim() === '' ? 'Type to search the documentation.' : 'No results found.'}</p>` : hits.map((hit, index) => `<a role="option" aria-selected="${index === selected}" href="${escapeHtml(routeHref(base, hit.page.path))}"><span class="hit-title">${escapeHtml(hit.page.title)}<small>${escapeHtml(hit.page.section)}</small></span><span class="hit-snippet">${escapeHtml(hit.snippet)}</span></a>`).join('') }
	const openSearch = (): void => {
		if (dialog.open) return
		dialog.showModal()
		input.focus()
		input.select()
		drawHits()
	}
	input.addEventListener('input', () => {
		const token = ++searchToken
		void search(input.value.trim()).then((found) => {
			if (token !== searchToken) return
			hits = found
			selected = 0
			drawHits()
		})
	})
	input.addEventListener('keydown', (event) => {
		if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
			event.preventDefault()
			selected = Math.max(0, Math.min(hits.length - 1, selected + (event.key === 'ArrowDown' ? 1 : -1)))
			drawHits()
			results.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: 'nearest' })
		} else if (event.key === 'Enter') {
			results.querySelectorAll<HTMLAnchorElement>('a')[selected]?.click()
		}
	})
	results.addEventListener('click', () => dialog.close())
	dialog.addEventListener('click', (event) => {
		if (event.target === dialog) dialog.close()
	})
	byId('open-search').addEventListener('click', openSearch)
	const step = (delta: number): void => {
		const target = pages[pages.findIndex((page) => page.path === current) + delta]
		if (target !== undefined) byId('sidebar').querySelector<HTMLAnchorElement>(`a[data-path="${CSS.escape(target.path)}"]`)?.click()
	}
	document.addEventListener('keydown', (event) => {
		if (event.key === 'Escape') closeNav()
		if ((event.key === 'k' || event.key === 'K') && (event.metaKey || event.ctrlKey)) {
			event.preventDefault()
			openSearch()
			return
		}
		if (isTyping(event.target) || dialog.open || event.metaKey || event.ctrlKey || event.altKey) return
		if (event.key === '/') {
			event.preventDefault()
			openSearch()
		} else if (event.key === '[') step(-1)
		else if (event.key === ']') step(1)
	})
	const initial = document.body.dataset['page'] ?? ''
	if (initial === '') void show(pathFromPathname(base, location.pathname), location.hash, false)
	else void show(initial, location.hash, true)
}

main().catch((error: unknown) => {
	console.error(error)
})
