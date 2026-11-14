import { escapeHtml } from './escape.ts'
import { icon } from './icons.ts'
import type { Rendered } from './markdown.ts'
import type { DocNav, NavPage } from './nav.ts'
import { routeHref } from './urls.ts'

export interface Neighbours {
	readonly previous: NavPage | undefined
	readonly next: NavPage | undefined
}

export interface DocumentParts {
	readonly nav: DocNav
	readonly base: string
	readonly head: string
	readonly path: string
	readonly sidebar: string
	readonly main: string
	readonly toc: string
}

export function readingMinutes(html: string): number {
	const words = html.replace(/<[^>]+>/g, ' ').split(/\s+/).filter((word) => word.length > 0).length
	return Math.max(1, Math.round(words / 200))
}

/**
 * The sidebar navigation, with the active page marked.
 */
export function sidebarHtml(nav: DocNav, base: string, active: string): string {
	return nav.sections.map((section) => {
		const links = section.pages.map((page) => {
			const current = page.path === active ? ' class="active" aria-current="page"' : ''
			return `<li><a href="${escapeHtml(routeHref(base, page.path))}" data-path="${escapeHtml(page.path)}"${current}>${escapeHtml(page.title)}</a></li>`
		}).join('')
		return `<section class="nav-section"><h2>${escapeHtml(section.title)}</h2><ul>${links}</ul></section>`
	}).join('')
}

/**
 * The "On this page" list for a rendered document.
 */
export function tocHtml(doc: Rendered): string {
	const entries = doc.headings.filter((heading) => heading.level === 2 || heading.level === 3)
	if (entries.length === 0) return ''
	const items = entries.map((heading) => `<li><a href="#${heading.id}" data-level="${heading.level}">${escapeHtml(heading.text)}</a></li>`).join('')
	return `<h2>On this page</h2><ul>${items}</ul>`
}

function pagerCard(base: string, page: NavPage | undefined, kind: 'prev' | 'next'): string {
	if (page === undefined) return '<span></span>'
	const label = kind === 'prev' ? `${icon('arrowLeft')}Previous` : `Next${icon('arrowRight')}`
	return `<a class="pager-card ${kind}" href="${escapeHtml(routeHref(base, page.path))}"><span class="pager-label">${label}</span><span class="pager-title">${escapeHtml(page.title)}</span></a>`
}

/**
 * The main column of a documentation page: breadcrumb, article, pager and edit link.
 */
export function articleHtml(nav: DocNav, page: NavPage, doc: Rendered, base: string, neighbours: Neighbours): string {
	const chevron = icon('chevronRight', 14)
	return (
		'<div class="page-head">' +
		`<nav class="breadcrumb" aria-label="Breadcrumb"><a href="${escapeHtml(base)}">Docs</a>${chevron}<span>${escapeHtml(page.section)}</span>${chevron}<span class="current" aria-current="page">${escapeHtml(page.title)}</span></nav>` +
		`<span class="badge">${readingMinutes(doc.html)} min read</span></div>` +
		`<article class="doc">${doc.html}</article>` +
		`<div class="pager">${pagerCard(base, neighbours.previous, 'prev')}${pagerCard(base, neighbours.next, 'next')}</div>` +
		`<div class="page-foot"><a class="btn btn-ghost btn-sm" href="${escapeHtml(nav.repo)}/blob/master/docs/${escapeHtml(page.path)}.md" target="_blank" rel="noopener noreferrer">${icon('external', 14)}Edit this page on GitHub</a></div>`
	)
}

export function notFoundHtml(base: string): string {
	return `<div class="doc"><h1>Page not found</h1><p>This page does not exist or has moved.</p><p><a href="${escapeHtml(base)}">Back to the documentation</a></p></div>`
}

/**
 * A complete HTML document: head, header, sidebar, main column, table of contents and search dialog.
 */
export function documentHtml(parts: DocumentParts): string {
	const { nav, base } = parts
	const b = escapeHtml(base)
	return `<!doctype html>
<html lang="${escapeHtml(nav.language)}" data-theme="dark" data-base="${b}">
<head>
${parts.head}
<script>try{var t=localStorage.getItem('scantling-theme');if(t==='light'||t==='dark')document.documentElement.dataset.theme=t}catch(e){}</script>
<link rel="stylesheet" href="${b}styles.css">
</head>
<body data-page="${escapeHtml(parts.path)}">
<a class="skip" href="#content">Skip to content</a>
<header class="site-header"><div class="header-inner">
<button id="menu" type="button" class="btn btn-ghost btn-icon" aria-label="Toggle navigation" aria-expanded="false">${icon('menu', 18)}</button>
<a class="brand" href="${b}">${icon('logo', 20)}<span>${escapeHtml(nav.title)}</span></a>
<span class="badge">v${escapeHtml(nav.version)}</span>
<div class="spacer"></div>
<button id="open-search" type="button" class="search-trigger" aria-label="Search documentation">${icon('search')}<span>Search documentation...</span><kbd>/</kbd></button>
<a class="btn btn-ghost btn-icon" href="${escapeHtml(nav.repo)}" target="_blank" rel="noopener noreferrer" aria-label="Source code on GitHub">${icon('code', 18)}</a>
<button id="theme" type="button" class="btn btn-ghost btn-icon" aria-label="Switch theme">${icon('sun', 18)}</button>
</div></header>
<div class="layout">
<aside id="sidebar" class="sidebar" aria-label="Documentation">${parts.sidebar}</aside>
<main id="content" tabindex="-1">${parts.main}</main>
<aside id="toc" aria-label="On this page">${parts.toc}</aside>
</div>
<div id="overlay" hidden></div>
<dialog id="search-dialog" aria-label="Search documentation"><div class="command"><div class="command-input">${icon('search')}<input id="search-input" type="text" placeholder="Search documentation..." autocomplete="off" spellcheck="false" aria-label="Search documentation"></div><div id="results" class="command-list" role="listbox"></div></div></dialog>
<script type="module" src="${b}app.js"></script>
</body>
</html>
`
}
