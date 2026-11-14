import { escapeHtml } from './escape.ts'
import type { DocNav, NavPage } from './nav.ts'
import { absoluteUrl, ogName } from './urls.ts'

export const OG_WIDTH = 1200
export const OG_HEIGHT = 630
export const TITLE_RANGE = [50, 60] as const
export const DESCRIPTION_RANGE = [120, 160] as const

const LOCALES: Readonly<Record<string, string>> = { en: 'en_US', fr: 'fr_FR', de: 'de_DE', es: 'es_ES' }

/**
 * Chooses a page title of 50-60 characters by trying progressively shorter site suffixes.
 * Falls back to the longest title that fits, then to a truncated one.
 */
export function fitTitle(title: string, name: string, descriptor: string): string {
	const [min, max] = TITLE_RANGE
	const candidates = [
		`${title} | ${name} docs – ${descriptor}`,
		`${title} | ${name} – ${descriptor}`,
		`${title} | ${name} documentation and guides`,
		`${title} | ${name} documentation`,
		`${title} | ${name}`
	]
	const exact = candidates.find((candidate) => candidate.length >= min && candidate.length <= max)
	if (exact !== undefined) return exact
	const fitting = candidates.filter((candidate) => candidate.length <= max).sort((a, b) => b.length - a.length)[0]
	if (fitting !== undefined) return fitting
	return `${title.slice(0, max - 1).trimEnd()}…`
}

/**
 * Normalises a description to at most 160 characters. Text shorter than 120 characters is
 * extended with the first suffix from `extras` that brings it into range.
 */
export function fitDescription(text: string, extras: readonly string[]): string {
	const [min, max] = DESCRIPTION_RANGE
	let value = text.replace(/\s+/g, ' ').trim()
	if (value !== '' && !/[.!?…]$/.test(value)) value += '.'
	if (value.length < min) {
		const options = extras.map((extra) => `${value} ${extra}`.trim()).filter((candidate) => candidate.length <= max)
		const best = options.find((candidate) => candidate.length >= min) ?? options.sort((a, b) => b.length - a.length)[0]
		if (best !== undefined) value = best
	}
	if (value.length <= max) return value
	const cut = value.slice(0, max - 1)
	const space = cut.lastIndexOf(' ')
	return `${cut.slice(0, space > 100 ? space : max - 1)}…`
}

/**
 * The title and description used for a page's `<title>`, meta description and social tags.
 */
export function pageMeta(nav: DocNav, page: NavPage): { title: string; description: string } {
	return {
		title: page.seoTitle !== '' ? page.seoTitle : fitTitle(page.title, nav.title, nav.descriptor),
		description: fitDescription(page.description, [nav.tagline, `${nav.title}: ${nav.descriptor}.`])
	}
}

export interface HeadInput {
	readonly path: string
	readonly heading: string
	readonly title: string
	readonly description: string
	readonly section: string
	readonly modified: string | undefined
	readonly base: string
}

function jsonLd(nav: DocNav, input: HeadInput, canonical: string, image: string): string {
	const site = { '@type': 'WebSite', '@id': `${nav.url}#website`, url: nav.url, name: nav.title, description: nav.description, inLanguage: nav.language }
	const main = input.path === 'index'
		? { '@type': 'SoftwareSourceCode', name: nav.title, description: input.description, codeRepositoryUrl: nav.repo, programmingLanguage: 'TypeScript', url: canonical }
		: {
			'@type': 'TechArticle',
			headline: input.heading,
			description: input.description,
			url: canonical,
			mainEntityOfPage: canonical,
			image,
			inLanguage: nav.language,
			articleSection: input.section,
			...(input.modified === undefined ? {} : { dateModified: input.modified }),
			author: { '@type': 'Person', name: nav.author },
			isPartOf: { '@id': `${nav.url}#website` }
		}
	const crumbs = [{ name: 'Docs', url: nav.url }, ...(input.path === 'index' ? [] : [{ name: input.heading, url: canonical }])]
	const breadcrumbs = {
		'@type': 'BreadcrumbList',
		itemListElement: crumbs.map((crumb, index) => ({ '@type': 'ListItem', position: index + 1, name: crumb.name, item: crumb.url }))
	}
	return JSON.stringify({ '@context': 'https://schema.org', '@graph': [site, main, breadcrumbs] }).replace(/</g, '\\u003c')
}

/**
 * The complete `<head>` contents for a page: title, description, canonical URL, robots,
 * icons, Open Graph, X (Twitter) cards and JSON-LD structured data.
 */
export function buildHead(nav: DocNav, input: HeadInput): string {
	const canonical = absoluteUrl(nav.url, input.path)
	const image = new URL(`og/${ogName(input.path)}.png`, nav.url).href
	const imageAlt = `${input.heading} – ${nav.title} documentation`
	const e = escapeHtml
	const name = (key: string, content: string): string => `<meta name="${key}" content="${e(content)}">`
	const prop = (key: string, content: string): string => `<meta property="${key}" content="${e(content)}">`
	const article = input.path !== 'index'
	return [
		'<meta charset="utf-8">',
		name('viewport', 'width=device-width, initial-scale=1'),
		`<title>${e(input.title)}</title>`,
		name('description', input.description),
		`<link rel="canonical" href="${e(canonical)}">`,
		name('robots', 'index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1'),
		...(nav.author === '' ? [] : [name('author', nav.author)]),
		'<meta name="theme-color" content="#09090b" media="(prefers-color-scheme: dark)">',
		'<meta name="theme-color" content="#ffffff" media="(prefers-color-scheme: light)">',
		name('color-scheme', 'dark light'),
		`<link rel="icon" href="${e(input.base)}favicon.svg" type="image/svg+xml">`,
		`<link rel="icon" href="${e(input.base)}favicon-32.png" sizes="32x32" type="image/png">`,
		`<link rel="apple-touch-icon" href="${e(input.base)}apple-touch-icon.png">`,
		`<link rel="manifest" href="${e(input.base)}site.webmanifest">`,
		`<link rel="sitemap" type="application/xml" href="${e(input.base)}sitemap.xml">`,
		prop('og:site_name', nav.title),
		prop('og:type', article ? 'article' : 'website'),
		prop('og:title', input.title),
		prop('og:description', input.description),
		prop('og:url', canonical),
		prop('og:locale', LOCALES[nav.language] ?? nav.language),
		prop('og:image', image),
		prop('og:image:secure_url', image),
		prop('og:image:type', 'image/png'),
		prop('og:image:width', String(OG_WIDTH)),
		prop('og:image:height', String(OG_HEIGHT)),
		prop('og:image:alt', imageAlt),
		...(article && input.modified !== undefined ? [prop('article:modified_time', input.modified)] : []),
		...(article ? [prop('article:section', input.section)] : []),
		name('twitter:card', 'summary_large_image'),
		name('twitter:title', input.title),
		name('twitter:description', input.description),
		name('twitter:image', image),
		name('twitter:image:alt', imageAlt),
		...(nav.twitter === '' ? [] : [name('twitter:site', nav.twitter.startsWith('@') ? nav.twitter : `@${nav.twitter}`)]),
		`<script type="application/ld+json">${jsonLd(nav, input, canonical, image)}</script>`
	].join('\n')
}
