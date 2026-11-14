import { describe, expect, test } from 'bun:test'
import type { DocNav } from '../src/nav.ts'
import { buildHead, fitDescription, fitTitle } from '../src/seo.ts'

const nav: DocNav = {
	title: 'Scantling',
	tagline: 'Scantling is a local-first AI coding assistant with its own model, built from scratch in TypeScript and Bun.',
	descriptor: 'local-first AI coding assistant',
	description: 'Documentation.',
	repo: 'https://github.com/thenolle/scantling',
	version: '0.1.0',
	url: 'https://example.github.io/scantling/',
	language: 'en',
	author: 'Nolly',
	twitter: '',
	sections: []
}

const input = {
	path: 'guide/getting-started',
	heading: 'Getting "started"',
	title: 'Getting started | Scantling documentation and guides',
	description: 'Install the tooling and verify a fresh checkout.',
	section: 'Guide',
	modified: '2026-11-14T12:00:00+01:00',
	base: '/scantling/'
}

describe('seo', () => {
	test('titles land between 50 and 60 characters', () => {
		for (const title of ['Roadmap', 'Limitations', 'Getting started', 'Development workflow', 'Configuration and logging']) {
			const fitted = fitTitle(title, 'Scantling', 'local-first AI coding assistant')
			expect(fitted.length).toBeGreaterThanOrEqual(50)
			expect(fitted.length).toBeLessThanOrEqual(60)
		}
	})

	test('short descriptions are extended and long ones are clamped', () => {
		const extended = fitDescription('Install the tooling and verify a fresh checkout.', [nav.tagline])
		expect(extended.length).toBeGreaterThanOrEqual(120)
		expect(extended.length).toBeLessThanOrEqual(160)
		const clamped = fitDescription('word '.repeat(80), [])
		expect(clamped.length).toBeLessThanOrEqual(160)
		expect(clamped.endsWith('…')).toBe(true)
	})

	test('the head carries complete Open Graph and X tags', () => {
		const head = buildHead(nav, input)
		const image = 'https://example.github.io/scantling/og/guide-getting-started.png'
		expect(head).toContain(`<meta property="og:image" content="${image}">`)
		expect(head).toContain('<meta property="og:image:width" content="1200">')
		expect(head).toContain('<meta property="og:site_name" content="Scantling">')
		expect(head).toContain('<meta property="og:type" content="article">')
		expect(head).toContain('<meta name="twitter:card" content="summary_large_image">')
		expect(head).toContain(`<meta name="twitter:image" content="${image}">`)
		expect(head).toContain('<link rel="canonical" href="https://example.github.io/scantling/guide/getting-started/">')
		expect(head).toContain('<meta property="og:title" content="Getting started | Scantling documentation and guides">')
		expect(head).not.toContain('twitter:site')
	})

	test('escapes text and emits valid JSON-LD', () => {
		const head = buildHead(nav, input)
		expect(head).toContain('Getting &quot;started&quot; – Scantling documentation')
		const json = /<script type="application\/ld\+json">(.*)<\/script>/.exec(head)?.[1] ?? ''
		const graph = (JSON.parse(json) as { '@graph': { '@type': string }[] })['@graph']
		expect(graph.map((node) => node['@type'])).toEqual(['WebSite', 'TechArticle', 'BreadcrumbList'])
	})

	test('adds twitter:site only when a handle is configured', () => {
		expect(buildHead({ ...nav, twitter: 'example' }, input)).toContain('<meta name="twitter:site" content="@example">')
	})
})
