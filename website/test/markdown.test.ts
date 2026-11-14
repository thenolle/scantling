import { describe, expect, test } from 'bun:test'
import { renderMarkdown } from '../src/markdown.ts'

const render = (lines: readonly string[], docPath = 'guide/page') => renderMarkdown(lines.join('\n'), { docPath })

describe('markdown renderer', () => {
	test('assigns unique heading ids and reports the title', () => {
		const doc = render(['# Title', '## Same', '## Same'])
		expect(doc.title).toBe('Title')
		expect(doc.headings.map((heading) => heading.id)).toEqual(['title', 'same', 'same-1'])
	})

	test('escapes raw html', () => {
		expect(render(['<script>alert(1)</script>']).html).not.toContain('<script>')
	})

	test('drops javascript links', () => {
		const html = render(['[bad](javascript:alert(1))']).html
		expect(html).not.toContain('<a')
		expect(html).not.toContain('href')
	})

	test('rewrites relative markdown links to routes', () => {
		const html = render(['[x](../architecture/overview.md#top)']).html
		expect(html).toContain('href="#/architecture/overview@top"')
	})

	test('opens external links safely', () => {
		expect(render(['[x](https://example.com)']).html).toContain('rel="noopener noreferrer"')
	})

	test('renders highlighted fenced code', () => {
		const html = render(['```ts', 'const answer = 42', '```']).html
		expect(html).toContain('<span class="tok-k">const</span>')
		expect(html).toContain('<span class="tok-n">42</span>')
	})

	test('renders tables, nested lists and task lists', () => {
		const html = render(['| a | b |', '| --- | ---: |', '| 1 | 2 |', '', '- one', '  - nested', '- [x] done']).html
		expect(html).toContain('<th class="a-right">b</th>')
		expect(html).toContain('<ul><li>nested</li></ul>')
		expect(html).toContain('<li class="task"><input type="checkbox" disabled checked> done</li>')
	})

	test('renders alerts', () => {
		expect(render(['> [!NOTE]', '> Be careful.']).html).toContain('class="alert note"')
	})
})
