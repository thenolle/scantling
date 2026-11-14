import { escapeHtml } from './escape.ts'
import { parseFrontMatter } from './frontmatter.ts'
import { highlight } from './highlight.ts'
import { routeHref } from './urls.ts'

export interface Heading {
	readonly level: number
	readonly id: string
	readonly text: string
}

export interface Rendered {
	readonly title: string
	readonly html: string
	readonly headings: readonly Heading[]
}

export interface RenderOptions {
	readonly docPath: string
	readonly base?: string
}

interface State {
	readonly options: RenderOptions
	readonly headings: Heading[]
	readonly used: Map<string, number>
}

interface ListItem {
	readonly lines: string[]
	readonly indent: number
}

const FENCE = /^ {0,3}(`{3,}|~{3,})\s*([\w+#.-]*)[^\n]*$/
const HEADING = /^ {0,3}(#{1,6})\s+(.+?)(?:\s+#+)?\s*$/
const RULE = /^ {0,3}([-*_])(?:\s*\1){2,}\s*$/
const LIST_ITEM = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/
const QUOTE = /^ {0,3}>/
const TABLE_DIVIDER = /^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$/
const ALERT = /^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*$/i

function slugify(text: string): string {
	return text.toLowerCase().replace(/[^\p{L}\p{N}\s-]/gu, '').trim().replace(/\s+/g, '-') || 'section'
}

function uniqueId(base: string, used: Map<string, number>): string {
	const count = used.get(base) ?? 0
	used.set(base, count + 1)
	return count === 0 ? base : `${base}-${count}`
}

function leadingSpaces(line: string): number {
	return line.length - line.trimStart().length
}

function resolvePath(docPath: string, target: string): string {
	const segments = target.startsWith('/') ? [] : docPath.split('/').slice(0, -1)
	for (const segment of target.split('/')) {
		if (segment === '' || segment === '.') continue
		if (segment === '..') segments.pop()
		else segments.push(segment)
	}
	return segments.join('/')
}

function resolveHref(href: string, options: RenderOptions): { href: string; external: boolean } | undefined {
	if (/^(https?:|mailto:)/i.test(href)) return { href, external: !/^mailto:/i.test(href) }
	if (/^[a-z][a-z0-9+.-]*:/i.test(href)) return undefined
	if (href.startsWith('#')) return { href, external: false }
	const [target = '', anchor] = href.split('#')
	if (!target.endsWith('.md')) return { href, external: false }
	const route = routeHref(options.base ?? '/', resolvePath(options.docPath, target.slice(0, -3)))
	return { href: anchor === undefined ? route : `${route}#${anchor}`, external: false }
}

function assetUrl(src: string, options: RenderOptions): string | undefined {
	if (/^https?:/i.test(src)) return src
	if (/^[a-z][a-z0-9+.-]*:/i.test(src)) return undefined
	return `${options.base ?? '/'}docs/${resolvePath(options.docPath, src)}`
}

function renderSpans(text: string, options: RenderOptions): string {
	return escapeHtml(text)
		.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (_match: string, alt: string, src: string) => {
			const url = assetUrl(src, options)
			return url === undefined ? alt : `<img src="${url}" alt="${alt}" loading="lazy">`
		})
		.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_match: string, label: string, href: string) => {
			const link = resolveHref(href, options)
			if (link === undefined) return label
			const target = link.external ? ' target="_blank" rel="noopener noreferrer"' : ''
			return `<a href="${link.href}"${target}>${label}</a>`
		})
		.replace(/&lt;(https?:\/\/[^\s&]+)&gt;/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>')
		.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
		.replace(/~~([^~]+)~~/g, '<del>$1</del>')
		.replace(/(?<![*\w])\*([^*\s][^*]*)\*(?!\*)/g, '<em>$1</em>')
}

function renderInline(text: string, options: RenderOptions): string {
	return text.split(/(`[^`]+`)/g).map((part, index) => (index % 2 === 1 ? `<code>${escapeHtml(part.slice(1, -1))}</code>` : renderSpans(part, options))).join('')
}

function renderCode(code: string, language: string): string {
	const label = escapeHtml(language === '' ? 'text' : language)
	return (`<figure class="code"><figcaption><span>${label}</span><button type="button" class="copy" aria-label="Copy code">Copy</button></figcaption><pre><code class="language-${label}">${highlight(code, language)}</code></pre></figure>`)
}

function splitRow(line: string): string[] {
	return line.trim().replace(/^\|/, '').replace(/\|$/, '').split(/(?<!\\)\|/).map((cell) => cell.trim().replace(/\\\|/g, '|'))
}

function alignClass(cell: string): string {
	const value = cell.trim()
	const left = value.startsWith(':')
	const right = value.endsWith(':')
	if (left && right) return ' class="a-center"'
	return right ? ' class="a-right"' : ''
}

function renderTable(header: string[], dividers: string[], rows: string[][], state: State): string {
	const aligns = dividers.map(alignClass)
	const cell = (tag: string, value: string, index: number): string => `<${tag}${aligns[index] ?? ''}>${renderInline(value, state.options)}</${tag}>`
	const head = header.map((value, index) => cell('th', value, index)).join('')
	const body = rows.map((row) => `<tr>${row.map((value, index) => cell('td', value, index)).join('')}</tr>`).join('')
	return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`
}

function renderItem(item: ListItem, state: State): string {
	const lines = [...item.lines]
	const task = /^\[([ xX])\]\s+/.exec(lines[0] ?? '')
	if (task !== null) lines[0] = (lines[0] ?? '').slice(task[0].length)
	const body = renderBlocks(lines, state)
	const single = /^<p>([\s\S]*)<\/p>$/.exec(body)
	const content = single !== null && !(single[1] ?? '').includes('<p>') ? (single[1] ?? '') : body
	const box = task === null ? '' : `<input type="checkbox" disabled${task[1] === ' ' ? '' : ' checked'}> `
	return `<li${task === null ? '' : ' class="task"'}>${box}${content}</li>`
}

function renderList(lines: readonly string[], start: number, state: State): { html: string; next: number } {
	const first = LIST_ITEM.exec(lines[start] ?? '')
	const base = (first?.[1] ?? '').length
	const ordered = /^\d/.test(first?.[2] ?? '')
	const items: ListItem[] = []
	let i = start
	while (i < lines.length) {
		const line = lines[i] ?? ''
		const match = LIST_ITEM.exec(line)
		const sibling =
			match !== null &&
			(match[1] ?? '').length === base &&
			/^\d/.test(match[2] ?? '') === ordered &&
			!RULE.test(line)
		if (match !== null && sibling) {
			items.push({ lines: [match[3] ?? ''], indent: base + (match[2] ?? '').length + 1 })
			i += 1
			continue
		}
		const current = items[items.length - 1]
		if (current === undefined) break
		if (line.trim() === '') {
			const upcoming = lines[i + 1] ?? ''
			const indented = leadingSpaces(upcoming) > base
			const nextSibling = LIST_ITEM.test(upcoming) && leadingSpaces(upcoming) === base
			if (upcoming.trim() === '' || (!indented && !nextSibling)) break
			current.lines.push('')
			i += 1
			continue
		}
		if (leadingSpaces(line) <= base) break
		current.lines.push(line.slice(Math.min(leadingSpaces(line), current.indent)))
		i += 1
	}
	const tag = ordered ? 'ol' : 'ul'
	const number = ordered ? Number.parseInt(first?.[2] ?? '1', 10) : 1
	const attribute = ordered && number !== 1 ? ` start="${number}"` : ''
	const body = items.map((item) => renderItem(item, state)).join('')
	return { html: `<${tag}${attribute}>${body}</${tag}>`, next: i }
}

function startsBlock(lines: readonly string[], index: number): boolean {
	const line = lines[index] ?? ''
	return (FENCE.test(line) || HEADING.test(line) || RULE.test(line) || QUOTE.test(line) || LIST_ITEM.test(line) || (line.includes('|') && TABLE_DIVIDER.test(lines[index + 1] ?? '')))
}

function renderBlocks(lines: readonly string[], state: State): string {
	const out: string[] = []
	let i = 0
	while (i < lines.length) {
		const line = lines[i] ?? ''
		if (line.trim() === '') {
			i += 1
			continue
		}
		const fence = FENCE.exec(line)
		if (fence !== null) {
			const marker = fence[1] ?? '```'
			const code: string[] = []
			i += 1
			while (i < lines.length) {
				const closing = (lines[i] ?? '').trim()
				if (closing.startsWith(marker) && /^[`~]+$/.test(closing)) break
				code.push(lines[i] ?? '')
				i += 1
			}
			i += 1
			out.push(renderCode(code.join('\n'), fence[2] ?? ''))
			continue
		}
		const heading = HEADING.exec(line)
		if (heading !== null) {
			const level = (heading[1] ?? '#').length
			const raw = heading[2] ?? ''
			const plain = raw.replace(/\[([^\]]*)\]\([^)]*\)/g, '$1').replace(/[`*~]/g, '')
			const id = uniqueId(slugify(plain), state.used)
			state.headings.push({ level, id, text: plain })
			const anchor = `<a class="anchor" href="#${id}" aria-label="Link to this section">#</a>`
			out.push(`<h${level} id="${id}">${renderInline(raw, state.options)}${anchor}</h${level}>`)
			i += 1
			continue
		}
		if (RULE.test(line)) {
			out.push('<hr>')
			i += 1
			continue
		}
		if (QUOTE.test(line)) {
			const inner: string[] = []
			while (i < lines.length && QUOTE.test(lines[i] ?? '')) {
				inner.push((lines[i] ?? '').replace(/^ {0,3}> ?/, ''))
				i += 1
			}
			const alert = inner[0] !== undefined ? ALERT.exec(inner[0]) : null
			if (alert !== null) {
				const kind = alert[1]?.toLowerCase()
				if (kind === undefined) {
					out.push(`<blockquote>${renderBlocks(inner, state)}</blockquote>`)
				} else {
					const title = `${kind.charAt(0).toUpperCase()}${kind.slice(1)}`
					out.push(`<aside class="alert ${kind}"><p class="alert-title">${title}</p>${renderBlocks(inner.slice(1), state)}</aside>`)
				}
			} else {
				out.push(`<blockquote>${renderBlocks(inner, state)}</blockquote>`)
			}
			continue
		}
		if (line.includes('|') && TABLE_DIVIDER.test(lines[i + 1] ?? '')) {
			const header = splitRow(line)
			const dividers = splitRow(lines[i + 1] ?? '')
			const rows: string[][] = []
			i += 2
			while (i < lines.length && (lines[i] ?? '').trim() !== '' && (lines[i] ?? '').includes('|')) {
				rows.push(splitRow(lines[i] ?? ''))
				i += 1
			}
			out.push(renderTable(header, dividers, rows, state))
			continue
		}
		if (LIST_ITEM.test(line)) {
			const list = renderList(lines, i, state)
			out.push(list.html)
			i = list.next
			continue
		}
		const buffer = [line]
		i += 1
		while (i < lines.length) {
			const next = lines[i] ?? ''
			if (next.trim() === '' || startsBlock(lines, i)) break
			buffer.push(next)
			i += 1
		}
		out.push(`<p>${renderInline(buffer.join('\n'), state.options).replace(/ {2,}\n/g, '<br>\n')}</p>`)
	}
	return out.join('\n')
}

/**
 * Renders Markdown to HTML. Raw HTML in the source is always escaped.
 * Relative `.md` links become hash routes; `javascript:` and `data:` URLs are dropped.
 */
export function renderMarkdown(source: string, options: RenderOptions): Rendered {
	const { body } = parseFrontMatter(source)
	const state: State = { options, headings: [], used: new Map() }
	const html = renderBlocks(body.split('\n'), state)
	const title = state.headings.find((heading) => heading.level === 1)?.text ?? ''
	return { title, html, headings: state.headings }
}
