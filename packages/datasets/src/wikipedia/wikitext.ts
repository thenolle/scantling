const SKIP_SECTIONS = new Set(['references', 'external links', 'see also', 'further reading', 'notes', 'bibliography', 'sources', 'footnotes', 'citations', 'gallery', 'footnotes and references'])
const ENTITIES: Readonly<Record<string, string>> = { lt: '<', gt: '>', amp: '&', quot: '"', apos: "'", nbsp: ' ', ndash: '-', mdash: '-' }
const MIN_CHARACTERS = 600

export function decodeEntities(text: string): string {
	return text.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (match: string, body: string) => {
		if (body.startsWith('#')) {
			const code = body[1] === 'x' || body[1] === 'X' ? Number.parseInt(body.slice(2), 16) : Number.parseInt(body.slice(1), 10)
			return Number.isFinite(code) && code > 31 && code < 0x110000 ? String.fromCodePoint(code) : ' '
		}
		return ENTITIES[body.toLowerCase()] ?? match
	})
}

function removeNested(text: string, open: string, close: string): string {
	let depth = 0
	let out = ''
	let index = 0
	while (index < text.length) {
		if (text.startsWith(open, index)) {
			depth += 1
			index += open.length
		} else if (depth > 0 && text.startsWith(close, index)) {
			depth -= 1
			index += close.length
		} else {
			if (depth === 0) out += text.charAt(index)
			index += 1
		}
	}
	return out
}

function removeTables(text: string): string {
	const out: string[] = []
	let depth = 0
	for (const line of text.split('\n')) {
		const trimmed = line.trimStart()
		if (trimmed.startsWith('{|')) depth += 1
		else if (depth > 0) {
			if (trimmed.startsWith('|}')) depth -= 1
		} else out.push(line)
	}
	return out.join('\n')
}

function removeMediaLinks(text: string): string {
	const media = /\[\[(?:file|image|media|category):/iy
	let out = ''
	let index = 0
	while (index < text.length) {
		media.lastIndex = index
		if (text.startsWith('[[', index) && media.test(text)) {
			let depth = 0
			let end = index
			while (end < text.length) {
				if (text.startsWith('[[', end)) {
					depth += 1
					end += 2
				} else if (text.startsWith(']]', end)) {
					depth -= 1
					end += 2
					if (depth === 0) break
				} else end += 1
			}
			index = end
		} else {
			out += text.charAt(index)
			index += 1
		}
	}
	return out
}

/** Plain text of an article, or undefined when it is a redirect, a stub, a list or a disambiguation page. */
export function cleanWikitext(raw: string, title: string): string | undefined {
	let text = decodeEntities(raw)
	text = text.replace(/<!--[\s\S]*?-->/g, '')
	text = text.replace(/<ref\b[^>]*\/>/gi, '').replace(/<ref\b[^>]*>[\s\S]*?<\/ref>/gi, '')
	text = text.replace(/<(gallery|math|timeline|score|syntaxhighlight|source|code|pre|imagemap|mapframe|noinclude|templatedata)\b[^>]*>[\s\S]*?<\/\1>/gi, '')
	if (/^#redirect/i.test(text.trimStart())) return undefined
	text = removeNested(text, '{{', '}}')
	text = removeTables(text)
	text = removeMediaLinks(text)
	text = text.replace(/\[\[([^[\]|]*)\|([^[\]]*)\]\]/g, '$2').replace(/\[\[([^[\]|]*)\]\]/g, '$1')
	text = text.replace(/\[(?:https?:)?\/\/[^\s\]]+\s+([^\]]*)\]/g, '$1').replace(/\[(?:https?:)?\/\/[^\s\]]+\]/g, '')
	text = text.replace(/'{2,5}/g, '')
	text = text.replace(/<br\s*\/?>/gi, '\n').replace(/<\/?[a-z][^>]*>/gi, '')
	text = text.replace(/__[A-Z]+__/g, '')
	const out: string[] = []
	let skipping = false
	for (const rawLine of text.split('\n')) {
		let line = rawLine.replace(/\s+$/g, '')
		const heading = /^(=+)\s*(.*?)\s*\1$/.exec(line)
		if (heading !== null) {
			const name = (heading[2] ?? '').trim()
			skipping = SKIP_SECTIONS.has(name.toLowerCase())
			if (!skipping && name !== '') out.push('', name, '')
			continue
		}
		if (skipping) continue
		line = line.replace(/^[*#:;]+\s*/, '- ').replace(/^- $/, '')
		if (/^\s*(\||!|\{\||\|\})/.test(line)) continue
		out.push(line)
	}
	let result = decodeEntities(out.join('\n')).replace(/[\u0000-\u0008\u000b-\u001f\u007f]/g, ' ')
	result = result.replace(/[ \t]+/g, ' ').replace(/ ?\n ?/g, '\n').replace(/\n{3,}/g, '\n\n').trim()
	if (result.length < MIN_CHARACTERS) return undefined
	if (/(may|can) (also )?refer to[:.]?\s*$/im.test(result.slice(0, 600)) || /^list of /i.test(title)) return undefined
	return `${title}\n\n${result}`
}

/** The articles of one decoded dump chunk: main-namespace pages that are not redirects. */
export function articlesOf(xml: string): string[] {
	const articles: string[] = []
	for (const match of xml.matchAll(/<page>([\s\S]*?)<\/page>/g)) {
		const page = match[1] ?? ''
		if (!/<ns>0<\/ns>/.test(page) || /<redirect\b/.test(page)) continue
		const title = /<title>([\s\S]*?)<\/title>/.exec(page)?.[1]
		const body = /<text\b[^>]*>([\s\S]*?)<\/text>/.exec(page)?.[1]
		if (title === undefined || body === undefined) continue
		const cleaned = cleanWikitext(body, decodeEntities(title))
		if (cleaned !== undefined) articles.push(cleaned)
	}
	return articles
}
