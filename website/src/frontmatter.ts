export interface FrontMatter {
	readonly meta: Readonly<Record<string, string>>
	readonly body: string
}

/**
 * Splits optional `---` delimited `key: value` front matter from a Markdown source.
 * Line endings are normalised to `\n`.
 */
export function parseFrontMatter(source: string): FrontMatter {
	const text = source.replace(/\r\n/g, '\n')
	if (!text.startsWith('---\n')) return { meta: {}, body: text }
	const end = text.indexOf('\n---', 4)
	if (end === -1) return { meta: {}, body: text }
	const meta: Record<string, string> = {}
	for (const line of text.slice(4, end).split('\n')) {
		const match = /^([A-Za-z][\w-]*):\s*(.*)$/.exec(line)
		if (match?.[1] === undefined) continue
		meta[match[1]] = (match[2] ?? '').trim().replace(/^(['"])(.*)\1$/, '$2')
	}
	const rest = text.slice(end + 4)
	return { meta, body: rest.startsWith('\n') ? rest.slice(1) : rest }
}
