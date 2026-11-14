import { escapeHtml } from './escape.ts'

interface Spec {
	readonly comments: readonly string[]
	readonly keywords: ReadonlySet<string>
	readonly foldCase?: boolean
}

const STRING = /"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`/
const NUMBER = /\b0x[\da-fA-F]+\b|\b\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?\b/
const WORD = /[A-Za-z_$][\w$]*/
const SLASH_LINE = /\/\/[^\n]*/.source
const BLOCK = /\/\*[\s\S]*?\*\//.source
const HASH_LINE = /#[^\n]*/.source
const DASH_LINE = /--[^\n]*/.source

const words = (list: string): ReadonlySet<string> => new Set(list.split(' '))

const SCRIPT: Spec = {
	comments: [SLASH_LINE, BLOCK],
	keywords: words('abstract as async await break case catch class const continue declare default delete do else enum export extends false final finally fn for from func function go if impl implements import in instanceof interface let match mod mut namespace new nil null of override package private protected pub public readonly return satisfies self static struct super switch this throw true try type typeof undefined use val var void where while yield')
}
const SHELL: Spec = {
	comments: [HASH_LINE],
	keywords: words('if then else elif fi for in do done while until case esac function return export local set unset echo cd source exit')
}
const PYTHON: Spec = {
	comments: [HASH_LINE],
	keywords: words('and as assert async await break class continue def del elif else except False finally for from global if import in is lambda None nonlocal not or pass raise return True try while with yield')
}
const SQL: Spec = {
	comments: [DASH_LINE, BLOCK],
	foldCase: true,
	keywords: words('select from where insert into values update set delete create table drop alter index join left right inner outer on group by order having limit offset and or not null as primary key foreign references unique default distinct union all case when then else end')
}
const JSON_SPEC: Spec = { comments: [], keywords: words('true false null') }
const YAML: Spec = { comments: [HASH_LINE], keywords: words('true false null yes no') }
const CSS: Spec = { comments: [BLOCK], keywords: words('') }

const LANGUAGES = new Map<string, Spec>()
const register = (names: readonly string[], spec: Spec): void => { for (const name of names) LANGUAGES.set(name, spec) }
register(['ts', 'tsx', 'typescript', 'js', 'jsx', 'mjs', 'javascript', 'java', 'c', 'cpp', 'cs', 'csharp', 'go', 'rust', 'rs', 'kotlin', 'kt', 'swift', 'php'], SCRIPT)
register(['sh', 'bash', 'shell', 'zsh', 'powershell', 'ps1', 'console'], SHELL)
register(['py', 'python', 'rb', 'ruby'], PYTHON)
register(['sql'], SQL)
register(['json', 'jsonc'], JSON_SPEC)
register(['yaml', 'yml', 'toml', 'ini'], YAML)
register(['css'], CSS)

const PATTERNS = new Map<Spec, RegExp>()

function patternFor(spec: Spec): RegExp {
	let pattern = PATTERNS.get(spec)
	if (pattern === undefined) {
		const comments = spec.comments.length > 0 ? spec.comments.join('|') : '(?!)'
		pattern = new RegExp(`(${comments})|(${STRING.source})|(${NUMBER.source})|(${WORD.source})`, 'g')
		PATTERNS.set(spec, pattern)
	}
	return pattern
}

function span(kind: string, text: string): string {
	return `<span class="tok-${kind}">${escapeHtml(text)}</span>`
}

function classify(match: RegExpMatchArray, spec: Spec, following: string): string {
	const text = match[0]
	if (match[1] !== undefined) return span('c', text)
	if (match[2] !== undefined) return span('s', text)
	if (match[3] !== undefined) return span('n', text)
	if (spec.keywords.has(spec.foldCase === true ? text.toLowerCase() : text)) return span('k', text)
	if (following === '(') return span('f', text)
	if (/^[A-Z]/.test(text)) return span('t', text)
	return escapeHtml(text)
}

/**
 * Highlights source code for a known language and returns HTML.
 * Unknown languages are returned escaped and unstyled.
 */
export function highlight(code: string, language: string): string {
	const spec = LANGUAGES.get(language.toLowerCase())
	if (spec === undefined) return escapeHtml(code)
	let html = ''
	let cursor = 0
	for (const match of code.matchAll(patternFor(spec))) {
		const index = match.index ?? 0
		html += escapeHtml(code.slice(cursor, index))
		cursor = index + match[0].length
		html += classify(match, spec, code.charAt(cursor))
	}
	return html + escapeHtml(code.slice(cursor))
}
