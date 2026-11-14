export interface NavPage {
	readonly path: string
	readonly title: string
	readonly section: string
	readonly description: string
}

export interface NavSection {
	readonly title: string
	readonly pages: readonly NavPage[]
}

export interface DocNav {
	readonly title: string
	readonly description: string
	readonly repo: string
	readonly version: string
	readonly sections: readonly NavSection[]
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function field(record: Record<string, unknown>, key: string): string {
	const value = record[key]
	if (typeof value !== 'string') throw new Error(`nav.json: "${key}" must be a string`)
	return value
}

function parsePage(value: unknown): NavPage {
	if (!isRecord(value)) throw new Error('nav.json: invalid page')
	return {
		path: field(value, 'path'),
		title: field(value, 'title'),
		section: field(value, 'section'),
		description: field(value, 'description')
	}
}

function parseSection(value: unknown): NavSection {
	if (!isRecord(value) || !Array.isArray(value['pages'])) throw new Error('nav.json: invalid section')
	return { title: field(value, 'title'), pages: value['pages'].map(parsePage) }
}

/**
 * Validates the parsed contents of `nav.json`.
 */
export function parseNav(value: unknown): DocNav {
	if (!isRecord(value) || !Array.isArray(value['sections'])) throw new Error('nav.json is malformed')
	return {
		title: field(value, 'title'),
		description: field(value, 'description'),
		repo: field(value, 'repo'),
		version: field(value, 'version'),
		sections: value['sections'].map(parseSection)
	}
}

/**
 * Returns every page in reading order.
 */
export function flatten(nav: DocNav): NavPage[] {
	return nav.sections.flatMap((section) => [...section.pages])
}
