export type Bump = 'major' | 'minor' | 'patch'

const HEADING = /^## \[([^\]]+)\](?: - (\d{4}-\d{2}-\d{2}))?[ \t]*$/gm

/**
 * Returns the trimmed body of the changelog section for `version`,
 * or `undefined` when the section is missing or empty.
 */
export function extractSection(changelog: string, version: string): string | undefined {
	const headings = [...changelog.matchAll(HEADING)]
	const index = headings.findIndex((match) => match[1] === version)
	const current = headings[index]
	if (current === undefined) return undefined
	const start = (current.index ?? 0) + current[0].length
	const end = headings[index + 1]?.index ?? changelog.length
	const body = changelog.slice(start, end).trim()
	return body.length > 0 ? body : undefined
}

/**
 * Moves the current Unreleased entries under a new dated version heading.
 */
export function promoteUnreleased(changelog: string, version: string, date: string): string {
	const marker = '## [Unreleased]'
	if (!changelog.includes(marker)) throw new Error('CHANGELOG.md has no [Unreleased] section')
	return changelog.replace(marker, `${marker}\n\n## [${version}] - ${date}`)
}

/**
 * Computes the next semantic version for the requested bump.
 */
export function bumpVersion(version: string, bump: Bump): string {
	const match = /^(\d+)\.(\d+)\.(\d+)$/.exec(version)
	if (match === null) throw new Error(`Unsupported version "${version}"`)
	const major = Number(match[1])
	const minor = Number(match[2])
	const patch = Number(match[3])
	switch (bump) {
		case 'major': return `${major + 1}.0.0`
		case 'minor': return `${major}.${minor + 1}.0`
		case 'patch': return `${major}.${minor}.${patch + 1}`
	}
}
