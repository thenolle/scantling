import { resolve } from 'node:path'
import { extractSection } from './lib/changelog.ts'

const version = Bun.argv[2]
if (version === undefined) {
	console.error('Usage: bun tools/release-notes.ts <version>')
	process.exit(1)
}

const changelog = await Bun.file(resolve(import.meta.dir, '..', 'CHANGELOG.md')).text()
const notes = extractSection(changelog, version)
if (notes === undefined) {
	console.error(`CHANGELOG.md has no entries for version ${version}`)
	process.exit(1)
}

console.log(notes)
