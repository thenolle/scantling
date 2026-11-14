import { resolve } from 'node:path'
import type { Bump } from './lib/changelog.ts'
import { bumpVersion, extractSection, promoteUnreleased } from './lib/changelog.ts'

const root = resolve(import.meta.dir, '..')
const manifestPath = resolve(root, 'package.json')
const changelogPath = resolve(root, 'CHANGELOG.md')

function parseBump(value: string | undefined): Bump {
	if (value === 'major' || value === 'minor' || value === 'patch') return value
	throw new Error('Usage: bun run release <major|minor|patch>')
}

async function main(): Promise<void> {
	const bump = parseBump(Bun.argv[2])
	const manifest = await Bun.file(manifestPath).text()
	const current = /"version":\s*"([^"]+)"/.exec(manifest)?.[1]
	if (current === undefined) throw new Error('package.json has no version field')
	const changelog = await Bun.file(changelogPath).text()
	if (extractSection(changelog, 'Unreleased') === undefined) throw new Error('The Unreleased section of CHANGELOG.md is empty. Document the changes first.')
	const next = bumpVersion(current, bump)
	const date = new Date().toISOString().slice(0, 10)
	await Bun.write(manifestPath, manifest.replace(/("version":\s*")[^"]+(")/, `$1${next}$2`))
	await Bun.write(changelogPath, promoteUnreleased(changelog, next, date))
	console.log(`Version ${current} -> ${next}`)
	console.log('Next:')
	console.log('  git add package.json CHANGELOG.md')
	console.log(`  git commit -m "chore(release): v${next}"`)
	console.log('  git push origin master')
}

await main()
