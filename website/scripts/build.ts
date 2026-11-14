import { cp, mkdir, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { flatten } from '../src/nav.ts'
import { bundleApp, docsRoot, siteRoot } from './bundle.ts'
import { buildNav } from './docs.ts'

const dist = join(siteRoot, 'dist')

await rm(dist, { recursive: true, force: true })
await mkdir(dist, { recursive: true })
await cp(join(siteRoot, 'public'), dist, { recursive: true })
await cp(docsRoot, join(dist, 'docs'), { recursive: true })
await bundleApp(dist, true)

const nav = await buildNav(docsRoot)
await writeFile(join(dist, 'docs', 'nav.json'), JSON.stringify(nav), 'utf8')
await cp(join(dist, 'index.html'), join(dist, '404.html'))
await writeFile(join(dist, '.nojekyll'), '', 'utf8')

console.log(`Documentation built in ${dist} with ${flatten(nav).length} pages`)
