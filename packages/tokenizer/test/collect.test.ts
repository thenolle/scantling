import { expect, test } from 'bun:test'
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { collectTextFiles } from '../src/files.ts'

test('collection keeps its order at any concurrency and never reads rejected files', async () => {
	const root = mkdtempSync(join(tmpdir(), 'collect-'))
	mkdirSync(join(root, 'sub'))
	for (let index = 0; index < 40; index += 1) writeFileSync(join(root, `file-${String(index).padStart(2, '0')}.${index % 2 === 0 ? 'ts' : 'map'}`), `text ${index}`)
	writeFileSync(join(root, 'sub', 'inner.ts'), 'inner')
	const list = async (concurrency: number, accept?: (path: string) => boolean): Promise<string[]> => {
		const paths: string[] = []
		for await (const file of collectTextFiles([root], undefined, accept, concurrency)) paths.push(file.path)
		return paths
	}
	const serial = await list(1)
	expect(await list(16)).toEqual(serial)
	const rejected: string[] = []
	const onlyTs = await list(16, (path) => { if (!path.endsWith('.ts')) rejected.push(path); return path.endsWith('.ts') })
	expect(onlyTs.every((path) => path.endsWith('.ts'))).toBe(true)
	expect(onlyTs.length).toBe(21)
	expect(rejected.length).toBe(20)
	rmSync(root, { recursive: true, force: true })
})
