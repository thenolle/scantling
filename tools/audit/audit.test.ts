import { afterEach, describe, expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import type { AuditPolicy } from './policy.ts'
import { runAudit } from './run.ts'

const policy: AuditPolicy = {
	allowedDevDependencies: ['typescript'],
	allowedRuntimePatterns: ['@gambrel/*'],
	workspaceScopes: ['@scantling/'],
	forbiddenModelExtensions: ['.safetensors'],
	forbiddenHosts: ['api.' + 'openai.com'],
	sourceRoots: ['packages'],
	sourceExtensions: ['.ts'],
	ignoredDirectories: ['node_modules']
}

const lock = JSON.stringify({ lockfileVersion: 1, packages: { typescript: ['typescript@7.0.2', '', {}, 'x'] } })
const baseline = JSON.stringify({ packages: ['typescript'] })
const manifest = JSON.stringify({ name: 'x', devDependencies: { typescript: '7.0.2' } })
const importing = (specifier: string): string => `import x from ${JSON.stringify(specifier)}\n`

const roots: string[] = []

async function project(extra: Record<string, string> = {}): Promise<string> {
	const root = await mkdtemp(join(tmpdir(), 'scantling-audit-'))
	roots.push(root)
	const files: Record<string, string> = { 'package.json': manifest, 'bun.lock': lock, 'tools/audit.baseline.json': baseline, ...extra }
	for (const [name, content] of Object.entries(files)) {
		const target = join(root, name)
		await mkdir(dirname(target), { recursive: true })
		await Bun.write(target, content)
	}
	return root
}

async function rulesFor(extra: Record<string, string>): Promise<string[]> {
	const findings = await runAudit(await project(extra), policy)
	return findings.map((finding) => finding.rule)
}

afterEach(async () => {
	await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })))
})

describe('dependency audit', () => {
	test('passes a clean project', async () => {
		expect(await rulesFor({ 'packages/a/src/index.ts': importing('node:path') })).toEqual([])
	})

	test('rejects an unlisted dependency', async () => {
		const bad = JSON.stringify({ name: 'x', dependencies: { 'left-pad': '1.0.0' } })
		expect(await rulesFor({ 'packages/a/package.json': bad })).toContain('dependency')
	})

	test('rejects workspace packages without a workspace spec', async () => {
		const bad = JSON.stringify({ name: 'x', dependencies: { '@scantling/tensor': '1.0.0' } })
		expect(await rulesFor({ 'packages/a/package.json': bad })).toContain('spec')
	})

	test('rejects install lifecycle scripts', async () => {
		const bad = JSON.stringify({ name: 'x', scripts: { postinstall: 'echo hi' } })
		expect(await rulesFor({ 'packages/a/package.json': bad })).toContain('lifecycle')
	})

	test('rejects unauthorized imports', async () => {
		expect(await rulesFor({ 'packages/a/src/index.ts': importing('left-pad') })).toContain('import')
	})

	test('allows gambrel and workspace imports', async () => {
		const source = importing('@gambrel/view') + importing('@scantling/tensor')
		expect(await rulesFor({ 'packages/a/src/index.ts': source })).toEqual([])
	})

	test('rejects external AI hosts', async () => {
		const source = `export const url = 'https://${'api.' + 'openai.com'}/v1'\n`
		expect(await rulesFor({ 'packages/a/src/index.ts': source })).toContain('endpoint')
	})

	test('rejects pretrained model files', async () => {
		expect(await rulesFor({ 'data/model.safetensors': 'x' })).toContain('model-file')
	})

	test('rejects lockfile packages missing from the baseline', async () => {
		const extra = { 'tools/audit.baseline.json': JSON.stringify({ packages: [] }) }
		expect(await rulesFor(extra)).toContain('lockfile')
	})

	test('ignores string literals that contain the word import', async () => {
		const source = `export const rule = 'import'\nexport const rules = ['import', 'endpoint']\n`
		expect(await rulesFor({ 'packages/a/src/index.ts': source })).toEqual([])
	})
})
