import { join } from 'node:path'

export const siteRoot = join(import.meta.dir, '..')
export const docsRoot = join(siteRoot, '..', 'docs')

/**
 * Bundles the browser application into `outdir`.
 */
export async function bundleApp(outdir: string, minify: boolean): Promise<void> {
	const result = await Bun.build({
		entrypoints: [join(siteRoot, 'src', 'app.ts')],
		outdir,
		target: 'browser',
		format: 'esm',
		naming: '[name].[ext]',
		minify,
		sourcemap: minify ? 'none' : 'inline'
	})
	if (!result.success) throw new Error(result.logs.map((log) => log.message).join('\n'))
}
