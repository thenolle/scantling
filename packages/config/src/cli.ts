import { ConfigError, loadConfig } from './index.ts'

try {
	const loaded = await loadConfig()
	console.log(`File         ${loaded.file ?? 'none (defaults only)'}`)
	console.log(`Environment  ${loaded.envOverrides.length === 0 ? 'no overrides' : loaded.envOverrides.join(', ')}`)
	console.log(JSON.stringify(loaded.config, null, 2))
} catch (error) {
	if (!(error instanceof ConfigError)) throw error
	console.error(error.message)
	process.exit(1)
}
