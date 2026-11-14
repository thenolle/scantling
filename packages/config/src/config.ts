import type { Infer, Issue } from './schema.ts'
import { bool, choice, number, object, text, withDefault } from './schema.ts'

export const LOG_LEVELS = ['debug', 'info', 'warn', 'error'] as const

/**
 * Every setting with its default. Sizes that users edit are in GiB or MiB;
 * the loader does not convert them, consumers do.
 */
export const configSchema = object({
	server: object({
		host: withDefault(text({ minLength: 1 }), '127.0.0.1'),
		port: withDefault(number({ integer: true, min: 1, max: 65535 }), 4100),
		allowRemote: withDefault(bool(), false)
	}),
	paths: object({
		dataDir: withDefault(text({ minLength: 1 }), 'data'),
		logDir: withDefault(text({ minLength: 1 }), 'data/logs'),
		datasetDir: withDefault(text({ minLength: 1 }), 'data/datasets'),
		checkpointDir: withDefault(text({ minLength: 1 }), 'data/checkpoints')
	}),
	resources: object({
		memoryFraction: withDefault(number({ min: 0.05, max: 1 }), 0.6),
		reserveGiB: withDefault(number({ min: 0 }), 3),
		maxWorkers: withDefault(number({ integer: true, min: 1, max: 64 }), 8),
		reservedCores: withDefault(number({ integer: true, min: 0, max: 64 }), 2),
		minFreeDiskGiB: withDefault(number({ min: 0 }), 10)
	}),
	privacy: object({
		memoryEnabled: withDefault(bool(), true),
		trainingAllowed: withDefault(bool(), false),
		webSearchEnabled: withDefault(bool(), false)
	}),
	logging: object({
		level: withDefault(choice(LOG_LEVELS), 'info'),
		toFile: withDefault(bool(), true),
		maxFileMiB: withDefault(number({ min: 1 }), 10),
		maxFiles: withDefault(number({ integer: true, min: 1, max: 100 }), 5)
	})
})

export type ScantlingConfig = Infer<typeof configSchema>

const LOOPBACK = new Set(['localhost', '::1'])

/**
 * Cross-field rules that a per-field schema cannot express.
 */
export function validateConfig(config: ScantlingConfig): Issue[] {
	const issues: Issue[] = []
	const host = config.server.host
	const loopback = LOOPBACK.has(host) || host.startsWith('127.')
	if (!loopback && !config.server.allowRemote) issues.push({ path: 'server.host', message: `"${host}" is not a loopback address; set server.allowRemote to true to expose the server` })
	return issues
}
