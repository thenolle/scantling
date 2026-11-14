export interface CommandResult {
	readonly exitCode: number
	readonly stdout: string
}

/**
 * Runs a fixed diagnostic command with a hard timeout.
 * Only built-in commands with constant arguments may be passed; this is not an agent tool.
 * Returns `undefined` when the binary is missing or cannot be started.
 */
export async function runDiagnostic(command: readonly string[], timeoutMs = 4000): Promise<CommandResult | undefined> {
	try {
		const subprocess = Bun.spawn([...command], { stdin: 'ignore', stdout: 'pipe', stderr: 'ignore' })
		const timer = setTimeout(() => subprocess.kill(), timeoutMs)
		try {
			const stdout = await new Response(subprocess.stdout).text()
			return { exitCode: await subprocess.exited, stdout }
		} finally {
			clearTimeout(timer)
		}
	} catch {
		return undefined
	}
}
