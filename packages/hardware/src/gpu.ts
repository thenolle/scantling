import { MIB } from './format.ts'
import { runDiagnostic } from './exec.ts'
import type { GpuInfo, GpuReport } from './types.ts'

const FULL_QUERY = 'name,memory.total,memory.free,driver_version,compute_cap'
const BASIC_QUERY = 'name,memory.total,memory.free,driver_version'

/**
 * Parses `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` output.
 * Memory values are expected in MiB. Rows that cannot be parsed are skipped.
 */
export function parseNvidiaSmiCsv(output: string, withComputeCapability: boolean): GpuInfo[] {
	return output.split('\n').map((line) => line.trim()).filter((line) => line.length > 0).flatMap((line): GpuInfo[] => {
		const [name, total, free, driver, compute] = line.split(',').map((field) => field.trim())
		const totalMib = Number(total)
		const freeMib = Number(free)
		if (name === undefined || driver === undefined || !Number.isFinite(totalMib) || !Number.isFinite(freeMib)) return []
		return [{
			name,
			vramTotalBytes: totalMib * MIB,
			vramFreeBytes: freeMib * MIB,
			driverVersion: driver,
			...(withComputeCapability && compute !== undefined && compute !== '' ? { computeCapability: compute } : {})
		}]
	})
}

/**
 * Extracts the highest CUDA version the driver supports from plain `nvidia-smi` output.
 */
export function parseCudaVersion(output: string): string | undefined {
	return /CUDA Version:\s*([\d.]+)/.exec(output)?.[1]
}

async function query(fields: string): Promise<string | undefined> {
	const result = await runDiagnostic(['nvidia-smi', `--query-gpu=${fields}`, '--format=csv,noheader,nounits'])
	return result?.exitCode === 0 ? result.stdout : undefined
}

/**
 * Detects NVIDIA GPUs through `nvidia-smi`. Detection only: it does not mean the GPU
 * can be used for computation.
 */
export async function detectGpu(): Promise<GpuReport> {
	const full = await query(FULL_QUERY)
	const basic = full === undefined ? await query(BASIC_QUERY) : undefined
	const output = full ?? basic
	if (output === undefined) return { source: 'unavailable', devices: [], reason: 'nvidia-smi was not found or did not report any device' }
	const devices = parseNvidiaSmiCsv(output, full !== undefined)
	const summary = await runDiagnostic(['nvidia-smi'])
	const cuda = summary?.exitCode === 0 ? parseCudaVersion(summary.stdout) : undefined
	return { source: 'nvidia-smi', devices, ...(cuda === undefined ? {} : { cudaDriverVersion: cuda }) }
}
