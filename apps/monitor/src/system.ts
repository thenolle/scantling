import { cpus, freemem, totalmem } from 'node:os'

export interface GpuSample {
	readonly name: string
	readonly utilization: number
	readonly memoryUsedMiB: number
	readonly memoryTotalMiB: number
	readonly temperatureC: number
	readonly powerW?: number
	readonly clockMHz?: number
}

export interface SystemSample {
	readonly time: number
	readonly cpuPercent: number
	readonly memoryUsedBytes: number
	readonly memoryTotalBytes: number
	readonly gpu?: GpuSample
}

const QUERY = 'name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm'

export function parseGpuCsv(output: string): GpuSample | undefined {
	const line = output.split('\n').map((row) => row.trim()).find((row) => row.length > 0)
	if (line === undefined) return undefined
	const [name, utilization, used, total, temperature, power, clock] = line.split(',').map((field) => field.trim())
	const number = (value: string | undefined): number | undefined => {
		const parsed = Number(value)
		return value === undefined || value === '' || !Number.isFinite(parsed) ? undefined : parsed
	}
	const [u, m, t, c] = [number(utilization), number(used), number(total), number(temperature)]
	if (name === undefined || u === undefined || m === undefined || t === undefined || c === undefined) return undefined
	const [powerW, clockMHz] = [number(power), number(clock)]
	return { name, utilization: u, memoryUsedMiB: m, memoryTotalMiB: t, temperatureC: c, ...(powerW === undefined ? {} : { powerW }), ...(clockMHz === undefined ? {} : { clockMHz }) }
}

export async function readGpu(): Promise<GpuSample | undefined> {
	try {
		const child = Bun.spawn(['nvidia-smi', `--query-gpu=${QUERY}`, '--format=csv,noheader,nounits'], { stdout: 'pipe', stderr: 'ignore' })
		const output = await new Response(child.stdout).text()
		await child.exited
		return parseGpuCsv(output)
	} catch {
		return undefined
	}
}

const totals = (): { idle: number; total: number } => cpus().reduce((sum, cpu) => ({ idle: sum.idle + cpu.times.idle, total: sum.total + cpu.times.user + cpu.times.nice + cpu.times.sys + cpu.times.idle + cpu.times.irq }), { idle: 0, total: 0 })

export class CpuMeter {
	#last = totals()

	sample(): number {
		const now = totals()
		const idle = now.idle - this.#last.idle
		const total = now.total - this.#last.total
		this.#last = now
		return total <= 0 ? 0 : Math.round((1 - idle / total) * 1000) / 10
	}
}

export async function sampleSystem(meter: CpuMeter): Promise<SystemSample> {
	const gpu = await readGpu()
	return { time: Date.now(), cpuPercent: meter.sample(), memoryUsedBytes: totalmem() - freemem(), memoryTotalBytes: totalmem(), ...(gpu === undefined ? {} : { gpu }) }
}
