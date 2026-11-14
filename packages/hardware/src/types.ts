export interface CpuInfo {
	readonly model: string
	readonly architecture: string
	readonly logicalCores: number
	readonly speedMhz: number
}

export interface MemoryInfo {
	readonly totalBytes: number
	readonly availableBytes: number
}

export interface DiskInfo {
	readonly path: string
	readonly totalBytes: number
	readonly freeBytes: number
	readonly writeMegabytesPerSecond?: number
}

export interface GpuInfo {
	readonly name: string
	readonly vramTotalBytes: number
	readonly vramFreeBytes: number
	readonly driverVersion: string
	readonly computeCapability?: string
}

export interface GpuReport {
	readonly source: 'nvidia-smi' | 'unavailable'
	readonly devices: readonly GpuInfo[]
	readonly cudaDriverVersion?: string
	readonly reason?: string
}

export interface RuntimeInfo {
	readonly bunVersion: string
	readonly platform: string
	readonly osName: string
	readonly osRelease: string
}

export interface HardwareProfile {
	readonly cpu: CpuInfo
	readonly memory: MemoryInfo
	readonly disks: readonly DiskInfo[]
	readonly gpu: GpuReport
	readonly runtime: RuntimeInfo
	readonly detectedAt: string
}
