export interface LogRecord {
	readonly time: string
	readonly level: 'debug' | 'info' | 'warn' | 'error'
	readonly scope: string
	readonly message: string
	readonly fields: Readonly<Record<string, unknown>>
}

export type RunStatus = 'running' | 'finished' | 'stopped'
export interface RunPlan { readonly startedAt: string; readonly fields: Readonly<Record<string, unknown>> }
export interface StepPoint {
	readonly step: number
	readonly total: number
	readonly time: string
	readonly loss: number
	readonly gradNorm: number
	readonly lr: number
	readonly tokensPerSecond: number
	readonly etaMinutes: number
	readonly skipped: boolean
	readonly gpuMs?: number
	readonly optimizerMs?: number
	readonly transferMs?: number
}
export interface EvalPoint { readonly step: number; readonly time: string; readonly loss: number; readonly perplexity: number; readonly bitsPerByte: number }
export interface CheckpointPoint { readonly time: string; readonly id: string; readonly evalLoss: number; readonly promoted: boolean }
export interface LogLine { readonly time: string; readonly level: LogRecord['level']; readonly message: string; readonly text: string }
export interface DatasetState {
	readonly phase: string
	readonly done: number
	readonly total: number
	readonly cached: number
	readonly startedAt: string
	readonly updatedAt: string
	readonly finished: boolean
	readonly summary?: Readonly<Record<string, unknown>>
}
export interface PastRun { readonly startedAt: string; readonly status: RunStatus; readonly lastStep: number; readonly total: number }

export type Delta =
	| { readonly type: 'log'; readonly line: LogLine }
	| { readonly type: 'plan'; readonly plan: RunPlan }
	| { readonly type: 'step'; readonly point: StepPoint }
	| { readonly type: 'eval'; readonly point: EvalPoint }
	| { readonly type: 'checkpoint'; readonly point: CheckpointPoint }
	| { readonly type: 'status'; readonly status: RunStatus; readonly summary: Readonly<Record<string, unknown>> }
	| { readonly type: 'dataset'; readonly dataset: DatasetState }

export interface Snapshot {
	readonly status: RunStatus | 'idle'
	readonly plan: RunPlan | undefined
	readonly steps: readonly StepPoint[]
	readonly evals: readonly EvalPoint[]
	readonly checkpoints: readonly CheckpointPoint[]
	readonly lines: readonly LogLine[]
	readonly summary: Readonly<Record<string, unknown>> | undefined
	readonly dataset: DatasetState | undefined
	readonly past: readonly PastRun[]
	readonly lastEntryAt: number
}
