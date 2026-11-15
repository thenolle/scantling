import type { CheckpointPoint, DatasetState, Delta, EvalPoint, LogLine, LogRecord, PastRun, RunPlan, Snapshot, StepPoint } from './types.ts'

const STEP = /^step (\d+)\/(\d+)$/
const EVAL = /^validation @(\d+)$/
const MAX_LINES = 200
const MAX_POINTS = 3000

const num = (value: unknown): number => (typeof value === 'number' && Number.isFinite(value) ? value : 0)

function thin<T>(points: readonly T[], max: number): T[] {
	if (points.length <= max) return [...points]
	const stride = Math.ceil(points.length / max)
	return points.filter((_, index) => index % stride === 0 || index === points.length - 1)
}

/** Folds log records into the state of the latest run and reports what changed. */
export class RunTracker {
	#plan: RunPlan | undefined
	#status: Snapshot['status'] = 'idle'
	#steps: StepPoint[] = []
	#evals: EvalPoint[] = []
	#checkpoints: CheckpointPoint[] = []
	#lines: LogLine[] = []
	#summary: Record<string, unknown> | undefined
	#dataset: DatasetState | undefined
	#past: PastRun[] = []
	#lastEntryAt = 0

	get lastEntryAt(): number {
		return this.#lastEntryAt
	}

	ingest(record: LogRecord): Delta[] {
		const at = Date.parse(record.time)
		if (Number.isFinite(at)) this.#lastEntryAt = at
		const line: LogLine = { time: record.time, level: record.level, message: record.message, text: Object.keys(record.fields).length === 0 ? '' : JSON.stringify(record.fields) }
		this.#lines.push(line)
		if (this.#lines.length > MAX_LINES) this.#lines.shift()
		const deltas: Delta[] = [{ type: 'log', line }]
		const { message, fields } = record
		if (message === 'Training plan') {
			this.#archive()
			this.#start(record, fields)
			deltas.push({ type: 'plan', plan: this.#plan! })
			return deltas
		}
		const step = STEP.exec(message)
		if (step !== null) {
			this.#ensureRun(record, deltas)
			const point: StepPoint = {
				step: Number(step[1]),
				total: Number(step[2]),
				time: record.time,
				loss: num(fields['loss']),
				gradNorm: num(fields['gradNorm']),
				lr: num(fields['lr']),
				tokensPerSecond: num(fields['tokensPerSecond']),
				etaMinutes: num(fields['etaMinutes']),
				skipped: fields['skipped'] === true,
				...(typeof fields['gpuMs'] === 'number' ? { gpuMs: fields['gpuMs'], optimizerMs: num(fields['optimizerMs']), transferMs: num(fields['transferMs']) } : {})
			}
			this.#steps.push(point)
			deltas.push({ type: 'step', point })
			return deltas
		}
		const evaluation = EVAL.exec(message)
		if (evaluation !== null) {
			this.#ensureRun(record, deltas)
			const point: EvalPoint = { step: Number(evaluation[1]), time: record.time, loss: num(fields['loss']), perplexity: num(fields['perplexity']), bitsPerByte: num(fields['bitsPerByte']) }
			this.#evals.push(point)
			deltas.push({ type: 'eval', point })
		} else if (message === 'Checkpoint') {
			this.#ensureRun(record, deltas)
			const point: CheckpointPoint = { time: record.time, id: String(fields['id'] ?? ''), evalLoss: num(fields['evalLoss']), promoted: fields['promoted'] === true }
			this.#checkpoints.push(point)
			deltas.push({ type: 'checkpoint', point })
		} else if (message === 'Finished' || message === 'Stopped early') {
			this.#status = message === 'Finished' ? 'finished' : 'stopped'
			this.#summary = { ...fields }
			deltas.push({ type: 'status', status: this.#status, summary: this.#summary })
		} else if (message.startsWith('Dataset ')) {
			const dataset = this.#datasetUpdate(record)
			if (dataset !== undefined) deltas.push({ type: 'dataset', dataset })
		}
		return deltas
	}

	snapshot(): Snapshot {
		return {
			status: this.#status,
			plan: this.#plan,
			steps: thin(this.#steps, MAX_POINTS),
			evals: this.#evals,
			checkpoints: this.#checkpoints.slice(-50),
			lines: [...this.#lines],
			summary: this.#summary,
			dataset: this.#dataset,
			past: [...this.#past],
			lastEntryAt: this.#lastEntryAt
		}
	}

	#datasetUpdate(record: LogRecord): DatasetState | undefined {
		const { message, fields, time } = record
		if (message === 'Dataset build started') this.#dataset = { phase: 'scan', done: 0, total: 0, cached: 0, startedAt: time, updatedAt: time, finished: false }
		else if (this.#dataset === undefined) return undefined
		else if (message === 'Dataset progress') this.#dataset = { ...this.#dataset, phase: String(fields['phase'] ?? ''), done: num(fields['done']), total: num(fields['total']), cached: num(fields['cached']) || this.#dataset.cached, updatedAt: time }
		else if (message === 'Dataset finished') this.#dataset = { ...this.#dataset, finished: true, updatedAt: time, summary: { ...fields } }
		return this.#dataset
	}

	#start(record: LogRecord, fields: Readonly<Record<string, unknown>>): void {
		this.#plan = { startedAt: record.time, fields }
		this.#status = 'running'
		this.#steps = []
		this.#evals = []
		this.#checkpoints = []
		this.#summary = undefined
	}

	#ensureRun(record: LogRecord, deltas: Delta[]): void {
		if (this.#plan !== undefined) return
		this.#start(record, {})
		deltas.push({ type: 'plan', plan: this.#plan! })
	}

	#archive(): void {
		if (this.#plan === undefined) return
		const last = this.#steps[this.#steps.length - 1]
		this.#past.push({ startedAt: this.#plan.startedAt, status: this.#status === 'running' ? 'stopped' : (this.#status as 'finished' | 'stopped'), lastStep: last?.step ?? 0, total: last?.total ?? 0 })
		if (this.#past.length > 20) this.#past.shift()
	}
}
