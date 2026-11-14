import { appendFileSync, existsSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { readCheckpoint, readMeta, writeCheckpoint } from './format.ts'
import type { CheckpointMeta, CheckpointState } from './types.ts'

export interface RegressionResult {
	readonly name: string
	readonly passed: boolean
	readonly detail?: string
}

export interface PromotionEvidence {
	readonly evalLoss: number
	readonly regression: readonly RegressionResult[]
}

export interface GateOptions {
	/** Allowed relative worsening of the evaluation loss against the stable checkpoint. Defaults to 0.005. */
	readonly tolerance?: number
	/** Reject candidates that come with no regression results. Defaults to true. */
	readonly requireRegression?: boolean
}

export interface PromotionDecision {
	readonly accepted: boolean
	readonly reasons: readonly string[]
}

/**
 * Decides whether a candidate may become the stable checkpoint. Training loss plays no part: the
 * evaluation loss must be finite and no worse than the stable one beyond a small tolerance, and every
 * regression check must have passed.
 */
export function decidePromotion(evidence: PromotionEvidence, stableEvalLoss: number | undefined, options: GateOptions = {}): PromotionDecision {
	const reasons: string[] = []
	if (!Number.isFinite(evidence.evalLoss)) reasons.push('the evaluation loss is not finite')
	if ((options.requireRegression ?? true) && evidence.regression.length === 0) reasons.push('no regression results were provided')
	for (const check of evidence.regression) { if (!check.passed) reasons.push(`regression check "${check.name}" failed${check.detail === undefined ? '' : `: ${check.detail}`}`) }
	const tolerance = options.tolerance ?? 0.005
	if (Number.isFinite(evidence.evalLoss) && stableEvalLoss !== undefined && evidence.evalLoss > stableEvalLoss * (1 + tolerance)) reasons.push(`the evaluation loss ${evidence.evalLoss} is worse than the stable checkpoint's ${stableEvalLoss} by more than ${tolerance * 100}%`)
	return { accepted: reasons.length === 0, reasons }
}

interface Stable {
	readonly id: string
	readonly evalLoss: number
	readonly promotedAt: string
}

export interface HistoryEntry {
	readonly at: string
	readonly action: 'promote' | 'reject' | 'rollback'
	readonly id: string
	readonly evalLoss?: number
	readonly previous?: string
	readonly reasons?: readonly string[]
}

export const checkpointId = (step: number): string => `step-${String(step).padStart(8, '0')}`
const ID_PATTERN = /^step-\d{8,}$/

function writeAtomically(path: string, text: string): void {
	const temporary = `${path}.tmp-${process.pid}`
	writeFileSync(temporary, text)
	renameSync(temporary, path)
}

/**
 * A directory of checkpoints with a promotable "stable" pointer. All writes are atomic: a checkpoint is
 * assembled in a temporary directory and renamed into place, and the pointer file is replaced by rename.
 * Calls are synchronous; run training in a worker so saving does not block the interface.
 */
export class CheckpointStore {
	constructor(readonly root: string) {
		mkdirSync(root, { recursive: true })
	}

	/** Saves a checkpoint named after its step. Existing checkpoints are never overwritten. */
	save(state: CheckpointState): string {
		const id = checkpointId(state.step)
		const final = join(this.root, id)
		if (existsSync(final)) throw new Error(`Checkpoint ${id} already exists`)
		const temporary = join(this.root, `.tmp-${id}-${process.pid}`)
		rmSync(temporary, { recursive: true, force: true })
		try {
			writeCheckpoint(temporary, state, new Date().toISOString())
			renameSync(temporary, final)
		} catch (error) {
			rmSync(temporary, { recursive: true, force: true })
			throw error
		}
		return id
	}

	/** Loads a checkpoint, verifying every checksum. */
	load(id: string): CheckpointState {
		return readCheckpoint(this.#path(id))
	}

	/** Summaries of every checkpoint, oldest first. */
	list(): (CheckpointMeta & { readonly id: string })[] {
		return readdirSync(this.root).filter((name) => ID_PATTERN.test(name)).sort().map((id) => ({ ...readMeta(join(this.root, id)), id }))
	}

	latest(): string | undefined {
		return this.list().at(-1)?.id
	}

	stable(): Stable | undefined {
		const file = join(this.root, 'stable.json')
		return existsSync(file) ? (JSON.parse(readFileSync(file, 'utf8')) as Stable) : undefined
	}

	history(): HistoryEntry[] {
		const file = join(this.root, 'promotions.jsonl')
		if (!existsSync(file)) return []
		return readFileSync(file, 'utf8').split('\n').filter((line) => line !== '').map((line) => JSON.parse(line) as HistoryEntry)
	}

	/**
	 * Promotes `id` to stable if the gate accepts the evidence. The decision, accepted or not, is appended
	 * to the history. The previous stable checkpoint stays on disk.
	 */
	promote(id: string, evidence: PromotionEvidence, options: GateOptions = {}): PromotionDecision {
		this.#path(id)
		const current = this.stable()
		const decision = decidePromotion(evidence, current?.evalLoss, options)
		const at = new Date().toISOString()
		if (!decision.accepted) {
			this.#record({ at, action: 'reject', id, evalLoss: evidence.evalLoss, reasons: decision.reasons })
			return decision
		}
		this.#record({ at, action: 'promote', id, evalLoss: evidence.evalLoss, ...(current === undefined ? {} : { previous: current.id }) })
		writeAtomically(join(this.root, 'stable.json'), JSON.stringify({ id, evalLoss: evidence.evalLoss, promotedAt: at } satisfies Stable, null, 2))
		return decision
	}

	/** Makes the previously promoted checkpoint stable again. Returns its id, or `undefined` if there is none. */
	rollback(): string | undefined {
		const current = this.stable()
		if (current === undefined) return undefined
		const promotions = this.history().filter((entry) => entry.action === 'promote')
		const index = promotions.map((entry) => entry.id).lastIndexOf(current.id)
		const previous = promotions[index - 1]
		if (previous === undefined || previous.evalLoss === undefined) return undefined
		this.#path(previous.id)
		const at = new Date().toISOString()
		this.#record({ at, action: 'rollback', id: previous.id, previous: current.id })
		writeAtomically(join(this.root, 'stable.json'), JSON.stringify({ id: previous.id, evalLoss: previous.evalLoss, promotedAt: at } satisfies Stable, null, 2))
		return previous.id
	}

	/**
	 * Deletes all but the newest `keep` checkpoints. The stable checkpoint and every checkpoint that was
	 * ever promoted are kept so that rollback always works. Returns the deleted ids.
	 */
	prune(keep: number): string[] {
		const protectedIds = new Set([this.stable()?.id, ...this.history().filter((entry) => entry.action !== 'reject').map((entry) => entry.id)])
		const ids = this.list().map((meta) => meta.id)
		const removable = ids.slice(0, Math.max(0, ids.length - keep)).filter((id) => !protectedIds.has(id))
		for (const id of removable) rmSync(join(this.root, id), { recursive: true, force: true })
		return removable
	}

	#path(id: string): string {
		if (!ID_PATTERN.test(id)) throw new RangeError(`"${id}" is not a checkpoint id`)
		const path = join(this.root, id)
		if (!existsSync(path)) throw new Error(`Checkpoint ${id} does not exist`)
		return path
	}

	#record(entry: HistoryEntry): void {
		appendFileSync(join(this.root, 'promotions.jsonl'), `${JSON.stringify(entry)}\n`)
	}
}
