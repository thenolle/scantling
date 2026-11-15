import { closeSync, fsyncSync, mkdirSync, openSync, readFileSync, writeSync } from 'node:fs'
import { join } from 'node:path'
import { Tensor } from '@scantling/tensor'
import type { ModelConfigInput } from '@scantling/nn'
import type { CheckpointMeta, CheckpointState, TensorEntry } from './types.ts'
import { CorruptCheckpointError } from './types.ts'

const FORMAT = 'scantling-checkpoint'
const sha256 = (bytes: Uint8Array): string => new Bun.CryptoHasher('sha256').update(bytes).digest('hex')

function writeDurably(path: string, bytes: Uint8Array | string): void {
	const fd = openSync(path, 'w')
	try {
		writeSync(fd, typeof bytes === 'string' ? new TextEncoder().encode(bytes) : bytes)
		fsyncSync(fd)
	} finally {
		closeSync(fd)
	}
}

/**
 * Writes a checkpoint into an empty `directory`: raw Float32 tensors in `tensors.bin` (sorted by group
 * and name, each with its own checksum) and the description in `meta.json`.
 */
export function writeCheckpoint(directory: string, state: CheckpointState, createdAt: string): void {
	mkdirSync(directory, { recursive: true })
	const entries: TensorEntry[] = []
	const chunks: Uint8Array[] = []
	const overall = new Bun.CryptoHasher('sha256')
	let offset = 0
	const groups = [['weights', state.weights], ['optimizer', state.optimizer.tensors]] as const
	for (const [group, tensors] of groups) {
		for (const name of [...tensors.keys()].sort()) {
			const tensor = tensors.get(name)!
			const bytes = new Uint8Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength)
			entries.push({ group, name, shape: [...tensor.shape], offset, bytes: bytes.length, sha256: sha256(bytes) })
			overall.update(bytes)
			chunks.push(bytes)
			offset += bytes.length
		}
	}
	const joined = new Uint8Array(offset)
	let position = 0
	for (const chunk of chunks) {
		joined.set(chunk, position)
		position += chunk.length
	}
	writeDurably(join(directory, 'tensors.bin'), joined)
	const meta: CheckpointMeta = {
		format: FORMAT,
		version: 1,
		createdAt,
		step: state.step,
		skippedSteps: state.skippedSteps,
		config: state.config,
		tensors: entries,
		optimizer: { kind: state.optimizer.kind, step: state.optimizer.step },
		random: state.random,
		metrics: state.metrics,
		notes: state.notes ?? '',
		sha256: overall.digest('hex')
	}
	writeDurably(join(directory, 'meta.json'), JSON.stringify(meta, null, 2))
}

/** Reads and validates `meta.json` without touching the tensors. */
export function readMeta(directory: string): CheckpointMeta {
	let raw: unknown
	try { raw = JSON.parse(readFileSync(join(directory, 'meta.json'), 'utf8')) } catch (error) { throw new CorruptCheckpointError(`Cannot read meta.json in ${directory}: ${error instanceof Error ? error.message : String(error)}`) }
	const meta = raw as Partial<CheckpointMeta> | null
	if (meta === null || typeof meta !== 'object' || meta.format !== FORMAT || meta.version !== 1) throw new CorruptCheckpointError('Not a scantling checkpoint, or an unsupported version')
	if (!Array.isArray(meta.tensors) || typeof meta.sha256 !== 'string' || typeof meta.step !== 'number') throw new CorruptCheckpointError('meta.json is incomplete')
	return meta as CheckpointMeta
}

/**
 * Loads a checkpoint and verifies the whole-file checksum and every tensor checksum.
 * Throws `CorruptCheckpointError` on any mismatch.
 */
export function readCheckpoint(directory: string): CheckpointState {
	const meta = readMeta(directory)
	let data: Uint8Array
	try { data = new Uint8Array(readFileSync(join(directory, 'tensors.bin'))) } catch (error) { throw new CorruptCheckpointError(`Cannot read tensors.bin in ${directory}: ${error instanceof Error ? error.message : String(error)}`) }
	if (sha256(data) !== meta.sha256) throw new CorruptCheckpointError('tensors.bin does not match its recorded checksum')
	const weights = new Map<string, Tensor>()
	const optimizer = new Map<string, Tensor>()
	for (const entry of meta.tensors) {
		if (entry.offset < 0 || entry.bytes % 4 !== 0 || entry.offset + entry.bytes > data.length) throw new CorruptCheckpointError(`Tensor "${entry.name}" lies outside the file`)
		const bytes = data.slice(entry.offset, entry.offset + entry.bytes)
		if (sha256(bytes) !== entry.sha256) throw new CorruptCheckpointError(`Tensor "${entry.name}" does not match its recorded checksum`)
		const tensor = new Tensor(entry.shape, new Float32Array(bytes.buffer, 0, entry.bytes / 4))
			; (entry.group === 'weights' ? weights : optimizer).set(entry.name, tensor)
	}
	return {
		step: meta.step,
		skippedSteps: meta.skippedSteps,
		config: meta.config as ModelConfigInput,
		weights,
		optimizer: { kind: meta.optimizer.kind, step: meta.optimizer.step, tensors: optimizer },
		random: meta.random,
		metrics: meta.metrics,
		notes: meta.notes
	}
}
