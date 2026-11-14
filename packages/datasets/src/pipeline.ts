import { mkdir } from 'node:fs/promises'
import { join, resolve, sep } from 'node:path'
import { collectTextFiles } from '@scantling/tokenizer'
import type { Tokenizer } from '@scantling/tokenizer'
import { isCodeLikePath, isDataList, isGeneratedPath, isMinifiedText, isSensitivePath, scanDocument } from './scan.ts'
import { TokenWriter } from './store.ts'

export const MANIFEST_FORMAT = 'scantling-dataset'

export interface Disposition {
	readonly path: string
	readonly status: 'included' | 'excluded'
	readonly reason?: string
	readonly split?: 'train' | 'validation'
	readonly tokens?: number
	readonly repeat?: number
}

export interface DatasetManifest {
	readonly format: typeof MANIFEST_FORMAT
	readonly version: 1
	readonly createdAt: string
	readonly tokenizer: { readonly vocabSize: number; readonly sha256: string }
	readonly validationPercent: number
	readonly totals: { readonly documents: number; readonly bytes: number; readonly trainTokens: number; readonly validationTokens: number }
	readonly files: readonly Disposition[]
}

export interface BuildOptions {
	readonly inputs: readonly string[]
	readonly tokenizer: Tokenizer
	readonly outputDirectory: string
	/** Share of files used for validation, from 0 to 100. Defaults to 5. */
	readonly validationPercent?: number
	/** Without approval nothing is written: the build only reports what it would do. */
	readonly approved: boolean
	readonly maxFileBytes?: number
	/** Whole-number repeat counts for training documents under a path prefix. Validation is never repeated. */
	readonly weights?: Readonly<Record<string, number>>
}

export interface DatasetReport {
	readonly approved: boolean
	readonly manifest: DatasetManifest
	readonly manifestPath: string | undefined
}

const sha256 = (text: string): string => new Bun.CryptoHasher('sha256').update(text).digest('hex')

/** Whitespace-normalised lines of at least 30 characters, hashed. */
export function lineHashes(text: string): string[] {
	return text.split('\n').map((line) => line.trim().replace(/\s+/g, ' ')).filter((line) => line.length >= 30).map((line) => String(Bun.hash(line)))
}

/** The share of `hashes` that also occur in `train`. A document without substantial lines scores 0. */
export function overlapRatio(hashes: readonly string[], train: ReadonlySet<string>): number {
	if (hashes.length === 0) return 0
	return hashes.filter((hash) => train.has(hash)).length / hashes.length
}

/** Assigns a path to a split from a hash of the path, so the assignment never changes between runs. */
export function splitOf(path: string, validationPercent: number): 'train' | 'validation' {
	return Number.parseInt(sha256(path.replace(/\\/g, '/')).slice(0, 8), 16) % 100 < validationPercent ? 'validation' : 'train'
}

/** How many times a training document is written: the largest multiplier whose path prefix matches, or 1. */
export function repeatOf(path: string, weights: Readonly<Record<string, number>> | undefined): number {
	if (weights === undefined) return 1
	const normal = path.replace(/\\/g, '/').toLowerCase()
	let repeat = 1
	for (const [prefix, count] of Object.entries(weights)) {
		const base = prefix.replace(/\\/g, '/').toLowerCase()
		if (normal.startsWith(base.endsWith('/') ? base : `${base}/`)) repeat = Math.max(repeat, count)
	}
	return repeat
}

/**
 * Reads the text files under `inputs`, drops sensitive, secret-bearing and duplicate files, splits the
 * rest into train and validation, drops validation files that overlap the training data, tokenizes each
 * file as `<|file|> ... <|endfile|>` and, when approved, writes `train.tokens`, `validation.tokens` and
 * `manifest.json`. Without approval it only reports what would happen.
 */
export async function buildDataset(options: BuildOptions): Promise<DatasetReport> {
	const percent = options.validationPercent ?? 5
	if (!(percent >= 0 && percent <= 100)) throw new RangeError('validationPercent must be between 0 and 100')
	const { tokenizer } = options
	const fileId = tokenizer.idOf('<|file|>')
	const endId = tokenizer.idOf('<|endfile|>')
	const encoder = new TextEncoder()
	const files: Disposition[] = []
	const seen = new Set<string>()
	const trainLines = new Set<string>()
	const pending: { path: string; ids: number[]; lines: string[] }[] = []
	const outputRoot = resolve(options.outputDirectory) + sep
	let bytes = 0
	let trainTokens = 0
	let validationTokens = 0
	if (options.approved) await mkdir(options.outputDirectory, { recursive: true })
	const train = options.approved ? new TokenWriter(join(options.outputDirectory, 'train.tokens'), tokenizer.vocabSize) : undefined
	const validation = options.approved ? new TokenWriter(join(options.outputDirectory, 'validation.tokens'), tokenizer.vocabSize) : undefined
	try {
		for await (const file of collectTextFiles(options.inputs, options.maxFileBytes)) {
			if (resolve(file.path).startsWith(outputRoot)) {
				files.push({ path: file.path, status: 'excluded', reason: 'output-directory' })
				continue
			}
			if (isGeneratedPath(file.path)) {
				files.push({ path: file.path, status: 'excluded', reason: 'generated' })
				continue
			}
			if (isSensitivePath(file.path)) {
				files.push({ path: file.path, status: 'excluded', reason: 'sensitive-path' })
				continue
			}
			if (!isCodeLikePath(file.path, file.text.length)) {
				files.push({ path: file.path, status: 'excluded', reason: 'not-code' })
				continue
			}
			if (isMinifiedText(file.text)) {
				files.push({ path: file.path, status: 'excluded', reason: 'minified' })
				continue
			}
			if (isDataList(file.text)) {
				files.push({ path: file.path, status: 'excluded', reason: 'data-list' })
				continue
			}
			const scan = scanDocument(file.text)
			if (scan.secrets.length > 0) {
				files.push({ path: file.path, status: 'excluded', reason: `secret:${scan.secrets.join('+')}` })
				continue
			}
			const digest = sha256(scan.text.replace(/\r\n/g, '\n').trimEnd())
			if (seen.has(digest)) {
				files.push({ path: file.path, status: 'excluded', reason: 'duplicate' })
				continue
			}
			seen.add(digest)
			const ids = [fileId, ...tokenizer.encode(scan.text), endId]
			bytes += encoder.encode(scan.text).length
			if (splitOf(file.path, percent) === 'train') {
				const repeat = repeatOf(file.path, options.weights)
				for (let copy = 0; copy < repeat; copy += 1) train?.append(ids)
				trainTokens += ids.length * repeat
				for (const hash of lineHashes(scan.text)) trainLines.add(hash)
				files.push({ path: file.path, status: 'included', split: 'train', tokens: ids.length, repeat })
			} else {
				pending.push({ path: file.path, ids, lines: lineHashes(scan.text) })
			}
		}
		for (const document of pending) {
			if (overlapRatio(document.lines, trainLines) >= 0.5) {
				files.push({ path: document.path, status: 'excluded', reason: 'overlaps-train' })
				continue
			}
			validation?.append(document.ids)
			validationTokens += document.ids.length
			files.push({ path: document.path, status: 'included', split: 'validation', tokens: document.ids.length })
		}
	} finally {
		train?.close()
		validation?.close()
	}
	const manifest: DatasetManifest = {
		format: MANIFEST_FORMAT,
		version: 1,
		createdAt: new Date().toISOString(),
		tokenizer: { vocabSize: tokenizer.vocabSize, sha256: sha256(tokenizer.toJson()) },
		validationPercent: percent,
		totals: { documents: files.filter((file) => file.status === 'included').length, bytes, trainTokens, validationTokens },
		files
	}
	const manifestPath = options.approved ? join(options.outputDirectory, 'manifest.json') : undefined
	if (manifestPath !== undefined) await Bun.write(manifestPath, JSON.stringify(manifest, null, 2))
	return { approved: options.approved, manifest, manifestPath }
}
