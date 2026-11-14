import { mkdir } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { backward, noGrad, ops, parameter, variable, zeroGrad } from '@scantling/autodiff'
import { computeBudget, detectHardware, formatBytes } from '@scantling/hardware'
import * as T from '@scantling/tensor'
import { ParallelMatmul } from '@scantling/workers'
import { measure } from './harness.ts'

interface Result {
	readonly section: string
	readonly name: string
	readonly value: number
	readonly unit: string
	readonly note: string
}

const args = new Set(Bun.argv.slice(2))
const sizes = args.has('--quick') ? [64, 128, 256] : args.has('--full') ? [64, 128, 256, 512, 1024] : [64, 128, 256, 512]
const results: Result[] = []
const gflops = (flops: number, milliseconds: number): number => flops / (milliseconds * 1e6)
const record = (section: string, name: string, value: number, unit: string, note = ''): void => {
	results.push({ section, name, value, unit, note })
}

const profile = await detectHardware({ storagePaths: [process.cwd()] })
const budget = computeBudget(profile)
const random = new T.Random(1)
const singleThread = new Map<number, number>()

for (const simd of [false, true]) {
	T.setSimdEnabled(simd)
	for (const n of sizes) {
		const a = T.randomNormal([n, n], random)
		const b = T.randomNormal([n, n], random)
		for (const [name, options] of [['nn', {}], ['nt', { transposeB: true }], ['tn', { transposeA: true }]] as const) {
			const sample = await measure(`${name} ${n}`, () => T.matmul(a, b, options))
			const rate = gflops(2 * n ** 3, sample.medianMs)
			if (name === 'nn' && !simd) singleThread.set(n, rate)
			record(`Matmul, one thread, ${simd ? 'SIMD' : 'scalar'}`, `${name} ${n}x${n}`, rate, 'GFLOP/s', `${sample.medianMs.toFixed(2)} ms median over ${sample.runs} runs`)
		}
	}
}
T.setSimdEnabled(false)

const poolSizes = [...new Set([1, 2, 4, budget.workerThreads, profile.cpu.logicalCores])].filter((size) => size <= profile.cpu.logicalCores).sort((x, y) => x - y)
for (const size of poolSizes) {
	const pool = new ParallelMatmul({ workers: size, minimumFlops: 0 })
	for (const n of sizes.filter((value) => value >= 64)) {
		const a = T.randomNormal([n, n], random)
		const b = T.randomNormal([n, n], random)
		const sample = await measure(`parallel ${size} ${n}`, () => pool.matmul(a, b))
		const rate = gflops(2 * n ** 3, sample.medianMs)
		const base = singleThread.get(n)
		record('Matmul, worker pool (nn)', `${size} workers, ${n}x${n}`, rate, 'GFLOP/s', base === undefined ? '' : `${(rate / base).toFixed(2)}x of one thread`)
	}
	await pool.close()
}

const wide = T.randomNormal([1 << 20], random)
const add = await measure('add', () => T.add(wide, wide))
record('Elementwise', 'add 1M elements', (1 << 20) / (add.medianMs * 1e3), 'Melem/s')
const logits = T.randomNormal([512, 1024], random)
const soft = await measure('softmax', () => T.softmax(logits))
record('Elementwise', 'softmax 512x1024', 512 / (soft.medianMs / 1000), 'rows/s')

const x = variable(T.randomNormal([512, 128], random))
const weight = parameter(T.randomNormal([128, 512], random))
const linearFlops = 2 * 512 * 128 * 512
const forward = await measure('forward', () => noGrad(() => ops.matmul(x, weight)))
record('Linear layer 512x128 -> 512', 'forward', gflops(linearFlops, forward.medianMs), 'GFLOP/s', `${forward.medianMs.toFixed(2)} ms`)
const both = await measure('forward+backward', () => {
	zeroGrad([weight])
	backward(ops.sum(ops.matmul(x, weight)))
})
record('Linear layer 512x128 -> 512', 'forward + backward', gflops(3 * linearFlops, both.medianMs), 'GFLOP/s', `${both.medianMs.toFixed(2)} ms`)

const best = Math.max(...results.filter((result) => result.section.startsWith('Matmul')).map((result) => result.value))
for (const [name, parameters] of [['Micro', 0.15e6], ['Tiny', 1.3e6], ['Small', 14e6]] as const) record('Ceiling tokens/s', `${name} (~${(parameters / 1e6).toFixed(2)}M parameters)`, (best * 1e9) / (6 * parameters), 'tokens/s', 'upper bound: best measured GFLOP/s / (6 x parameters)')

const header = `${profile.cpu.model} | ${profile.cpu.logicalCores} logical cores | ${formatBytes(profile.memory.totalBytes)} RAM | Bun ${profile.runtime.bunVersion} | ${profile.runtime.osName}`
if (args.has('--markdown')) {
	console.log(`Measured on: ${header}\n`)
	for (const section of [...new Set(results.map((result) => result.section))]) {
		console.log(`### ${section}\n\n| Case | Result | Notes |\n| --- | --- | --- |`)
		for (const result of results.filter((item) => item.section === section)) console.log(`| ${result.name} | ${result.value.toFixed(2)} ${result.unit} | ${result.note} |`)
		console.log('')
	}
} else {
	console.log(header)
	for (const section of [...new Set(results.map((result) => result.section))]) {
		console.log(`\n${section}`)
		for (const result of results.filter((item) => item.section === section)) {
			console.log(`  ${result.name.padEnd(34)} ${result.value.toFixed(2).padStart(10)} ${result.unit.padEnd(9)} ${result.note}`)
		}
	}
}

const directory = resolve(process.cwd(), 'data', 'benchmarks')
await mkdir(directory, { recursive: true })
const file = join(directory, `${new Date().toISOString().replace(/[:.]/g, '-')}.json`)
await Bun.write(file, JSON.stringify({ profile, results }, null, 2))
console.error(`\nSaved ${file}`)
