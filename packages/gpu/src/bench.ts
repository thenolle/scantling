import { cpus } from 'node:os'
import { Random, Tensor, matmul, randomNormal } from '@scantling/tensor'
import { CudaDevice } from './cuda/device.ts'

interface Case {
	readonly name: string
	readonly m: number
	readonly k: number
	readonly n: number
	readonly transposeA: boolean
	readonly transposeB: boolean
	readonly batch: number
}

const cases: Case[] = [
	{ name: 'tiny ffn up (nn)', m: 256, k: 128, n: 344, transposeA: false, transposeB: false, batch: 1 },
	{ name: 'tiny head (nt)', m: 256, k: 128, n: 4096, transposeA: false, transposeB: true, batch: 1 },
	{ name: 'tiny weight grad (tn)', m: 128, k: 256, n: 344, transposeA: true, transposeB: false, batch: 1 },
	{ name: 'attention scores (nt x8)', m: 128, k: 32, n: 128, transposeA: false, transposeB: true, batch: 8 },
	{ name: 'square 512', m: 512, k: 512, n: 512, transposeA: false, transposeB: false, batch: 1 },
	{ name: 'square 1024', m: 1024, k: 1024, n: 1024, transposeA: false, transposeB: false, batch: 1 },
	{ name: 'square 2048', m: 2048, k: 2048, n: 2048, transposeA: false, transposeB: false, batch: 1 },
	{ name: 'square 4096', m: 4096, k: 4096, n: 4096, transposeA: false, transposeB: false, batch: 1 },
	{ name: 'medium ffn 4096x512x2048', m: 4096, k: 512, n: 2048, transposeA: false, transposeB: false, batch: 1 }
]

const device = CudaDevice.create(Number(Bun.argv[Bun.argv.indexOf('--index') + 1] ?? 0) || 0)
const random = new Random(11)
const gflops = (flops: number, ms: number): number => flops / (ms * 1e6)
const results: Record<string, unknown>[] = []

console.log(`Hardware  ${cpus()[0]?.model ?? 'CPU'} | ${device.info.name} (compute ${device.info.computeCapability}, ${(device.info.totalMemoryBytes / 1_073_741_824).toFixed(1)} GiB) | CUDA driver ${device.info.driverVersion} | Bun ${Bun.version}`)
console.log('Precision float32, GPU time from device events after synchronising, one thread for the CPU SIMD column\n')
console.log('case'.padEnd(28) + 'CPU SIMD'.padStart(11) + 'GPU resident'.padStart(14) + 'speedup'.padStart(9) + 'with copies'.padStart(13) + 'max error'.padStart(11))

for (const item of cases) {
	const { m, k, n, transposeA, transposeB, batch } = item
	const flops = 2 * m * k * n * batch
	const a = randomNormal([batch * m * k], random).data
	const b = randomNormal([batch * k * n], random).data
	const bufferA = device.alloc(a.length)
	const bufferB = device.alloc(b.length)
	const bufferC = device.alloc(batch * m * n)
	try {
		const spec = { m, k, n, transposeA, transposeB, batch, strideA: m * k, strideB: k * n, strideC: m * n }
		device.upload(bufferA, a)
		device.upload(bufferB, b)
		device.matmul(bufferA, bufferB, bufferC, spec)
		device.synchronize()
		const c = new Float32Array(batch * m * n)
		device.download(bufferC, c)
		let worst = 0
		for (let sample = 0; sample < 64; sample += 1) {
			const matrix = random.nextInt(batch)
			const i = random.nextInt(m)
			const j = random.nextInt(n)
			let expected = 0
			let magnitude = 0
			for (let p = 0; p < k; p += 1) {
				const term = (transposeA ? a[matrix * m * k + p * m + i]! : a[matrix * m * k + i * k + p]!) * (transposeB ? b[matrix * k * n + j * k + p]! : b[matrix * k * n + p * n + j]!)
				expected += term
				magnitude += Math.abs(term)
			}
			worst = Math.max(worst, Math.abs(c[matrix * m * n + i * n + j]! - expected) / (1 + magnitude))
		}
		const probe = device.measure(() => device.matmul(bufferA, bufferB, bufferC, spec))
		const iterations = Math.min(2000, Math.max(3, Math.round(100 / Math.max(probe, 0.001))))
		for (let warm = 0; warm < 3; warm += 1) device.matmul(bufferA, bufferB, bufferC, spec)
		const gpuMs = device.measure(() => {
			for (let run = 0; run < iterations; run += 1) device.matmul(bufferA, bufferB, bufferC, spec)
		}) / iterations
		const copies: number[] = []
		for (let run = 0; run < 5; run += 1) {
			const start = performance.now()
			device.upload(bufferA, a)
			device.upload(bufferB, b)
			device.matmul(bufferA, bufferB, bufferC, spec)
			device.download(bufferC, c)
			copies.push(performance.now() - start)
		}
		const copyMs = copies.sort((x, y) => x - y)[2]!
		let cpuRate: number | undefined
		if (flops <= 3e10) {
			const left = new Tensor(transposeA ? [batch, k, m] : [batch, m, k], a)
			const right = new Tensor(transposeB ? [batch, n, k] : [batch, k, n], b)
			matmul(left, right, { transposeA, transposeB })
			const samples: number[] = []
			const started = performance.now()
			while (samples.length < 3 || performance.now() - started < 150) {
				const begin = performance.now()
				matmul(left, right, { transposeA, transposeB })
				samples.push(performance.now() - begin)
				if (samples.length >= 200) break
			}
			cpuRate = gflops(flops, samples.sort((x, y) => x - y)[Math.floor(samples.length / 2)]!)
		}
		const gpuRate = gflops(flops, gpuMs)
		console.log(
			item.name.padEnd(28) +
				(cpuRate === undefined ? 'skipped' : cpuRate.toFixed(1)).padStart(11) +
				gpuRate.toFixed(1).padStart(14) +
				(cpuRate === undefined ? '-' : `${(gpuRate / cpuRate).toFixed(1)}x`).padStart(9) +
				gflops(flops, copyMs).toFixed(1).padStart(13) +
				worst.toExponential(1).padStart(11)
		)
		results.push({ ...item, flops, cpuGflops: cpuRate ?? null, gpuGflops: gpuRate, gpuMs, withCopiesGflops: gflops(flops, copyMs), maxRelativeError: worst, iterations })
	} finally {
		device.free(bufferA)
		device.free(bufferB)
		device.free(bufferC)
	}
}

const memory = device.memory()
console.log(`\nGPU memory after the run: ${(memory.allocatedBytes / 1_048_576).toFixed(0)} MiB held (should be 0), ${(memory.freeBytes / 1_073_741_824).toFixed(1)} GiB free`)
const path = `data/benchmarks/gpu-${new Date().toISOString().replace(/[:.]/g, '-')}.json`
await Bun.write(path, JSON.stringify({ createdAt: new Date().toISOString(), device: device.info, bun: Bun.version, results }, null, 2))
console.log(`Saved ${path}`)
device.dispose()
