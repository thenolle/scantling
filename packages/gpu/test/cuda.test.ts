import { expect, test } from 'bun:test'
import { matmul, Random, randomNormal } from '@scantling/tensor'
import { CudaDevice, deviceMatmul, selfTest, toDevice, toHost } from '../src/index.ts'
import { referenceMatmul } from './reference.ts'

let reason: string | undefined
try {
	CudaDevice.create().dispose()
} catch (error) {
	reason = error instanceof Error ? error.message : String(error)
	console.log(`GPU tests skipped: ${reason}`)
}
const gpu = reason === undefined ? test : test.skip
const random = new Random(21)

gpu('the device reports itself and accounts for its memory', () => {
	const device = CudaDevice.create()
	expect(device.info.kind).toBe('cuda')
	expect(device.info.totalMemoryBytes).toBeGreaterThan(0)
	const buffer = device.alloc(1000)
	expect(device.memory().allocatedBytes).toBe(4000)
	device.free(buffer)
	expect(device.memory().allocatedBytes).toBe(0)
	expect(() => device.free(buffer)).toThrow('freed')
	device.dispose()
	expect(() => device.alloc(1)).toThrow('disposed')
})

gpu('uploads, downloads, fills and elementwise kernels are exact', () => {
	const device = CudaDevice.create()
	const source = Float32Array.from({ length: 1000 }, (_, index) => index)
	const buffer = device.alloc(1000)
	device.upload(buffer, source.subarray(100, 600), 10)
	const back = new Float32Array(500)
	device.download(buffer, back, 10)
	expect(Array.from(back)).toEqual(Array.from(source.subarray(100, 600)))
	device.fill(buffer, 3)
	const [other, out] = [device.alloc(1000), device.alloc(1000)]
	device.upload(other, source)
	device.add(buffer, other, out)
	device.scale(out, out, 2)
	const result = new Float32Array(1000)
	device.download(out, result)
	for (let index = 0; index < 1000; index += 1) expect(result[index]!).toBe((index + 3) * 2)
	device.dispose()
})

gpu('matmul matches a double-precision reference in every mode on awkward sizes', () => {
	const device = CudaDevice.create()
	for (const transposeA of [false, true]) {
		for (const transposeB of [false, true]) {
			for (const [m, k, n] of [[1, 1, 1], [3, 17, 5], [64, 16, 64], [65, 33, 66], [130, 70, 9], [7, 129, 200]] as const) {
				const a = randomNormal([m * k], random).data
				const b = randomNormal([k * n], random).data
				const [bufferA, bufferB, bufferC] = [device.alloc(a.length), device.alloc(b.length), device.alloc(m * n)]
				device.upload(bufferA, a)
				device.upload(bufferB, b)
				device.matmul(bufferA, bufferB, bufferC, { m, k, n, transposeA, transposeB, batch: 1, strideA: 0, strideB: 0, strideC: m * n })
				const c = new Float32Array(m * n)
				device.download(bufferC, c)
				const { values, magnitude } = referenceMatmul(a, b, m, k, n, transposeA, transposeB)
				for (let index = 0; index < c.length; index += 1) {
					expect(Math.abs(c[index]! - values[index]!), `m=${m} k=${k} n=${n} ta=${transposeA} tb=${transposeB} at ${index}`).toBeLessThanOrEqual(1e-5 * (1 + magnitude[index]!))
				}
				for (const buffer of [bufferA, bufferB, bufferC]) device.free(buffer)
			}
		}
	}
	expect(device.memory().allocatedBytes).toBe(0)
	device.dispose()
})

gpu('alpha, beta, batches and a broadcast operand behave as specified', () => {
	const device = CudaDevice.create()
	const [batch, m, k, n] = [4, 20, 30, 10]
	const a = randomNormal([batch * m * k], random).data
	const b = randomNormal([k * n], random).data
	const c0 = randomNormal([batch * m * n], random).data
	const [bufferA, bufferB, bufferC] = [device.alloc(a.length), device.alloc(b.length), device.alloc(c0.length)]
	device.upload(bufferA, a)
	device.upload(bufferB, b)
	device.upload(bufferC, c0)
	device.matmul(bufferA, bufferB, bufferC, { m, k, n, transposeA: false, transposeB: false, batch, strideA: m * k, strideB: 0, strideC: m * n, alpha: 0.5, beta: 2 })
	const c = new Float32Array(c0.length)
	device.download(bufferC, c)
	for (let index = 0; index < batch; index += 1) {
		const { values, magnitude } = referenceMatmul(a.subarray(index * m * k, (index + 1) * m * k), b, m, k, n, false, false)
		for (let element = 0; element < m * n; element += 1) {
			const old = c0[index * m * n + element]!
			expect(Math.abs(c[index * m * n + element]! - (0.5 * values[element]! + 2 * old))).toBeLessThanOrEqual(1e-5 * (1 + 0.5 * magnitude[element]! + 2 * Math.abs(old)))
		}
	}
	device.dispose()
})

gpu('device tensors agree with the CPU matmul, including irregular broadcasts', () => {
	const device = CudaDevice.create()
	const a = randomNormal([2, 1, 40, 50], random)
	const b = randomNormal([1, 3, 50, 70], random)
	const product = toHost(deviceMatmul(toDevice(device, a), toDevice(device, b)))
	const expected = matmul(a, b)
	expect(product.shape).toEqual([2, 3, 40, 70])
	for (let index = 0; index < expected.size; index += 1) expect(Math.abs(product.data[index]! - expected.data[index]!)).toBeLessThan(1e-3)
	device.dispose()
})

gpu('the self-test passes', () => {
	const device = CudaDevice.create()
	const started = performance.now()
	selfTest(device)
	console.log(`self-test took ${(performance.now() - started).toFixed(0)} ms`)
	device.dispose()
})

gpu('an allocation beyond free memory fails with a clear error', () => {
	const device = CudaDevice.create()
	const started = performance.now()
	const beyond = Math.ceil((device.memory().freeBytes + 2 * 1024 ** 3) / 4)
	expect(() => device.alloc(beyond)).toThrow('out of memory')
	console.log(`the refused allocation took ${(performance.now() - started).toFixed(0)} ms`)
	device.dispose()
}, 30_000)

gpu('bad matmul specs and foreign buffers are rejected', () => {
	const device = CudaDevice.create()
	const small = device.alloc(4)
	const big = device.alloc(100)
	expect(() => device.matmul(small, big, big, { m: 4, k: 4, n: 4, transposeA: false, transposeB: false, batch: 1, strideA: 0, strideB: 0, strideC: 16 })).toThrow(RangeError)
	const other = CudaDevice.create()
	expect(() => other.free(small)).toThrow('different device')
	other.dispose()
	device.dispose()
})

gpu('timing waits for the queued work', () => {
	const device = CudaDevice.create()
	const [m, k, n] = [1024, 1024, 1024]
	const [a, b, c] = [device.alloc(m * k), device.alloc(k * n), device.alloc(m * n)]
	device.fill(a, 1)
	device.fill(b, 1)
	const spec = { m, k, n, transposeA: false, transposeB: false, batch: 1, strideA: 0, strideB: 0, strideC: m * n }
	const once = device.measure(() => device.matmul(a, b, c, spec))
	const eight = device.measure(() => {
		for (let run = 0; run < 8; run += 1) device.matmul(a, b, c, spec)
	})
	expect(once).toBeGreaterThan(0)
	expect(eight).toBeGreaterThan(once * 2)
	const result = new Float32Array(1)
	device.download(c, result)
	expect(result[0]).toBe(1024)
	device.dispose()
})
