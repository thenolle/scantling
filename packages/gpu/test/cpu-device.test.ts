import { expect, test } from 'bun:test'
import { Random, Tensor, matmul, randomNormal } from '@scantling/tensor'
import { CpuDevice, deviceMatmul, deviceZeros, selectDevice, toDevice, toHost } from '../src/index.ts'
import { referenceMatmul } from './reference.ts'

const random = new Random(3)

test('buffers round-trip and the elementwise operations work', () => {
	const device = new CpuDevice()
	const [a, b, out] = [device.alloc(4), device.alloc(4), device.alloc(4)]
	device.upload(a, Float32Array.of(1, 2, 3, 4))
	device.fill(b, 10)
	device.add(a, b, out)
	device.scale(out, out, 2)
	const result = new Float32Array(4)
	device.download(out, result)
	expect(Array.from(result)).toEqual([22, 24, 26, 28])
	expect(device.memory().allocatedBytes).toBe(48)
	for (const buffer of [a, b, out]) device.free(buffer)
	expect(device.memory().allocatedBytes).toBe(0)
	expect(() => device.upload(a, Float32Array.of(1))).toThrow('freed')
})

test('matmul covers every transpose mode, alpha and beta', () => {
	const device = new CpuDevice()
	for (const transposeA of [false, true]) {
		for (const transposeB of [false, true]) {
			const [m, k, n] = [5, 7, 3]
			const a = randomNormal([m * k], random).data
			const b = randomNormal([k * n], random).data
			const [bufferA, bufferB, bufferC] = [device.alloc(a.length), device.alloc(b.length), device.alloc(m * n)]
			device.upload(bufferA, a)
			device.upload(bufferB, b)
			device.fill(bufferC, 1)
			device.matmul(bufferA, bufferB, bufferC, { m, k, n, transposeA, transposeB, batch: 1, strideA: 0, strideB: 0, strideC: m * n, alpha: 2, beta: 0.5 })
			const c = new Float32Array(m * n)
			device.download(bufferC, c)
			const { values } = referenceMatmul(a, b, m, k, n, transposeA, transposeB)
			for (let index = 0; index < c.length; index += 1) expect(c[index]!).toBeCloseTo(2 * values[index]! + 0.5, 4)
		}
	}
})

test('device tensors match CPU matmul, including irregular batch broadcasting', () => {
	const device = new CpuDevice()
	const a = randomNormal([2, 1, 4, 5], random)
	const b = randomNormal([1, 3, 5, 6], random)
	const product = toHost(deviceMatmul(toDevice(device, a), toDevice(device, b)))
	const expected = matmul(a, b)
	expect(product.shape).toEqual([2, 3, 4, 6])
	for (let index = 0; index < expected.size; index += 1) expect(product.data[index]!).toBeCloseTo(expected.data[index]!, 5)
	const shared = toHost(deviceMatmul(toDevice(device, randomNormal([3, 4, 5], random)), toDevice(device, randomNormal([5, 2], random))))
	expect(shared.shape).toEqual([3, 4, 2])
})

test('ownership rules reject cross-device use, disposed tensors and bad sizes', () => {
	const first = new CpuDevice()
	const second = new CpuDevice()
	const a = toDevice(first, Tensor.ones([2, 3]))
	const b = toDevice(second, Tensor.ones([3, 2]))
	expect(() => deviceMatmul(a, b)).toThrow('different devices')
	const c = deviceZeros(first, [3, 2])
	a.dispose()
	expect(() => deviceMatmul(a, c)).toThrow('disposed')
	expect(() => toHost(a)).toThrow('disposed')
	const small = first.alloc(4)
	const big = first.alloc(100)
	expect(() => first.matmul(small, big, big, { m: 4, k: 4, n: 4, transposeA: false, transposeB: false, batch: 1, strideA: 0, strideB: 0, strideC: 16 })).toThrow(RangeError)
	expect(() => first.free(b.buffer)).toThrow('different device')
})

test('selection honours the request and never reports a GPU it is not using', () => {
	const cpu = selectDevice('cpu')
	expect(cpu.device.info.kind).toBe('cpu')
	expect(cpu.fallbackReason).toBeUndefined()
	const auto = selectDevice('auto')
	if (auto.device.info.kind === 'cpu') expect(auto.fallbackReason).toBeString()
	else expect(auto.fallbackReason).toBeUndefined()
	auto.device.dispose()
})
