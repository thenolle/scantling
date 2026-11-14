import { expect, test } from 'bun:test'
import { Random, randomNormal } from '../src/random.ts'

test('the same seed gives the same sequence', () => {
	const a = new Random(42)
	const b = new Random(42)
	expect(Array.from({ length: 5 }, () => a.nextUint32())).toEqual(Array.from({ length: 5 }, () => b.nextUint32()))
	expect(new Random(1).nextUint32()).not.toBe(new Random(2).nextUint32())
})

test('state can be saved and restored mid-stream', () => {
	const random = new Random(5)
	random.normal()
	const saved = random.state()
	const expected = [random.normal(), random.normal(), random.next()]
	random.restore(saved)
	expect([random.normal(), random.normal(), random.next()]).toEqual(expected)
})

test('normal samples have mean 0 and standard deviation 1', () => {
	const tensor = randomNormal([20000], new Random(3))
	const mean = tensor.data.reduce((total, value) => total + value, 0) / tensor.size
	const variance = tensor.data.reduce((total, value) => total + (value - mean) ** 2, 0) / tensor.size
	expect(Math.abs(mean)).toBeLessThan(0.05)
	expect(Math.abs(Math.sqrt(variance) - 1)).toBeLessThan(0.05)
})

test('nextInt stays in range and covers every value', () => {
	const random = new Random(9)
	const seen = new Set<number>()
	for (let index = 0; index < 500; index += 1) seen.add(random.nextInt(6))
	expect([...seen].sort()).toEqual([0, 1, 2, 3, 4, 5])
	expect(() => random.nextInt(0)).toThrow(RangeError)
})

test('shuffle keeps every element', () => {
	const shuffled = new Random(1).shuffle([1, 2, 3, 4, 5, 6, 7, 8])
	expect([...shuffled].sort((a, b) => a - b)).toEqual([1, 2, 3, 4, 5, 6, 7, 8])
})
