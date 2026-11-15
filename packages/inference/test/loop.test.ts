import { expect, test } from 'bun:test'
import { endsInLoop } from '../src/loop.ts'

test('a repeated block at the end is a loop and ordinary text is not', () => {
	const block = [5, 6, 7, 8, 9, 10, 11, 12, 13]
	const looped = [1, 2, 3, 4, ...block, ...block, ...block, ...block]
	expect(endsInLoop(looped)).toBe(true)
	expect(endsInLoop(looped.slice(0, -block.length * 2))).toBe(false)
	expect(endsInLoop(Array.from({ length: 200 }, (_, index) => index % 97))).toBe(false)
	expect(endsInLoop([3, ...Array.from({ length: 30 }, () => 4)])).toBe(true)
	expect(endsInLoop([])).toBe(false)
})
