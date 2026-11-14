import { expect } from 'bun:test'

/** Every register used is declared, every jump target exists, braces balance and the parameter count is right. */
export function checkStructure(text: string, parameters: number): void {
	expect((text.match(/\.param /g) ?? []).length).toBe(parameters)
	for (const [prefix, bank] of [['p', 'pred'], ['rd', 'b64'], ['r', 'b32'], ['f', 'f32']] as const) {
		const declared = Number(new RegExp(`\\.reg \\.${bank} %${prefix}<(\\d+)>`).exec(text)?.[1])
		const used = [...text.matchAll(new RegExp(`%${prefix}(\\d+)\\b`, 'g'))].map((match) => Number(match[1]))
		expect(declared).toBeGreaterThan(0)
		expect(Math.max(-1, ...used)).toBeLessThan(declared)
	}
	const labels = new Set([...text.matchAll(/^(\$\w+):$/gm)].map((match) => match[1]))
	for (const target of text.matchAll(/bra (\$\w+);/g)) expect(labels.has(target[1]!)).toBe(true)
	expect(text.split('{').length).toBe(text.split('}').length)
}
