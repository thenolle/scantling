import { expect, test } from 'bun:test'
import { moduleSource } from '../src/cuda/module.ts'
import { KERNEL_NAMES, elementwiseKernel, sgemmKernel } from '../src/cuda/ptx.ts'

const kernels = [
	['sgemm_nn', sgemmKernel('sgemm_nn', false, false), 15],
	['sgemm_nt', sgemmKernel('sgemm_nt', false, true), 15],
	['sgemm_tn', sgemmKernel('sgemm_tn', true, false), 15],
	['sgemm_tt', sgemmKernel('sgemm_tt', true, true), 15],
	['add_f32', elementwiseKernel('add_f32'), 4],
	['scale_f32', elementwiseKernel('scale_f32'), 4]
] as const

for (const [name, text, parameters] of kernels) {
	test(`${name} declares every register it uses and every label it jumps to`, () => {
		expect((text.match(/\.param /g) ?? []).length).toBe(parameters)
		for (const [prefix, bank] of [['p', 'pred'], ['rd', 'b64'], ['r', 'b32'], ['f', 'f32']] as const) {
			const declared = Number(new RegExp(`\\.reg \\.${bank} %${prefix}<(\\d+)>`).exec(text)?.[1])
			const used = [...text.matchAll(new RegExp(`%${prefix}(\\d+)\\b`, 'g'))].map((match) => Number(match[1]))
			expect(declared).toBeGreaterThan(0)
			expect(Math.max(...used)).toBeLessThan(declared)
		}
		const labels = new Set([...text.matchAll(/^(\$\w+):$/gm)].map((match) => match[1]))
		for (const target of text.matchAll(/bra (\$\w+);/g)) expect(labels.has(target[1]!)).toBe(true)
		expect(text.split('{').length).toBe(text.split('}').length)
	})
}

test('the module declares one version and target and contains every kernel once', () => {
	const source = moduleSource()
	expect(source.match(/\.version/g)?.length).toBe(1)
	expect(source.match(/\.target/g)?.length).toBe(1)
	for (const name of KERNEL_NAMES) expect(source.match(new RegExp(`\\.entry ${name}\\(`, 'g'))?.length).toBe(1)
})
