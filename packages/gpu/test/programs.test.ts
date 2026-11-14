import { expect, test } from 'bun:test'
import { PROGRAMS, PROGRAM_NAMES } from '../src/cuda/ops.ts'
import { moduleSource } from '../src/cuda/module.ts'
import { checkStructure } from './structure.ts'

for (const name of PROGRAM_NAMES) {
	test(`${name} is well-formed PTX`, () => {
		const program = PROGRAMS[name]
		checkStructure(program.text, program.buffers.length + program.ints.length + program.floats.length)
		expect(program.text).toContain(`.entry ${name}(`)
	})
}

test('the module contains each program once', () => {
	const source = moduleSource()
	for (const name of PROGRAM_NAMES) expect(source.match(new RegExp(`\\.entry ${name}\\(`, 'g'))?.length).toBe(1)
})
