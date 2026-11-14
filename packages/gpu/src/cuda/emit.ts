export class Registers {
	#p = 0
	#r = 0
	#rd = 0
	#f = 0

	p(): string {
		return `%p${this.#p++}`
	}

	r(): string {
		return `%r${this.#r++}`
	}

	rd(): string {
		return `%rd${this.#rd++}`
	}

	f(): string {
		return `%f${this.#f++}`
	}

	declarations(): string[] {
		return [
			`.reg .pred %p<${Math.max(1, this.#p)}>;`,
			`.reg .b32 %r<${Math.max(1, this.#r)}>;`,
			`.reg .b64 %rd<${Math.max(1, this.#rd)}>;`,
			`.reg .f32 %f<${Math.max(1, this.#f)}>;`
		]
	}
}

export function entry(name: string, params: readonly string[], regs: Registers, body: readonly string[], shared?: string): string {
	return [
		`.visible .entry ${name}(`,
		...params.map((param, index) => `\t${param}${index < params.length - 1 ? ',' : ''}`),
		')',
		'{',
		...regs.declarations().map((line) => `\t${line}`),
		...(shared === undefined ? [] : [`\t${shared}`]),
		...body,
		'\tret;',
		'}'
	].join('\n')
}
