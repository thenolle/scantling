import { Registers, entry } from './emit.ts'

export type CompareOp = 'eq' | 'lt' | 'ge'

/**
 * A scalar float expression evaluated per element, with these leaves: `at` reads `buffer[row * cols + col]`
 * (or `buffer[i]` in an elementwise kernel), `colAt` reads `buffer[col]`, `rowAt` reads `buffer[row]`,
 * `gather` reads `buffer[row * cols + index]`, `index` is the row or column number, `arg` is a kernel argument or a
 * reduced scalar computed earlier in the same kernel.
 */
export type Expr =
	| { readonly kind: 'num'; readonly value: number }
	| { readonly kind: 'arg'; readonly name: string }
	| { readonly kind: 'at'; readonly buffer: string }
	| { readonly kind: 'col'; readonly buffer: string }
	| { readonly kind: 'row'; readonly buffer: string }
	| { readonly kind: 'index'; readonly axis: 'row' | 'col' }
	| { readonly kind: 'gather'; readonly buffer: string; readonly position: Expr }
	| { readonly kind: 'bin'; readonly op: 'add' | 'sub' | 'mul' | 'div' | 'max'; readonly a: Expr; readonly b: Expr }
	| { readonly kind: 'un'; readonly op: 'neg' | 'exp' | 'log' | 'sqrt' | 'rsqrt'; readonly a: Expr }
	| { readonly kind: 'sel'; readonly cmp: CompareOp; readonly a: Expr; readonly b: Expr; readonly yes: Expr; readonly no: Expr }

export const num = (value: number): Expr => ({ kind: 'num', value })
export const arg = (name: string): Expr => ({ kind: 'arg', name })
export const at = (buffer: string): Expr => ({ kind: 'at', buffer })
export const colAt = (buffer: string): Expr => ({ kind: 'col', buffer })
export const rowAt = (buffer: string): Expr => ({ kind: 'row', buffer })
export const index = (axis: 'row' | 'col'): Expr => ({ kind: 'index', axis })
export const gather = (buffer: string, position: Expr): Expr => ({ kind: 'gather', buffer, position })
export const add = (a: Expr, b: Expr): Expr => ({ kind: 'bin', op: 'add', a, b })
export const sub = (a: Expr, b: Expr): Expr => ({ kind: 'bin', op: 'sub', a, b })
export const mul = (a: Expr, b: Expr): Expr => ({ kind: 'bin', op: 'mul', a, b })
export const div = (a: Expr, b: Expr): Expr => ({ kind: 'bin', op: 'div', a, b })
export const maximum = (a: Expr, b: Expr): Expr => ({ kind: 'bin', op: 'max', a, b })
export const neg = (a: Expr): Expr => ({ kind: 'un', op: 'neg', a })
export const exp = (a: Expr): Expr => ({ kind: 'un', op: 'exp', a })
export const log = (a: Expr): Expr => ({ kind: 'un', op: 'log', a })
export const sqrt = (a: Expr): Expr => ({ kind: 'un', op: 'sqrt', a })
export const rsqrt = (a: Expr): Expr => ({ kind: 'un', op: 'rsqrt', a })
export const select = (cmp: CompareOp, a: Expr, b: Expr, yes: Expr, no: Expr): Expr => ({ kind: 'sel', cmp, a, b, yes, no })

/** A compiled kernel and the order of its parameters: buffers (u64), then integers (u32), then floats (f32). */
export interface Program {
	readonly name: string
	readonly text: string
	readonly buffers: readonly string[]
	readonly ints: readonly string[]
	readonly floats: readonly string[]
}

const LOG2E = '0f3FB8AA3B'
const LN2 = '0f3F317218'

function fail(message: string): never {
	throw new Error(message)
}

function need(map: ReadonlyMap<string, string>, key: string, what: string): string {
	return map.get(key) ?? fail(`Unknown ${what} "${key}"`)
}

export function literal(value: number): string {
	const bits = new Uint32Array(new Float32Array([value]).buffer)[0]!
	return `0f${bits.toString(16).toUpperCase().padStart(8, '0')}`
}

interface Scope {
	readonly regs: Registers
	readonly emit: (line: string) => void
	readonly cache: Map<Expr, string>
	readonly arg: (name: string) => string
	readonly load: (node: Expr, position?: string) => string
}

const BINARY = { add: 'add.f32', sub: 'sub.f32', mul: 'mul.f32', div: 'div.rn.f32', max: 'max.f32' } as const

function compile(expr: Expr, scope: Scope): string {
	const cached = scope.cache.get(expr)
	if (cached !== undefined) return cached
	const { regs, emit } = scope
	let reg: string
	switch (expr.kind) {
		case 'num':
			reg = regs.f()
			emit(`mov.f32 ${reg}, ${literal(expr.value)};`)
			break
		case 'arg':
			reg = scope.arg(expr.name)
			break
		case 'at':
		case 'col':
		case 'row':
		case 'index':
			reg = scope.load(expr)
			break
		case 'gather':
			reg = scope.load(expr, compile(expr.position, scope))
			break
		case 'bin': {
			const [a, b] = [compile(expr.a, scope), compile(expr.b, scope)]
			reg = regs.f()
			emit(`${BINARY[expr.op]} ${reg}, ${a}, ${b};`)
			break
		}
		case 'un': {
			const a = compile(expr.a, scope)
			reg = regs.f()
			if (expr.op === 'neg') emit(`neg.f32 ${reg}, ${a};`)
			else if (expr.op === 'sqrt') emit(`sqrt.rn.f32 ${reg}, ${a};`)
			else if (expr.op === 'rsqrt') emit(`rsqrt.approx.f32 ${reg}, ${a};`)
			else if (expr.op === 'exp') {
				const scaled = regs.f()
				emit(`mul.f32 ${scaled}, ${a}, ${LOG2E};`)
				emit(`ex2.approx.ftz.f32 ${reg}, ${scaled};`)
			} else {
				const base2 = regs.f()
				emit(`lg2.approx.ftz.f32 ${base2}, ${a};`)
				emit(`mul.f32 ${reg}, ${base2}, ${LN2};`)
			}
			break
		}
		case 'sel': {
			const [a, b, yes, no] = [compile(expr.a, scope), compile(expr.b, scope), compile(expr.yes, scope), compile(expr.no, scope)]
			const test = regs.p()
			reg = regs.f()
			emit(`setp.${expr.cmp}.f32 ${test}, ${a}, ${b};`)
			emit(`selp.f32 ${reg}, ${yes}, ${no}, ${test};`)
			break
		}
		default: {
			const unknown: never = expr
			return fail(`Unknown expression ${String(unknown)}`)
		}
	}
	scope.cache.set(expr, reg)
	return reg
}

interface Prologue {
	readonly pointers: Map<string, string>
	readonly ints: Map<string, string>
	readonly floats: Map<string, string>
}

function prologue(regs: Registers, emit: (line: string) => void, buffers: readonly string[], ints: readonly string[], floats: readonly string[]): Prologue {
	const result: Prologue = { pointers: new Map(), ints: new Map(), floats: new Map() }
	for (const name of buffers) {
		const reg = regs.rd()
		emit(`ld.param.u64 ${reg}, [p_${name}];`)
		emit(`cvta.to.global.u64 ${reg}, ${reg};`)
		result.pointers.set(name, reg)
	}
	for (const name of ints) {
		const reg = regs.r()
		emit(`ld.param.u32 ${reg}, [p_${name}];`)
		result.ints.set(name, reg)
	}
	for (const name of floats) {
		const reg = regs.f()
		emit(`ld.param.f32 ${reg}, [p_${name}];`)
		result.floats.set(name, reg)
	}
	return result
}

const signature = (buffers: readonly string[], ints: readonly string[], floats: readonly string[]): string[] => [
	...buffers.map((name) => `.param .u64 p_${name}`),
	...ints.map((name) => `.param .u32 p_${name}`),
	...floats.map((name) => `.param .f32 p_${name}`)
]

export interface ElementwiseSpec {
	readonly name: string
	readonly buffers: readonly string[]
	readonly floats: readonly string[]
	readonly outputs: readonly { readonly buffer: string; readonly expr: Expr }[]
}

/** One thread per element: every output expression is evaluated, then stored, so a buffer may be both read and written. */
export function elementwise(spec: ElementwiseSpec): Program {
	const regs = new Registers()
	const body: string[] = []
	const emit = (line: string): void => {
		body.push(`\t${line}`)
	}
	const ints = ['n']
	const { pointers, ints: integers, floats } = prologue(regs, emit, spec.buffers, ints, spec.floats)
	const [cx, dim, tid, position, outside] = [regs.r(), regs.r(), regs.r(), regs.r(), regs.p()]
	emit(`mov.u32 ${cx}, %ctaid.x;`)
	emit(`mov.u32 ${dim}, %ntid.x;`)
	emit(`mov.u32 ${tid}, %tid.x;`)
	emit(`mad.lo.u32 ${position}, ${cx}, ${dim}, ${tid};`)
	emit(`setp.ge.u32 ${outside}, ${position}, ${need(integers, 'n', 'integer')};`)
	emit(`@${outside} bra $L_end;`)
	const scope: Scope = {
		regs,
		emit,
		cache: new Map(),
		arg: (name) => need(floats, name, 'argument'),
		load: (node) => {
			if (node.kind !== 'at') return fail('Elementwise kernels can only read buffers element by element')
			const address = regs.rd()
			const value = regs.f()
			emit(`mad.wide.u32 ${address}, ${position}, 4, ${need(pointers, node.buffer, 'buffer')};`)
			emit(`ld.global.f32 ${value}, [${address}];`)
			return value
		}
	}
	const values = spec.outputs.map((output) => compile(output.expr, scope))
	spec.outputs.forEach((output, slot) => {
		const address = regs.rd()
		emit(`mad.wide.u32 ${address}, ${position}, 4, ${need(pointers, output.buffer, 'buffer')};`)
		emit(`st.global.f32 [${address}], ${values[slot]!};`)
	})
	body.push('$L_end:')
	return { name: spec.name, text: entry(spec.name, signature(spec.buffers, ints, spec.floats), regs, body), buffers: spec.buffers, ints, floats: spec.floats }
}

export interface RowSpec {
	readonly name: string
	readonly buffers: readonly string[]
	readonly floats: readonly string[]
	/** Reductions over the valid columns of a row, in order. Later expressions may use earlier results as `arg(name)`. */
	readonly reductions: readonly { readonly name: string; readonly op: 'sum' | 'max'; readonly expr: Expr }[]
	/** Scalars derived from reductions and arguments, available as `arg(name)`. */
	readonly scalars: readonly { readonly name: string; readonly expr: Expr }[]
	/** One value per row, written to `buffer[row]`. */
	readonly rowOutputs: readonly { readonly buffer: string; readonly expr: Expr }[]
	/** One value per element; columns beyond the valid window receive `fill`. */
	readonly outputs: readonly { readonly buffer: string; readonly expr: Expr; readonly fill: number }[]
}

const BLOCK = 256

/**
 * One 256-thread block per row. The integers `cols` and `window` are launch parameters: with a window the row uses its
 * first `(row mod window) + 1` columns, which is a causal mask over rows laid out as consecutive query positions.
 */
export function rowKernel(spec: RowSpec): Program {
	const regs = new Registers()
	const body: string[] = []
	const emit = (line: string): void => {
		body.push(`\t${line}`)
	}
	let labels = 0
	const nextLabel = (): string => `$L_${labels++}`
	const ints = ['cols', 'window']
	const { pointers, ints: integers, floats } = prologue(regs, emit, spec.buffers, ints, spec.floats)
	const cols = need(integers, 'cols', 'integer')
	const window = need(integers, 'window', 'integer')
	const [row, tid, rowBase] = [regs.r(), regs.r(), regs.r()]
	emit(`mov.u32 ${row}, %ctaid.x;`)
	emit(`mov.u32 ${tid}, %tid.x;`)
	emit(`mul.lo.u32 ${rowBase}, ${row}, ${cols};`)
	const [windowed, divisor, remainder, limit, valid] = [regs.p(), regs.r(), regs.r(), regs.r(), regs.r()]
	emit(`setp.ne.u32 ${windowed}, ${window}, 0;`)
	emit(`selp.u32 ${divisor}, ${window}, 1, ${windowed};`)
	emit(`rem.u32 ${remainder}, ${row}, ${divisor};`)
	emit(`add.u32 ${limit}, ${remainder}, 1;`)
	emit(`selp.u32 ${valid}, ${limit}, ${cols}, ${windowed};`)
	const [shared, slot] = [regs.r(), regs.r()]
	emit(`mov.u32 ${shared}, red;`)
	emit(`mad.lo.u32 ${slot}, ${tid}, 4, ${shared};`)
	const named = new Map(floats)
	const read = (pointer: string, position: string): string => {
		const address = regs.rd()
		const value = regs.f()
		emit(`mad.wide.u32 ${address}, ${position}, 4, ${pointer};`)
		emit(`ld.global.f32 ${value}, [${address}];`)
		return value
	}
	const columnScope = (col: string, position: string): Scope => ({
		regs,
		emit,
		cache: new Map(),
		arg: (name) => need(named, name, 'argument'),
		load: (node, gathered) => {
			if (node.kind === 'at') return read(need(pointers, node.buffer, 'buffer'), position)
			if (node.kind === 'col') return read(need(pointers, node.buffer, 'buffer'), col)
			if (node.kind === 'row') return read(need(pointers, node.buffer, 'buffer'), row)
			if (node.kind === 'index') {
				const value = regs.f()
				emit(`cvt.rn.f32.u32 ${value}, ${node.axis === 'row' ? row : col};`)
				return value
			}
			if (node.kind === 'gather' && gathered !== undefined) {
				const target = regs.r()
				emit(`cvt.rzi.u32.f32 ${target}, ${gathered};`)
				emit(`add.u32 ${target}, ${target}, ${rowBase};`)
				return read(need(pointers, node.buffer, 'buffer'), target)
			}
			return fail('Unsupported read in a row kernel')
		}
	})
	const rowScope = (): Scope => ({
		regs,
		emit,
		cache: new Map(),
		arg: (name) => need(named, name, 'argument'),
		load: (node, gathered) => {
			if (node.kind === 'row') return read(need(pointers, node.buffer, 'buffer'), row)
			if (node.kind === 'index' && node.axis === 'row') {
				const value = regs.f()
				emit(`cvt.rn.f32.u32 ${value}, ${row};`)
				return value
			}
			if (node.kind === 'gather' && gathered !== undefined) {
				const target = regs.r()
				emit(`cvt.rzi.u32.f32 ${target}, ${gathered};`)
				emit(`add.u32 ${target}, ${target}, ${rowBase};`)
				return read(need(pointers, node.buffer, 'buffer'), target)
			}
			return fail('Scalars and row outputs can only read per-row values and gathers')
		}
	})
	for (const reduction of spec.reductions) {
		const [loop, done] = [nextLabel(), nextLabel()]
		const [acc, col, position, stop] = [regs.f(), regs.r(), regs.r(), regs.p()]
		const op = reduction.op === 'sum' ? 'add' : 'max'
		emit(`mov.f32 ${acc}, ${literal(reduction.op === 'sum' ? 0 : -3.4028234663852886e38)};`)
		emit(`mov.u32 ${col}, ${tid};`)
		body.push(`${loop}:`)
		emit(`setp.ge.u32 ${stop}, ${col}, ${valid};`)
		emit(`@${stop} bra ${done};`)
		emit(`add.u32 ${position}, ${rowBase}, ${col};`)
		const value = compile(reduction.expr, columnScope(col, position))
		emit(`${op}.f32 ${acc}, ${acc}, ${value};`)
		emit(`add.u32 ${col}, ${col}, ${BLOCK};`)
		emit(`bra ${loop};`)
		body.push(`${done}:`)
		emit(`st.shared.f32 [${slot}], ${acc};`)
		emit('bar.sync 0;')
		for (let half = BLOCK / 2; half >= 1; half /= 2) {
			const [active, mine, other] = [regs.p(), regs.f(), regs.f()]
			emit(`setp.lt.u32 ${active}, ${tid}, ${half};`)
			emit(`@${active} ld.shared.f32 ${mine}, [${slot}];`)
			emit(`@${active} ld.shared.f32 ${other}, [${slot}+${half * 4}];`)
			emit(`@${active} ${op}.f32 ${mine}, ${mine}, ${other};`)
			emit(`@${active} st.shared.f32 [${slot}], ${mine};`)
			emit('bar.sync 0;')
		}
		const result = regs.f()
		emit(`ld.shared.f32 ${result}, [${shared}];`)
		emit('bar.sync 0;')
		named.set(reduction.name, result)
	}
	for (const scalar of spec.scalars) named.set(scalar.name, compile(scalar.expr, rowScope()))
	if (spec.rowOutputs.length > 0) {
		const first = regs.p()
		emit(`setp.eq.u32 ${first}, ${tid}, 0;`)
		for (const output of spec.rowOutputs) {
			const value = compile(output.expr, rowScope())
			const address = regs.rd()
			emit(`mad.wide.u32 ${address}, ${row}, 4, ${need(pointers, output.buffer, 'buffer')};`)
			emit(`@${first} st.global.f32 [${address}], ${value};`)
		}
	}
	if (spec.outputs.length > 0) {
		const [loop, done] = [nextLabel(), nextLabel()]
		const [col, position, stop, inside] = [regs.r(), regs.r(), regs.p(), regs.p()]
		emit(`mov.u32 ${col}, ${tid};`)
		body.push(`${loop}:`)
		emit(`setp.ge.u32 ${stop}, ${col}, ${cols};`)
		emit(`@${stop} bra ${done};`)
		emit(`add.u32 ${position}, ${rowBase}, ${col};`)
		emit(`setp.lt.u32 ${inside}, ${col}, ${valid};`)
		const scope = columnScope(col, position)
		const values = spec.outputs.map((output) => {
			const value = compile(output.expr, scope)
			const result = regs.f()
			const filler = regs.f()
			emit(`mov.f32 ${filler}, ${literal(output.fill)};`)
			emit(`selp.f32 ${result}, ${value}, ${filler}, ${inside};`)
			return result
		})
		spec.outputs.forEach((output, slotIndex) => {
			const address = regs.rd()
			emit(`mad.wide.u32 ${address}, ${position}, 4, ${need(pointers, output.buffer, 'buffer')};`)
			emit(`st.global.f32 [${address}], ${values[slotIndex]!};`)
		})
		emit(`add.u32 ${col}, ${col}, ${BLOCK};`)
		emit(`bra ${loop};`)
		body.push(`${done}:`)
	}
	return { name: spec.name, text: entry(spec.name, signature(spec.buffers, ints, spec.floats), regs, body, `.shared .align 4 .b8 red[${BLOCK * 4}];`), buffers: spec.buffers, ints, floats: spec.floats }
}

export interface ColumnSpec {
	readonly name: string
	readonly buffers: readonly string[]
	readonly floats: readonly string[]
	/** The term summed over all rows for each column. */
	readonly term: Expr
	/** The buffer receiving `keep * old + sum` per column. */
	readonly output: string
}

/** One thread per column, looping over the rows. The integer parameters are `rows` and `cols`; the float `keep` is 0 or 1. */
export function columnKernel(spec: ColumnSpec): Program {
	const regs = new Registers()
	const body: string[] = []
	const emit = (line: string): void => {
		body.push(`\t${line}`)
	}
	const ints = ['rows', 'cols']
	const floatNames = [...spec.floats, 'keep']
	const { pointers, ints: integers, floats } = prologue(regs, emit, spec.buffers, ints, floatNames)
	const [rows, cols] = [need(integers, 'rows', 'integer'), need(integers, 'cols', 'integer')]
	const [cx, dim, tid, col, outside] = [regs.r(), regs.r(), regs.r(), regs.r(), regs.p()]
	emit(`mov.u32 ${cx}, %ctaid.x;`)
	emit(`mov.u32 ${dim}, %ntid.x;`)
	emit(`mov.u32 ${tid}, %tid.x;`)
	emit(`mad.lo.u32 ${col}, ${cx}, ${dim}, ${tid};`)
	emit(`setp.ge.u32 ${outside}, ${col}, ${cols};`)
	emit(`@${outside} bra $L_end;`)
	const [acc, row, stop, position] = [regs.f(), regs.r(), regs.p(), regs.r()]
	emit(`mov.f32 ${acc}, ${literal(0)};`)
	emit(`mov.u32 ${row}, 0;`)
	body.push('$L_loop:')
	emit(`setp.ge.u32 ${stop}, ${row}, ${rows};`)
	emit(`@${stop} bra $L_done;`)
	emit(`mad.lo.u32 ${position}, ${row}, ${cols}, ${col};`)
	const read = (pointer: string, at: string): string => {
		const address = regs.rd()
		const value = regs.f()
		emit(`mad.wide.u32 ${address}, ${at}, 4, ${pointer};`)
		emit(`ld.global.f32 ${value}, [${address}];`)
		return value
	}
	const term = compile(spec.term, {
		regs,
		emit,
		cache: new Map(),
		arg: (name) => need(floats, name, 'argument'),
		load: (node) => {
			if (node.kind === 'at') return read(need(pointers, node.buffer, 'buffer'), position)
			if (node.kind === 'col') return read(need(pointers, node.buffer, 'buffer'), col)
			if (node.kind === 'row') return read(need(pointers, node.buffer, 'buffer'), row)
			return fail('Unsupported read in a column kernel')
		}
	})
	emit(`add.f32 ${acc}, ${acc}, ${term};`)
	emit(`add.u32 ${row}, ${row}, 1;`)
	emit('bra $L_loop;')
	body.push('$L_done:')
	const [address, old, result] = [regs.rd(), regs.f(), regs.f()]
	emit(`mad.wide.u32 ${address}, ${col}, 4, ${need(pointers, spec.output, 'buffer')};`)
	emit(`ld.global.f32 ${old}, [${address}];`)
	emit(`fma.rn.f32 ${result}, ${need(floats, 'keep', 'argument')}, ${old}, ${acc};`)
	emit(`st.global.f32 [${address}], ${result};`)
	body.push('$L_end:')
	return { name: spec.name, text: entry(spec.name, signature(spec.buffers, ints, floatNames), regs, body), buffers: spec.buffers, ints, floats: floatNames }
}
