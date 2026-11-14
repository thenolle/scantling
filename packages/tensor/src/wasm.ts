/**
 * A minimal WebAssembly encoder, just enough to build a SIMD matrix-multiply kernel without a toolchain.
 * Opcodes follow the WebAssembly core and fixed-width SIMD specifications.
 */

const leb = (value: number): number[] => {
	const out: number[] = []
	let rest = value >>> 0
	do {
		let byte = rest & 0x7f
		rest >>>= 7
		if (rest !== 0) byte |= 0x80
		out.push(byte)
	} while (rest !== 0)
	return out
}

const sleb = (value: number): number[] => {
	const out: number[] = []
	let rest = value
	for (; ;) {
		const byte = rest & 0x7f
		rest >>= 7
		const done = (rest === 0 && (byte & 0x40) === 0) || (rest === -1 && (byte & 0x40) !== 0)
		out.push(done ? byte : byte | 0x80)
		if (done) return out
	}
}

const section = (id: number, content: number[]): number[] => [id, ...leb(content.length), ...content]
const text = (value: string): number[] => [...leb(value.length), ...Array.from(value, (char) => char.charCodeAt(0))]

const I32 = 0x7f
const V128 = 0x7b
const HEADER = [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]

const get = (index: number): number[] => [0x20, index]
const set = (index: number): number[] => [0x21, index]
const i32const = (value: number): number[] => [0x41, ...sleb(value)]
const ADD = [0x6a]
const SUB = [0x6b]
const MUL = [0x6c]
const SHL = [0x74]
const GE_U = [0x4f]
const EQZ = [0x45]
const f32load = (offset: number): number[] => [0x2a, 0x02, ...leb(offset)]
const v128load = (offset: number): number[] => [0xfd, ...leb(0x00), 0x02, ...leb(offset)]
const v128store = (offset: number): number[] => [0xfd, ...leb(0x0b), 0x02, ...leb(offset)]
const v128zero = [0xfd, ...leb(0x0c), ...new Array<number>(16).fill(0)]
const SPLAT = [0xfd, ...leb(0x13)]
const F32X4_ADD = [0xfd, ...leb(228)]
const F32X4_MUL = [0xfd, ...leb(230)]
const BLOCK_LOOP = [0x02, 0x40, 0x03, 0x40]
const BR_IF_OUT = [0x0d, 0x01]
const BR_LOOP = [0x0c, 0x00]
const END = [0x0b]

const A = 0
const B = 1
const C = 2
const M = 3
const K = 4
const N = 5
const I = 6
const J = 7
const P = 8
const AP = 9
const BP = 10
const CP = 11
const KB = 12
const NB = 13
const ACC = (row: number, half: number): number => 14 + row * 2 + half
const B0 = 22
const B1 = 23
const AV = 24

function gemmBody(): number[] {
	const code: number[] = []
	const emit = (...parts: number[][]): void => { for (const part of parts) code.push(...part) }
	const rowAddress = (base: number, step: number, row: number): number[][] => [get(base), ...Array.from({ length: row }, () => [...get(step), ...ADD])]
	emit(get(K), i32const(2), SHL, set(KB), get(N), i32const(2), SHL, set(NB), i32const(0), set(I))
	emit(BLOCK_LOOP, get(I), get(M), GE_U, BR_IF_OUT, i32const(0), set(J))
	emit(BLOCK_LOOP, get(J), get(N), GE_U, BR_IF_OUT)
	for (let row = 0; row < 4; row += 1) for (let half = 0; half < 2; half += 1) emit(v128zero, set(ACC(row, half)))
	emit(get(A), get(I), get(KB), MUL, ADD, set(AP), get(B), get(J), i32const(2), SHL, ADD, set(BP), get(K), set(P))
	emit(BLOCK_LOOP, get(P), EQZ, BR_IF_OUT)
	emit(get(BP), v128load(0), set(B0), get(BP), v128load(16), set(B1))
	for (let row = 0; row < 4; row += 1) {
		emit(...rowAddress(AP, KB, row), f32load(0), SPLAT, set(AV))
		for (let half = 0; half < 2; half += 1) emit(get(ACC(row, half)), get(AV), get(half === 0 ? B0 : B1), F32X4_MUL, F32X4_ADD, set(ACC(row, half)))
	}
	emit(get(AP), i32const(4), ADD, set(AP), get(BP), get(NB), ADD, set(BP), get(P), i32const(1), SUB, set(P), BR_LOOP, END, END)
	emit(get(C), get(I), get(N), MUL, get(J), ADD, i32const(2), SHL, ADD, set(CP))
	for (let row = 0; row < 4; row += 1) {
		emit(...rowAddress(CP, NB, row), get(ACC(row, 0)), v128store(0))
		emit(...rowAddress(CP, NB, row), get(ACC(row, 1)), v128store(16))
	}
	emit(get(J), i32const(8), ADD, set(J), BR_LOOP, END, END)
	emit(get(I), i32const(4), ADD, set(I), BR_LOOP, END, END, END)
	return code
}

/**
 * A module exporting `mem` and `gemm(a, b, c, m, k, n)`, which computes `C = A * B` for row-major float32
 * matrices at the given byte addresses. `m` must be a multiple of 4 and `n` a multiple of 8. Every output
 * element is a sequential float32 sum over `k`, so results do not depend on how rows are grouped.
 */
export function gemmModuleBytes(): Uint8Array {
	const body = gemmBody()
	const locals = [...leb(2), ...leb(8), I32, ...leb(11), V128]
	const type = [...leb(1), 0x60, ...leb(6), ...new Array<number>(6).fill(I32), ...leb(0)]
	const exports = [...leb(2), ...text('mem'), 0x02, 0x00, ...text('gemm'), 0x00, 0x00]
	return Uint8Array.from([
		...HEADER,
		...section(1, type),
		...section(3, [...leb(1), ...leb(0)]),
		...section(5, [...leb(1), 0x00, ...leb(1)]),
		...section(7, exports),
		...section(10, [...leb(1), ...leb(locals.length + body.length), ...locals, ...body])
	])
}

/** A tiny module that uses one SIMD instruction, to tell whether the runtime supports SIMD at all. */
export function simdProbeBytes(): Uint8Array {
	const body = [...leb(0), ...v128zero, ...END]
	return Uint8Array.from([
		...HEADER,
		...section(1, [...leb(1), 0x60, ...leb(0), ...leb(1), V128]),
		...section(3, [...leb(1), ...leb(0)]),
		...section(10, [...leb(1), ...leb(body.length), ...body])
	])
}
