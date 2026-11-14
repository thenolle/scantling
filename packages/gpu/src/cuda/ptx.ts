import { Registers, entry } from './emit.ts'

/** PTX target. Forward compatible: the driver compiles it for the installed GPU when the module loads. */
export const PTX_TARGET = 'sm_75'

export const KERNEL_NAMES = ['sgemm_nn', 'sgemm_nt', 'sgemm_tn', 'sgemm_tt', 'add_f32', 'scale_f32'] as const
export type KernelName = (typeof KERNEL_NAMES)[number]

const ZERO = '0f00000000'
const TILE_STRIDE = 65
const TILE_BYTES = 16 * TILE_STRIDE * 4

/**
 * `C = alpha * op(A) * op(B) + beta * C` over a grid of 64x64 tiles; `blockIdx.z` selects the matrix of the
 * batch. Each of the 256 threads accumulates a 4x4 block (rows `ty + 16 i`, columns `tx + 16 j`) over slices
 * of 16 along `k`. Strides are in elements. Loads guard every bound, so any `m`, `n` and `k` work.
 */
export function sgemmKernel(name: string, transposeA: boolean, transposeB: boolean): string {
	const regs = new Registers()
	const body: string[] = []
	const emit = (line: string): void => {
		body.push(`\t${line}`)
	}
	const globalPointer = (param: string): string => {
		const reg = regs.rd()
		emit(`ld.param.u64 ${reg}, [${param}];`)
		emit(`cvta.to.global.u64 ${reg}, ${reg};`)
		return reg
	}
	const load = (type: 'u32' | 's32' | 's64' | 'f32', param: string, reg: string): string => {
		emit(`ld.param.${type} ${reg}, [${param}];`)
		return reg
	}
	const pa = globalPointer('pA')
	const pb = globalPointer('pB')
	const pc = globalPointer('pC')
	const m = load('u32', 'pM', regs.r())
	const n = load('u32', 'pN', regs.r())
	const k = load('u32', 'pK', regs.r())
	const sai = load('s32', 'pSAi', regs.r())
	const sak = load('s32', 'pSAk', regs.r())
	const sbk = load('s32', 'pSBk', regs.r())
	const sbj = load('s32', 'pSBj', regs.r())
	const ldc = load('s32', 'pLdc', regs.r())
	const bsa = load('s64', 'pBsA', regs.rd())
	const bsb = load('s64', 'pBsB', regs.rd())
	const bsc = load('s64', 'pBsC', regs.rd())
	const alpha = load('f32', 'pAlpha', regs.f())
	const beta = load('f32', 'pBeta', regs.f())
	const tid = regs.r()
	const cx = regs.r()
	const cy = regs.r()
	const cz = regs.r()
	emit(`mov.u32 ${tid}, %tid.x;`)
	emit(`mov.u32 ${cx}, %ctaid.x;`)
	emit(`mov.u32 ${cy}, %ctaid.y;`)
	emit(`mov.u32 ${cz}, %ctaid.z;`)
	const tx = regs.r()
	const ty = regs.r()
	const t6 = regs.r()
	const kq = regs.r()
	const bm = regs.r()
	const bn = regs.r()
	emit(`and.b32 ${tx}, ${tid}, 15;`)
	emit(`shr.u32 ${ty}, ${tid}, 4;`)
	emit(`and.b32 ${t6}, ${tid}, 63;`)
	emit(`shr.u32 ${kq}, ${tid}, 6;`)
	emit(`shl.b32 ${bm}, ${cy}, 6;`)
	emit(`shl.b32 ${bn}, ${cx}, 6;`)
	const z = regs.rd()
	const batchOffset = regs.rd()
	emit(`cvt.u64.u32 ${z}, ${cz};`)
	for (const [pointer, stride] of [[pa, bsa], [pb, bsb], [pc, bsc]] as const) {
		emit(`mul.lo.s64 ${batchOffset}, ${stride}, ${z};`)
		emit(`shl.b64 ${batchOffset}, ${batchOffset}, 2;`)
		emit(`add.s64 ${pointer}, ${pointer}, ${batchOffset};`)
	}
	const sm = regs.r()
	const smB = regs.r()
	const aStore = regs.r()
	const bStore = regs.r()
	const aRead = regs.r()
	const bRead = regs.r()
	emit(`mov.u32 ${sm}, smem;`)
	emit(`add.s32 ${smB}, ${sm}, ${TILE_BYTES};`)
	if (!transposeA) {
		emit(`mad.lo.s32 ${aStore}, ${tx}, ${TILE_STRIDE * 4}, ${sm};`)
		emit(`mad.lo.s32 ${aStore}, ${ty}, 4, ${aStore};`)
	} else {
		emit(`mad.lo.s32 ${aStore}, ${kq}, ${TILE_STRIDE * 4}, ${sm};`)
		emit(`mad.lo.s32 ${aStore}, ${t6}, 4, ${aStore};`)
	}
	if (!transposeB) {
		emit(`mad.lo.s32 ${bStore}, ${kq}, ${TILE_STRIDE * 4}, ${smB};`)
		emit(`mad.lo.s32 ${bStore}, ${t6}, 4, ${bStore};`)
	} else {
		emit(`mad.lo.s32 ${bStore}, ${tx}, ${TILE_STRIDE * 4}, ${smB};`)
		emit(`mad.lo.s32 ${bStore}, ${ty}, 4, ${bStore};`)
	}
	emit(`mad.lo.s32 ${aRead}, ${ty}, 4, ${sm};`)
	emit(`mad.lo.s32 ${bRead}, ${tx}, 4, ${smB};`)
	const rowIndex: string[] = []
	const rowValid: string[] = []
	const kOffsetA: string[] = []
	const colIndex: string[] = []
	const colValid: string[] = []
	const kOffsetB: string[] = []
	for (let q = 0; q < 4; q += 1) {
		const row = regs.r()
		const rowOk = regs.p()
		emit(transposeA ? `add.s32 ${row}, ${bm}, ${t6};` : `add.s32 ${row}, ${bm}, ${ty};`)
		if (!transposeA) emit(`add.s32 ${row}, ${row}, ${16 * q};`)
		emit(`setp.lt.u32 ${rowOk}, ${row}, ${m};`)
		rowIndex.push(row)
		rowValid.push(rowOk)
		if (transposeA) {
			const offset = regs.r()
			emit(`add.s32 ${offset}, ${kq}, ${4 * q};`)
			kOffsetA.push(offset)
		} else {
			kOffsetA.push(tx)
		}
		const col = regs.r()
		const colOk = regs.p()
		emit(transposeB ? `add.s32 ${col}, ${bn}, ${ty};` : `add.s32 ${col}, ${bn}, ${t6};`)
		if (transposeB) emit(`add.s32 ${col}, ${col}, ${16 * q};`)
		emit(`setp.lt.u32 ${colOk}, ${col}, ${n};`)
		colIndex.push(col)
		colValid.push(colOk)
		if (!transposeB) {
			const offset = regs.r()
			emit(`add.s32 ${offset}, ${kq}, ${4 * q};`)
			kOffsetB.push(offset)
		} else {
			kOffsetB.push(tx)
		}
	}
	const acc = Array.from({ length: 16 }, () => regs.f())
	for (const reg of acc) emit(`mov.f32 ${reg}, ${ZERO};`)
	const k0 = regs.r()
	const done = regs.p()
	const kidx = regs.r()
	const kOk = regs.p()
	const valid = regs.p()
	const offset = regs.r()
	const address = regs.rd()
	const value = regs.f()
	emit(`mov.u32 ${k0}, 0;`)
	body.push('$L_loop:')
	emit(`setp.ge.u32 ${done}, ${k0}, ${k};`)
	emit(`@${done} bra $L_done;`)
	for (let q = 0; q < 4; q += 1) {
		emit(`add.s32 ${kidx}, ${k0}, ${kOffsetA[q]!};`)
		emit(`setp.lt.u32 ${kOk}, ${kidx}, ${k};`)
		emit(`and.pred ${valid}, ${rowValid[q]!}, ${kOk};`)
		emit(`mul.lo.s32 ${offset}, ${rowIndex[q]!}, ${sai};`)
		emit(`mad.lo.s32 ${offset}, ${kidx}, ${sak}, ${offset};`)
		emit(`mad.wide.s32 ${address}, ${offset}, 4, ${pa};`)
		emit(`mov.f32 ${value}, ${ZERO};`)
		emit(`@${valid} ld.global.f32 ${value}, [${address}];`)
		emit(`st.shared.f32 [${aStore}+${q * (transposeA ? TILE_STRIDE * 16 : 64)}], ${value};`)
	}
	for (let q = 0; q < 4; q += 1) {
		emit(`add.s32 ${kidx}, ${k0}, ${kOffsetB[q]!};`)
		emit(`setp.lt.u32 ${kOk}, ${kidx}, ${k};`)
		emit(`and.pred ${valid}, ${colValid[q]!}, ${kOk};`)
		emit(`mul.lo.s32 ${offset}, ${kidx}, ${sbk};`)
		emit(`mad.lo.s32 ${offset}, ${colIndex[q]!}, ${sbj}, ${offset};`)
		emit(`mad.wide.s32 ${address}, ${offset}, 4, ${pb};`)
		emit(`mov.f32 ${value}, ${ZERO};`)
		emit(`@${valid} ld.global.f32 ${value}, [${address}];`)
		emit(`st.shared.f32 [${bStore}+${q * (transposeB ? 64 : TILE_STRIDE * 16)}], ${value};`)
	}
	emit('bar.sync 0;')
	const av = Array.from({ length: 4 }, () => regs.f())
	const bv = Array.from({ length: 4 }, () => regs.f())
	for (let step = 0; step < 16; step += 1) {
		for (let i = 0; i < 4; i += 1) emit(`ld.shared.f32 ${av[i]!}, [${aRead}+${step * TILE_STRIDE * 4 + i * 64}];`)
		for (let j = 0; j < 4; j += 1) emit(`ld.shared.f32 ${bv[j]!}, [${bRead}+${step * TILE_STRIDE * 4 + j * 64}];`)
		for (let i = 0; i < 4; i += 1) for (let j = 0; j < 4; j += 1) emit(`fma.rn.f32 ${acc[i * 4 + j]!}, ${av[i]!}, ${bv[j]!}, ${acc[i * 4 + j]!};`)
	}
	emit('bar.sync 0;')
	emit(`add.u32 ${k0}, ${k0}, 16;`)
	emit('bra $L_loop;')
	body.push('$L_done:')
	const hasBeta = regs.p()
	const row = regs.r()
	const col = regs.r()
	const rowOkC = regs.p()
	const colOkC = regs.p()
	const inside = regs.p()
	const withBeta = regs.p()
	const index = regs.r()
	const target = regs.rd()
	const result = regs.f()
	const old = regs.f()
	emit(`setp.neu.f32 ${hasBeta}, ${beta}, ${ZERO};`)
	for (let i = 0; i < 4; i += 1) {
		emit(`add.s32 ${row}, ${bm}, ${ty};`)
		emit(`add.s32 ${row}, ${row}, ${16 * i};`)
		emit(`setp.lt.u32 ${rowOkC}, ${row}, ${m};`)
		for (let j = 0; j < 4; j += 1) {
			emit(`add.s32 ${col}, ${bn}, ${tx};`)
			emit(`add.s32 ${col}, ${col}, ${16 * j};`)
			emit(`setp.lt.u32 ${colOkC}, ${col}, ${n};`)
			emit(`and.pred ${inside}, ${rowOkC}, ${colOkC};`)
			emit(`mad.lo.s32 ${index}, ${row}, ${ldc}, ${col};`)
			emit(`mad.wide.s32 ${target}, ${index}, 4, ${pc};`)
			emit(`mul.f32 ${result}, ${acc[i * 4 + j]!}, ${alpha};`)
			emit(`and.pred ${withBeta}, ${inside}, ${hasBeta};`)
			emit(`mov.f32 ${old}, ${ZERO};`)
			emit(`@${withBeta} ld.global.f32 ${old}, [${target}];`)
			emit(`fma.rn.f32 ${result}, ${beta}, ${old}, ${result};`)
			emit(`@${inside} st.global.f32 [${target}], ${result};`)
		}
	}
	const params = [
		'.param .u64 pA', '.param .u64 pB', '.param .u64 pC',
		'.param .u32 pM', '.param .u32 pN', '.param .u32 pK',
		'.param .s32 pSAi', '.param .s32 pSAk', '.param .s32 pSBk', '.param .s32 pSBj', '.param .s32 pLdc',
		'.param .s64 pBsA', '.param .s64 pBsB', '.param .s64 pBsC',
		'.param .f32 pAlpha', '.param .f32 pBeta'
	]
	return entry(name, params, regs, body, `.shared .align 4 .b8 smem[${2 * TILE_BYTES}];`)
}

/** One thread per element: `out[i] = op(x[i], y[i])` for `add_f32`, or `out[i] = x[i] * alpha` for `scale_f32`. */
export function elementwiseKernel(name: 'add_f32' | 'scale_f32'): string {
	const regs = new Registers()
	const body: string[] = []
	const emit = (line: string): void => {
		body.push(`\t${line}`)
	}
	const binary = name === 'add_f32'
	const pointer = (param: string): string => {
		const reg = regs.rd()
		emit(`ld.param.u64 ${reg}, [${param}];`)
		emit(`cvta.to.global.u64 ${reg}, ${reg};`)
		return reg
	}
	const x = pointer('pX')
	const y = binary ? pointer('pY') : ''
	const out = pointer('pOut')
	const count = regs.r()
	emit(`ld.param.u32 ${count}, [pN];`)
	const alpha = regs.f()
	if (!binary) emit(`ld.param.f32 ${alpha}, [pAlpha];`)
	const [cx, dim, tid, index] = [regs.r(), regs.r(), regs.r(), regs.r()]
	const outside = regs.p()
	emit(`mov.u32 ${cx}, %ctaid.x;`)
	emit(`mov.u32 ${dim}, %ntid.x;`)
	emit(`mov.u32 ${tid}, %tid.x;`)
	emit(`mad.lo.s32 ${index}, ${cx}, ${dim}, ${tid};`)
	emit(`setp.ge.u32 ${outside}, ${index}, ${count};`)
	emit(`@${outside} bra $L_end;`)
	const bytes = regs.rd()
	emit(`mul.wide.u32 ${bytes}, ${index}, 4;`)
	const xa = regs.rd()
	const oa = regs.rd()
	const left = regs.f()
	const result = regs.f()
	emit(`add.s64 ${xa}, ${x}, ${bytes};`)
	emit(`add.s64 ${oa}, ${out}, ${bytes};`)
	emit(`ld.global.f32 ${left}, [${xa}];`)
	if (binary) {
		const ya = regs.rd()
		const right = regs.f()
		emit(`add.s64 ${ya}, ${y}, ${bytes};`)
		emit(`ld.global.f32 ${right}, [${ya}];`)
		emit(`add.f32 ${result}, ${left}, ${right};`)
	} else {
		emit(`mul.f32 ${result}, ${left}, ${alpha};`)
	}
	emit(`st.global.f32 [${oa}], ${result};`)
	body.push('$L_end:')
	const params = binary ? ['.param .u64 pX', '.param .u64 pY', '.param .u64 pOut', '.param .u32 pN'] : ['.param .u64 pX', '.param .u64 pOut', '.param .u32 pN', '.param .f32 pAlpha']
	return entry(name, params, regs, body)
}

/** The hand-written kernels, without the module header. */
export function baseKernels(): string[] {
	return [
		sgemmKernel('sgemm_nn', false, false),
		sgemmKernel('sgemm_nt', false, true),
		sgemmKernel('sgemm_tn', true, false),
		sgemmKernel('sgemm_tt', true, true),
		elementwiseKernel('add_f32'),
		elementwiseKernel('scale_f32')
	]
}
