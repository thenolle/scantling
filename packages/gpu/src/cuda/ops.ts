import * as E from './expr.ts'
import type { Program } from './expr.ts'

const sigmoid = (x: E.Expr): E.Expr => E.div(E.num(1), E.add(E.num(1), E.exp(E.neg(x))))

const gate = E.at('gate')
const up = E.at('up')
const dOut = E.at('dout')
const s = sigmoid(gate)
const silu = E.mul(gate, s)
const dSilu = E.mul(s, E.add(E.num(1), E.mul(gate, E.sub(E.num(1), s))))

const grad = E.mul(E.at('g'), E.arg('gscale'))
const first = E.add(E.mul(E.arg('beta1'), E.at('m')), E.mul(E.arg('omb1'), grad))
const second = E.add(E.mul(E.arg('beta2'), E.at('v')), E.mul(E.mul(E.arg('omb2'), grad), grad))
const update = E.div(E.mul(first, E.arg('invC1')), E.add(E.sqrt(E.mul(second, E.arg('invC2'))), E.arg('eps')))

const scaled = E.mul(E.arg('scale'), E.at('x'))
const normalised = E.mul(E.at('x'), E.rowAt('inverse'))
const shifted = E.sub(E.at('logits'), E.arg('peak'))

/** Every GPU program of the neural-network operations. Parameter order: buffers, then integers, then floats. */
export const PROGRAMS = {
	swiglu: E.elementwise({ name: 'swiglu', buffers: ['gate', 'up', 'out'], floats: [], outputs: [{ buffer: 'out', expr: E.mul(silu, up) }] }),
	swigluBackward: E.elementwise({
		name: 'swigluBackward',
		buffers: ['dout', 'gate', 'up', 'dgate', 'dup'],
		floats: [],
		outputs: [
			{ buffer: 'dgate', expr: E.mul(E.mul(dOut, up), dSilu) },
			{ buffer: 'dup', expr: E.mul(dOut, silu) }
		]
	}),
	adamw: E.elementwise({
		name: 'adamw',
		buffers: ['p', 'g', 'm', 'v'],
		floats: ['lr', 'beta1', 'beta2', 'omb1', 'omb2', 'eps', 'decay', 'invC1', 'invC2', 'gscale'],
		outputs: [
			{ buffer: 'p', expr: E.sub(E.sub(E.at('p'), E.mul(E.arg('lr'), update)), E.mul(E.arg('decay'), E.at('p'))) },
			{ buffer: 'm', expr: first },
			{ buffer: 'v', expr: second }
		]
	}),
	softmax: E.rowKernel({
		name: 'softmax',
		buffers: ['x', 'out'],
		floats: ['scale'],
		reductions: [
			{ name: 'peak', op: 'max', expr: scaled },
			{ name: 'total', op: 'sum', expr: E.exp(E.sub(scaled, E.arg('peak'))) }
		],
		scalars: [],
		rowOutputs: [],
		outputs: [{ buffer: 'out', expr: E.div(E.exp(E.sub(scaled, E.arg('peak'))), E.arg('total')), fill: 0 }]
	}),
	softmaxBackward: E.rowKernel({
		name: 'softmaxBackward',
		buffers: ['dy', 'y', 'dx'],
		floats: ['scale'],
		reductions: [{ name: 'dot', op: 'sum', expr: E.mul(E.at('dy'), E.at('y')) }],
		scalars: [],
		rowOutputs: [],
		outputs: [{ buffer: 'dx', expr: E.mul(E.arg('scale'), E.mul(E.at('y'), E.sub(E.at('dy'), E.arg('dot')))), fill: 0 }]
	}),
	rmsNorm: E.rowKernel({
		name: 'rmsNorm',
		buffers: ['x', 'w', 'out', 'inverse'],
		floats: ['eps', 'invCols'],
		reductions: [{ name: 'sumsq', op: 'sum', expr: E.mul(E.at('x'), E.at('x')) }],
		scalars: [{ name: 'r', expr: E.rsqrt(E.add(E.mul(E.arg('sumsq'), E.arg('invCols')), E.arg('eps'))) }],
		rowOutputs: [{ buffer: 'inverse', expr: E.arg('r') }],
		outputs: [{ buffer: 'out', expr: E.mul(E.mul(E.at('x'), E.arg('r')), E.colAt('w')), fill: 0 }]
	}),
	rmsNormBackward: E.rowKernel({
		name: 'rmsNormBackward',
		buffers: ['dy', 'x', 'w', 'inverse', 'dx'],
		floats: ['invCols'],
		reductions: [{ name: 'dot', op: 'sum', expr: E.mul(E.mul(E.at('dy'), E.colAt('w')), normalised) }],
		scalars: [{ name: 'mean', expr: E.mul(E.arg('dot'), E.arg('invCols')) }],
		rowOutputs: [],
		outputs: [{ buffer: 'dx', expr: E.mul(E.rowAt('inverse'), E.sub(E.mul(E.at('dy'), E.colAt('w')), E.mul(normalised, E.arg('mean')))), fill: 0 }]
	}),
	rmsNormWeightGrad: E.columnKernel({
		name: 'rmsNormWeightGrad',
		buffers: ['dy', 'x', 'inverse', 'dweight'],
		floats: [],
		term: E.mul(E.at('dy'), normalised),
		output: 'dweight'
	}),
	crossEntropy: E.rowKernel({
		name: 'crossEntropy',
		buffers: ['logits', 'target', 'grad', 'loss'],
		floats: ['scale'],
		reductions: [
			{ name: 'peak', op: 'max', expr: E.at('logits') },
			{ name: 'total', op: 'sum', expr: E.exp(shifted) }
		],
		scalars: [{ name: 'lse', expr: E.add(E.log(E.arg('total')), E.arg('peak')) }],
		rowOutputs: [{ buffer: 'loss', expr: E.sub(E.arg('lse'), E.gather('logits', E.rowAt('target'))) }],
		outputs: [
			{
				buffer: 'grad',
				expr: E.mul(E.arg('scale'), E.sub(E.div(E.exp(shifted), E.arg('total')), E.select('eq', E.index('col'), E.rowAt('target'), E.num(1), E.num(0)))),
				fill: 0
			}
		]
	}),
	sumSquares: E.rowKernel({
		name: 'sumSquares',
		buffers: ['x', 'out'],
		floats: [],
		reductions: [{ name: 'total', op: 'sum', expr: E.mul(E.at('x'), E.at('x')) }],
		scalars: [],
		rowOutputs: [{ buffer: 'out', expr: E.arg('total') }],
		outputs: []
	})
} satisfies Record<string, Program>

export type ProgramName = keyof typeof PROGRAMS
export const PROGRAM_NAMES = Object.keys(PROGRAMS) as ProgramName[]
