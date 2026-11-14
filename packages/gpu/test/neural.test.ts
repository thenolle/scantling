import { expect } from 'bun:test'
import { backward, ops, parameter } from '@scantling/autodiff'
import { Random, Tensor, randomNormal } from '@scantling/tensor'
import { AdamW } from '@scantling/training'
import { CudaDevice, NeuralOps } from '../src/index.ts'
import type { DeviceBuffer } from '../src/index.ts'
import { gpu } from './available.ts'

const random = new Random(77)
const normal = (shape: number[]): Tensor => randomNormal(shape, random)

function setup(): { device: CudaDevice; neural: NeuralOps; put: (values: ArrayLike<number>) => DeviceBuffer; get: (buffer: DeviceBuffer, length: number) => Float32Array } {
	const device = CudaDevice.create()
	return {
		device,
		neural: new NeuralOps(device),
		put: (values) => {
			const data = Float32Array.from(values)
			const buffer = device.alloc(data.length)
			device.upload(buffer, data)
			return buffer
		},
		get: (buffer, length) => {
			const out = new Float32Array(length)
			device.download(buffer, out)
			return out
		}
	}
}

function expectClose(actual: ArrayLike<number>, expected: ArrayLike<number>, label: string, rtol = 1e-4, atol = 1e-5): void {
	expect(actual.length).toBe(expected.length)
	for (let index = 0; index < expected.length; index += 1) {
		expect(Math.abs(actual[index]! - expected[index]!), `${label}[${index}] is ${actual[index]} but should be ${expected[index]}`).toBeLessThanOrEqual(atol + rtol * Math.abs(expected[index]!))
	}
}

gpu('SwiGLU forward and backward match the autograd operations', () => {
	const { device, neural, put, get } = setup()
	const count = 1000
	const [g, u, dh] = [parameter(normal([count])), parameter(normal([count])), normal([count])]
	const h = ops.mul(ops.silu(g), u)
	backward(h, dh)
	const [gate, up, out, dOut, dGate, dUp] = [put(g.value.data), put(u.value.data), device.alloc(count), put(dh.data), device.alloc(count), device.alloc(count)]
	neural.swiglu(gate, up, out, count)
	neural.swigluBackward(dOut, gate, up, dGate, dUp, count)
	expectClose(get(out, count), h.value.data, 'h')
	expectClose(get(dGate, count), g.grad!.data, 'dgate')
	expectClose(get(dUp, count), u.grad!.data, 'dup')
	device.dispose()
})

gpu('the causal softmax and its backward pass match masked autograd softmax', () => {
	const { device, neural, put, get } = setup()
	const [batch, length, scale] = [3, 9, 0.35]
	const x = parameter(normal([batch, length, length]))
	const mask = Tensor.zeros([length, length])
	for (let row = 0; row < length; row += 1) for (let col = row + 1; col < length; col += 1) mask.data[row * length + col] = 1
	const probabilities = ops.softmax(ops.maskedFill(ops.mulScalar(x, scale), mask, -1e9))
	const dy = normal([batch, length, length])
	backward(probabilities, dy)
	const rows = batch * length
	const [input, out, dOut, dx] = [put(x.value.data), device.alloc(rows * length), put(dy.data), device.alloc(rows * length)]
	neural.softmax(input, out, rows, length, scale, length)
	neural.softmaxBackward(dOut, out, dx, rows, length, scale)
	expectClose(get(out, rows * length), probabilities.value.data, 'probabilities')
	expectClose(get(dx, rows * length), x.grad!.data, 'dx')
	device.dispose()
})

gpu('RMSNorm forward, input gradient and weight gradient match autograd', () => {
	const { device, neural, put, get } = setup()
	const [rows, cols, eps] = [37, 300, 1e-5]
	const [x, w, dy] = [parameter(normal([rows, cols])), parameter(normal([cols])), normal([rows, cols])]
	const y = ops.rmsNorm(x, w, eps)
	backward(y, dy)
	const [input, weight, out, inverse, dOut, dx, dWeight] = [put(x.value.data), put(w.value.data), device.alloc(rows * cols), device.alloc(rows), put(dy.data), device.alloc(rows * cols), device.alloc(cols)]
	device.fill(dWeight, 0)
	neural.rmsNorm(input, weight, out, inverse, rows, cols, eps)
	neural.rmsNormBackward(dOut, input, weight, inverse, dx, dWeight, rows, cols, false)
	expectClose(get(out, rows * cols), y.value.data, 'out')
	expectClose(get(dx, rows * cols), x.grad!.data, 'dx')
	expectClose(get(dWeight, cols), w.grad!.data, 'dweight', 1e-4, 1e-4)
	neural.rmsNormBackward(dOut, input, weight, inverse, dx, dWeight, rows, cols, true)
	expectClose(get(dWeight, cols), Array.from(w.grad!.data, (value) => value * 2), 'accumulated dweight', 1e-4, 2e-4)
	device.dispose()
})

gpu('cross-entropy loss and gradient match autograd', () => {
	const { device, neural, put, get } = setup()
	const [rows, vocab] = [20, 700]
	const logits = parameter(normal([rows, vocab]))
	const targets = Array.from({ length: rows }, () => random.nextInt(vocab))
	const loss = ops.crossEntropy(logits, targets)
	backward(loss)
	const [input, target, grad, losses] = [put(logits.value.data), put(targets), device.alloc(rows * vocab), device.alloc(rows)]
	neural.crossEntropy(input, target, grad, losses, rows, vocab)
	const mean = Array.from(get(losses, rows)).reduce((total, value) => total + value, 0) / rows
	expect(Math.abs(mean - loss.value.item())).toBeLessThanOrEqual(1e-4 * Math.abs(loss.value.item()))
	expectClose(get(grad, rows * vocab), logits.grad!.data, 'grad')
	device.dispose()
})

gpu('AdamW follows the CPU optimizer over several steps', () => {
	const { device, neural, put, get } = setup()
	const count = 200
	const variable = parameter(normal([4, 50]))
	const gradient = normal([4, 50])
	const optimizer = new AdamW()
	const [p, g, m, v] = [put(variable.value.data), put(gradient.data), put(new Float32Array(count)), put(new Float32Array(count))]
	for (let step = 1; step <= 3; step += 1) {
		variable.grad = gradient
		optimizer.step([{ name: 'w', variable }], 0.01)
		neural.adamw(p, g, m, v, count, { step, learningRate: 0.01, decayed: true })
		expectClose(get(p, count), variable.value.data, `weights after step ${step}`, 1e-4, 1e-6)
	}
	device.dispose()
})

gpu('the sum of squares works across a long row, and sizes are validated', () => {
	const { device, neural, put, get } = setup()
	const count = 100_000
	const values = normal([count]).data
	const out = device.alloc(3)
	device.fill(out, 0)
	neural.sumSquares(put(values), out, count, 1)
	let expected = 0
	for (const value of values) expected += value * value
	const result = get(out, 3)
	expect(result[0]).toBe(0)
	expect(Math.abs(result[1]! - expected)).toBeLessThanOrEqual(1e-4 * expected)
	const small = device.alloc(10)
	expect(() => neural.softmax(small, small, 10, 10, 1)).toThrow(RangeError)
	expect(() => neural.crossEntropy(small, small, small, small, 10, 10)).toThrow(RangeError)
	device.dispose()
})
