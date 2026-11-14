import { beforeAll, expect, test } from 'bun:test'
import { noGrad } from '@scantling/autodiff'
import { Transformer } from '@scantling/nn'
import { allClose, Random, Tensor } from '@scantling/tensor'
import { AdamW, RandomBatchSource, TokenWindows, Trainer, warmupCosine } from '@scantling/training'
import { InferenceSession, StopMatcher, generate, sampleToken } from '../src/index.ts'
import type { GenerateOptions, GenerationEvent } from '../src/index.ts'

const base = { vocabSize: 23, dim: 16, layers: 2, heads: 2, maxSeqLen: 16, initStd: 0.3 }
const ids = [3, 7, 1, 9, 22, 0, 4, 4, 11, 5]

test('cached decoding matches full recomputation, with and without grouped-query attention', () => {
	for (const config of [base, { ...base, kvHeads: 1 }, { ...base, tieEmbeddings: false }]) {
		const model = new Transformer(config, 5)
		const full = noGrad(() => model.forward(ids, 1, ids.length)).value
		const row = (t: Tensor, index: number): Tensor => Tensor.from([23], t.data.subarray(index * 23, (index + 1) * 23))
		const close = (a: Tensor, b: Tensor): boolean => allClose(a, b, { rtol: 1e-4, atol: 1e-5 })
		const whole = new InferenceSession(model).feed(ids, { all: true })
		for (let index = 0; index < ids.length; index += 1) expect(close(row(whole, index), row(full, index))).toBe(true)
		const incremental = new InferenceSession(model)
		const prefill = incremental.feed(ids.slice(0, 6), { all: true })
		for (let index = 0; index < 6; index += 1) expect(close(row(prefill, index), row(full, index))).toBe(true)
		for (let index = 6; index < ids.length; index += 1) expect(close(incremental.feed([ids[index]!]).reshape([23]), row(full, index))).toBe(true)
		expect(incremental.length).toBe(ids.length)
		expect(close(new InferenceSession(model).feed(ids).reshape([23]), row(full, ids.length - 1))).toBe(true)
	}
})

test('a full context window is refused and reset clears the cache', () => {
	const session = new InferenceSession(new Transformer(base, 1))
	session.feed(Array.from({ length: 16 }, (_, index) => index))
	expect(() => session.feed([1])).toThrow(RangeError)
	session.reset()
	expect(session.length).toBe(0)
	expect(() => session.feed([])).toThrow(RangeError)
	expect(session.feed([1]).shape).toEqual([1, 23])
})

const logs = (...probabilities: number[]): Float32Array => Float32Array.from(probabilities.map(Math.log))

test('greedy sampling is deterministic and breaks ties toward the lowest id', () => {
	const random = new Random(1)
	expect(sampleToken(Float32Array.from([1, 5, 5, 2]), [], { temperature: 0 }, random).token).toBe(1)
	expect(random.state()).toEqual(new Random(1).state())
})

test('sampling frequencies follow the probabilities', () => {
	const random = new Random(3)
	const counts = [0, 0, 0]
	for (let draw = 0; draw < 20000; draw += 1) counts[sampleToken(logs(0.7, 0.2, 0.1), [], { temperature: 1 }, random).token]!++
	expect(counts[0]! / 20000).toBeCloseTo(0.7, 1)
	expect(Math.abs(counts[1]! / 20000 - 0.2)).toBeLessThan(0.02)
	expect(Math.abs(counts[2]! / 20000 - 0.1)).toBeLessThan(0.02)
})

test('top-k and top-p restrict the candidates', () => {
	const seen = (options: Parameters<typeof sampleToken>[2], probabilities: number[]): number[] => {
		const random = new Random(9)
		return [...new Set(Array.from({ length: 400 }, () => sampleToken(logs(...probabilities), [], options, random).token))].sort()
	}
	expect(seen({ temperature: 1, topK: 2 }, [0.5, 0.3, 0.15, 0.05])).toEqual([0, 1])
	expect(seen({ temperature: 1, topP: 0.7 }, [0.5, 0.3, 0.15, 0.05])).toEqual([0, 1])
	expect(seen({ temperature: 1, topP: 0.4 }, [0.5, 0.3, 0.15, 0.05])).toEqual([0])
})

test('penalties and bans change what can be produced', () => {
	const random = new Random(1)
	const repeated = sampleToken(Float32Array.from([2, 2]), [0], { temperature: 0, repetitionPenalty: 2 }, random)
	expect(repeated.token).toBe(1)
	expect(sampleToken(Float32Array.from([1, 1]), [0, 0, 0], { temperature: 0, frequencyPenalty: 0.5 }, random).token).toBe(1)
	expect(sampleToken(Float32Array.from([1, 1]), [0], { temperature: 0, presencePenalty: 0.5 }, random).token).toBe(1)
	const banned = Array.from({ length: 200 }, () => sampleToken(Float32Array.from([10, 1, 1]), [], { temperature: 1, banned: [0] }, random).token)
	expect(banned.includes(0)).toBe(false)
	expect(() => sampleToken(Float32Array.from([1, 1]), [], { banned: [0, 1] }, random)).toThrow(RangeError)
})

test('invalid sampling options are rejected and a seed reproduces a sequence', () => {
	const random = new Random(1)
	for (const options of [{ temperature: -1 }, { topK: 1.5 }, { topP: 0 }, { topP: 1.5 }, { repetitionPenalty: 0 }]) expect(() => sampleToken(Float32Array.from([1, 2]), [], options, random)).toThrow(RangeError)
	const draw = (seed: number): number[] => {
		const generator = new Random(seed)
		return Array.from({ length: 30 }, () => sampleToken(logs(0.3, 0.3, 0.2, 0.2), [], { temperature: 1 }, generator).token)
	}
	expect(draw(4)).toEqual(draw(4))
	expect(draw(4)).not.toEqual(draw(5))
})

test('stop sequences are matched across pieces and partial matches are not leaked', () => {
	const split = new StopMatcher(['END'])
	expect(split.push('ab')).toEqual({ emit: 'ab', stopped: false })
	expect(split.push('E')).toEqual({ emit: '', stopped: false })
	expect(split.push('N')).toEqual({ emit: '', stopped: false })
	expect(split.push('D!')).toEqual({ emit: '', stopped: true })
	const diverge = new StopMatcher(['END'])
	expect(diverge.push('xE')).toEqual({ emit: 'x', stopped: false })
	expect(diverge.push('Q')).toEqual({ emit: 'EQ', stopped: false })
	const before = new StopMatcher(['STOP', 'X'])
	expect(before.push('hello STOP')).toEqual({ emit: 'hello ', stopped: true })
	const held = new StopMatcher(['abc'])
	held.push('ab')
	expect(held.flush()).toBe('ab')
	expect(() => new StopMatcher([''])).toThrow(RangeError)
})

const cycle = Array.from({ length: 16 }, (_, index) => (index * 7 + 3) % 32)
const letter = (id: number): string => String.fromCharCode(97 + (id % 26))
const letters = { push: (id: number): string => letter(id), flush: (): string => '' }
let model: Transformer

beforeAll(async () => {
	model = new Transformer({ vocabSize: 32, dim: 32, layers: 2, heads: 2, maxSeqLen: 32 }, 1)
	const windows = new TokenWindows(Int32Array.from({ length: 320 }, (_, index) => cycle[index % 16]!), 16)
	const trainer = new Trainer({ model, optimizer: new AdamW(), schedule: warmupCosine({ peak: 5e-3, warmupSteps: 10, totalSteps: 150 }), maxGradNorm: 1 })
	await trainer.run({ steps: 150, source: new RandomBatchSource(windows, 8, new Random(2)) })
})

async function collect(options: GenerateOptions, prompt = cycle.slice(0, 3)): Promise<{ events: GenerationEvent[]; tokens: number[]; text: string; done: Extract<GenerationEvent, { type: 'done' }> }> {
	const events: GenerationEvent[] = []
	for await (const event of generate(model, prompt, options)) events.push(event)
	const tokens = events.flatMap((event) => (event.type === 'token' ? [event.token] : []))
	const done = events.at(-1) as Extract<GenerationEvent, { type: 'done' }>
	return { events, tokens, text: events.map((event) => (event.type === 'token' ? event.text : event.tail)).join(''), done }
}

test('greedy generation continues the learned sequence and honours the length limit', async () => {
	const result = await collect({ maxNewTokens: 20, sampling: { temperature: 0 } })
	expect(result.tokens).toEqual(Array.from({ length: 20 }, (_, index) => cycle[(3 + index) % 16]!))
	expect(result.done).toMatchObject({ reason: 'length', tokens: 20 })
})

test('stop tokens, stop sequences and cancellation end generation', async () => {
	const byToken = await collect({ maxNewTokens: 20, sampling: { temperature: 0 }, stopTokens: [cycle[5]!] })
	expect(byToken.tokens).toEqual([cycle[3]!, cycle[4]!])
	expect(byToken.done.reason).toBe('stop-token')
	const bySequence = await collect({ maxNewTokens: 20, sampling: { temperature: 0 }, text: letters, stopSequences: [letter(cycle[5]!) + letter(cycle[6]!)] })
	expect(bySequence.text).toBe(letter(cycle[3]!) + letter(cycle[4]!))
	expect(bySequence.done.reason).toBe('stop-sequence')
	const controller = new AbortController()
	const seen: number[] = []
	for await (const event of generate(model, cycle.slice(0, 3), { maxNewTokens: 50, sampling: { temperature: 0 }, signal: controller.signal })) {
		if (event.type === 'token') {
			seen.push(event.token)
			if (seen.length === 2) controller.abort()
		} else {
			expect(event.reason).toBe('cancelled')
		}
	}
	expect(seen).toHaveLength(2)
})

test('the context window slides without losing the pattern, or stops when asked', async () => {
	const sliding = await collect({ maxNewTokens: 100, sampling: { temperature: 0 } })
	expect(sliding.done.reason).toBe('length')
	expect(sliding.tokens).toEqual(Array.from({ length: 100 }, (_, index) => cycle[(3 + index) % 16]!))
	const stopping = await collect({ maxNewTokens: 100, sampling: { temperature: 0 }, contextPolicy: 'stop' })
	expect(stopping.done.reason).toBe('context')
	expect(stopping.tokens.length).toBeLessThan(33)
})

test('traces report candidates and a seed makes sampling reproducible', async () => {
	const traced = await collect({ maxNewTokens: 3, sampling: { temperature: 0 }, trace: 3 })
	const first = traced.events[0]!
	expect(first.type === 'token' && first.candidates.length).toBe(3)
	expect(first.type === 'token' && first.candidates[0]!.token).toBe(cycle[3]!)
	const options = { maxNewTokens: 30, sampling: { temperature: 1.5 }, seed: 11 }
	expect((await collect(options)).tokens).toEqual((await collect(options)).tokens)
})
