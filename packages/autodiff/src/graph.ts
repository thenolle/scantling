import * as T from '@scantling/tensor'

/**
 * Computes the gradient contribution for each parent from the gradient of the output.
 * An entry is `null` when that parent needs no gradient.
 */
export type BackwardFn = (grad: T.Tensor) => readonly (T.Tensor | null)[]

/**
 * A value in the computation graph. Leaves have no `backwardFn`; every other node records the
 * parents it was computed from and how to push a gradient back to them.
 */
export class Variable {
	grad: T.Tensor | null = null

	constructor(
		readonly value: T.Tensor,
		readonly requiresGrad: boolean,
		readonly parents: readonly Variable[] = [],
		readonly backwardFn: BackwardFn | undefined = undefined,
		readonly label = ''
	) { }

	get shape(): readonly number[] {
		return this.value.shape
	}

	get isLeaf(): boolean {
		return this.backwardFn === undefined
	}

	/** The same value as a constant that is cut off from the graph. */
	detach(): Variable {
		return new Variable(this.value, false)
	}

	zeroGrad(): void {
		this.grad = null
	}
}

export interface VariableOptions {
	readonly requiresGrad?: boolean
	readonly label?: string
}

export function variable(value: T.Tensor, options: VariableOptions = {}): Variable {
	return new Variable(value, options.requiresGrad === true, [], undefined, options.label ?? '')
}

/** A trainable leaf that collects gradients. */
export function parameter(value: T.Tensor, label = ''): Variable {
	return new Variable(value, true, [], undefined, label)
}

let gradEnabled = true

export function isGradEnabled(): boolean {
	return gradEnabled
}

/**
 * Runs synchronous `work` without recording a graph. Use it for inference and evaluation.
 */
export function noGrad<R>(work: () => R): R {
	const previous = gradEnabled
	gradEnabled = false
	try { return work() } finally { gradEnabled = previous }
}

/**
 * Creates the result of an operation. A graph node is recorded only when gradients are enabled
 * and at least one parent requires them.
 */
export function record(value: T.Tensor, parents: readonly Variable[], backwardFn: BackwardFn, label: string): Variable {
	const tracked = gradEnabled && parents.some((parent) => parent.requiresGrad)
	return tracked ? new Variable(value, true, parents, backwardFn, label) : new Variable(value, false)
}

function topologicalOrder(root: Variable): Variable[] {
	const order: Variable[] = []
	const visited = new Set<Variable>()
	const stack: { node: Variable; expanded: boolean }[] = [{ node: root, expanded: false }]
	while (stack.length > 0) {
		const frame = stack.pop()!
		if (frame.expanded) {
			order.push(frame.node)
			continue
		}
		if (visited.has(frame.node)) continue
		visited.add(frame.node)
		stack.push({ node: frame.node, expanded: true })
		for (const parent of frame.node.parents) {
			if (parent.requiresGrad && !visited.has(parent)) stack.push({ node: parent, expanded: false })
		}
	}
	return order
}

/**
 * Reverse-mode differentiation from `root`. Without a `seed` the root must be a scalar and the
 * seed is 1; with a seed it is the gradient of the final objective with respect to `root`.
 *
 * Gradients add up in the `.grad` of every parameter reached, so calling `backward` several times
 * before an optimizer step accumulates gradients. Call `zeroGrad` between steps.
 */
export function backward(root: Variable, seed?: T.Tensor): void {
	if (!root.requiresGrad) throw new Error('backward() was called on a value that does not require gradients')
	if (seed === undefined && root.value.size !== 1) throw new RangeError(`backward() needs a scalar root or an explicit seed, received shape [${root.shape.join(', ')}]`)
	if (seed !== undefined && !T.shapesEqual(seed.shape, root.shape)) throw new RangeError(`Seed shape [${seed.shape.join(', ')}] differs from root shape [${root.shape.join(', ')}]`)
	const pending = new Map<Variable, T.Tensor>([[root, seed ?? T.Tensor.full(root.shape, 1)]])
	const order = topologicalOrder(root)
	for (let index = order.length - 1; index >= 0; index -= 1) {
		const node = order[index]!
		const grad = pending.get(node)
		if (grad === undefined) continue
		pending.delete(node)
		if (node.backwardFn === undefined) {
			node.grad = node.grad === null ? grad.clone() : T.addInPlace(node.grad, grad)
			continue
		}
		const contributions = node.backwardFn(grad)
		node.parents.forEach((parent, position) => {
			const contribution = contributions[position]
			if (contribution === null || contribution === undefined || !parent.requiresGrad) return
			if (!T.shapesEqual(contribution.shape, parent.shape)) throw new RangeError(`The gradient produced by "${node.label}" has shape [${contribution.shape.join(', ')}] but its input has shape [${parent.shape.join(', ')}]`)
			const existing = pending.get(parent)
			pending.set(parent, existing === undefined ? contribution : T.add(existing, contribution))
		})
	}
}
