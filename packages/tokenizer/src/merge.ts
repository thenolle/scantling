/** Token ids must stay below this value so a pair fits in one safe integer key. */
export const PAIR_BASE = 1 << 20

export const pairKey = (left: number, right: number): number => left * PAIR_BASE + right

/**
 * Replaces every non-overlapping occurrence of `left, right` with `merged`, scanning left to right.
 */
export function applyMerge(ids: readonly number[], left: number, right: number, merged: number): number[] {
	const out: number[] = []
	for (let index = 0; index < ids.length; index += 1) {
		if (index < ids.length - 1 && ids[index] === left && ids[index + 1] === right) {
			out.push(merged)
			index += 1
		} else {
			out.push(ids[index]!)
		}
	}
	return out
}
