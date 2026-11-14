/** Double-precision `op(A) * op(B)` plus the sum of absolute products, which scales a fair tolerance. */
export function referenceMatmul(a: Float32Array, b: Float32Array, m: number, k: number, n: number, transposeA: boolean, transposeB: boolean): { values: Float64Array; magnitude: Float64Array } {
	const values = new Float64Array(m * n)
	const magnitude = new Float64Array(m * n)
	for (let i = 0; i < m; i += 1) {
		for (let j = 0; j < n; j += 1) {
			for (let p = 0; p < k; p += 1) {
				const term = (transposeA ? a[p * m + i]! : a[i * k + p]!) * (transposeB ? b[j * k + p]! : b[p * n + j]!)
				values[i * n + j] = values[i * n + j]! + term
				magnitude[i * n + j] = magnitude[i * n + j]! + Math.abs(term)
			}
		}
	}
	return { values, magnitude }
}
