/** A row-major `[rows, cols]` matrix of signed 8-bit integers with one scale per row. */
export interface QuantizedMatrix {
	readonly rows: number
	readonly cols: number
	readonly data: Int8Array
	readonly scales: Float32Array
}

/**
 * Symmetric per-row quantization: `scale = max|w| / 127` and `q = round(w / scale)`, so every weight is
 * reproduced to within half a step of its row. A row of zeros gets scale 1.
 */
export function quantizeRows(values: Float32Array, rows: number, cols: number): QuantizedMatrix {
	const data = new Int8Array(rows * cols)
	const scales = new Float32Array(rows)
	for (let row = 0; row < rows; row += 1) {
		let largest = 0
		for (let column = 0; column < cols; column += 1) largest = Math.max(largest, Math.abs(values[row * cols + column]!))
		const scale = largest === 0 ? 1 : largest / 127
		scales[row] = scale
		for (let column = 0; column < cols; column += 1) data[row * cols + column] = Math.round(values[row * cols + column]! / scale)
	}
	return { rows, cols, data, scales }
}
