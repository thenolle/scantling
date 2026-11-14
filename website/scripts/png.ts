import { deflateSync } from 'node:zlib'

export type Rgb = readonly [number, number, number]

const CRC_TABLE = (() => {
	const table = new Uint32Array(256)
	for (let n = 0; n < 256; n += 1) {
		let c = n
		for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
		table[n] = c >>> 0
	}
	return table
})()

function crc32(bytes: Uint8Array): number {
	let c = 0xffffffff
	for (const byte of bytes) c = CRC_TABLE[(c ^ byte) & 0xff]! ^ (c >>> 8)
	return (c ^ 0xffffffff) >>> 0
}

function chunk(type: string, data: Uint8Array): Uint8Array {
	const out = new Uint8Array(12 + data.length)
	const view = new DataView(out.buffer)
	view.setUint32(0, data.length)
	for (let index = 0; index < 4; index += 1) out[4 + index] = type.charCodeAt(index)
	out.set(data, 8)
	view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)))
	return out
}

/**
 * Encodes 8-bit RGBA pixels as a PNG file.
 */
export function encodePng(width: number, height: number, rgba: Uint8Array): Uint8Array {
	const stride = width * 4
	const raw = new Uint8Array((stride + 1) * height)
	for (let y = 0; y < height; y += 1) raw.set(rgba.subarray(y * stride, (y + 1) * stride), y * (stride + 1) + 1)
	const header = new Uint8Array(13)
	const view = new DataView(header.buffer)
	view.setUint32(0, width)
	view.setUint32(4, height)
	header[8] = 8
	header[9] = 6
	const parts = [Uint8Array.of(137, 80, 78, 71, 13, 10, 26, 10), chunk('IHDR', header), chunk('IDAT', deflateSync(raw)), chunk('IEND', new Uint8Array(0))]
	const out = new Uint8Array(parts.reduce((total, part) => total + part.length, 0))
	let offset = 0
	for (const part of parts) {
		out.set(part, offset)
		offset += part.length
	}
	return out
}

export class Canvas {
	readonly data: Uint8Array

	constructor(readonly width: number, readonly height: number) {
		this.data = new Uint8Array(width * height * 4)
	}

	blend(x: number, y: number, color: Rgb, alpha: number): void {
		if (alpha <= 0 || x < 0 || y < 0 || x >= this.width || y >= this.height) return
		const index = (y * this.width + x) * 4
		const below = this.data[index + 3]! / 255
		const out = alpha + below * (1 - alpha)
		for (let channel = 0; channel < 3; channel += 1) {
			this.data[index + channel] = Math.round((color[channel]! * alpha + this.data[index + channel]! * below * (1 - alpha)) / out)
		}
		this.data[index + 3] = Math.round(out * 255)
	}

	rect(x: number, y: number, width: number, height: number, color: Rgb): void {
		for (let row = y; row < y + height; row += 1) for (let column = x; column < x + width; column += 1) this.blend(column, row, color, 1)
	}

	gradient(top: Rgb, bottom: Rgb): void {
		for (let row = 0; row < this.height; row += 1) {
			const t = row / (this.height - 1)
			const color: Rgb = [0, 1, 2].map((channel) => Math.round(top[channel]! + (bottom[channel]! - top[channel]!) * t)) as unknown as Rgb
			this.rect(0, row, this.width, 1, color)
		}
	}

	/** Paints the region where `inside` is true, antialiased with 4x4 sampling. */
	coverage(x0: number, y0: number, x1: number, y1: number, inside: (x: number, y: number) => boolean, color: Rgb): void {
		for (let y = Math.floor(y0); y < Math.ceil(y1); y += 1) {
			for (let x = Math.floor(x0); x < Math.ceil(x1); x += 1) {
				let hits = 0
				for (let sy = 0; sy < 4; sy += 1) for (let sx = 0; sx < 4; sx += 1) if (inside(x + (sx + 0.5) / 4, y + (sy + 0.5) / 4)) hits += 1
				this.blend(x, y, color, hits / 16)
			}
		}
	}
}

const FONT: Readonly<Record<string, readonly number[]>> = {
	' ': [0, 0, 0, 0, 0, 0, 0],
	A: [14, 17, 17, 31, 17, 17, 17], B: [30, 17, 17, 30, 17, 17, 30], C: [14, 17, 16, 16, 16, 17, 14],
	D: [30, 17, 17, 17, 17, 17, 30], E: [31, 16, 16, 30, 16, 16, 31], F: [31, 16, 16, 30, 16, 16, 16],
	G: [14, 17, 16, 23, 17, 17, 15], H: [17, 17, 17, 31, 17, 17, 17], I: [14, 4, 4, 4, 4, 4, 14],
	J: [7, 2, 2, 2, 2, 18, 12], K: [17, 18, 20, 24, 20, 18, 17], L: [16, 16, 16, 16, 16, 16, 31],
	M: [17, 27, 21, 21, 17, 17, 17], N: [17, 17, 25, 21, 19, 17, 17], O: [14, 17, 17, 17, 17, 17, 14],
	P: [30, 17, 17, 30, 16, 16, 16], Q: [14, 17, 17, 17, 21, 18, 13], R: [30, 17, 17, 30, 20, 18, 17],
	S: [15, 16, 16, 14, 1, 1, 30], T: [31, 4, 4, 4, 4, 4, 4], U: [17, 17, 17, 17, 17, 17, 14],
	V: [17, 17, 17, 17, 17, 10, 4], W: [17, 17, 17, 21, 21, 21, 10], X: [17, 17, 10, 4, 10, 17, 17],
	Y: [17, 17, 17, 10, 4, 4, 4], Z: [31, 1, 2, 4, 8, 16, 31],
	'0': [14, 17, 19, 21, 25, 17, 14], '1': [4, 12, 4, 4, 4, 4, 14], '2': [14, 17, 1, 2, 4, 8, 31],
	'3': [31, 2, 4, 2, 1, 17, 14], '4': [2, 6, 10, 18, 31, 2, 2], '5': [31, 16, 30, 1, 1, 17, 14],
	'6': [6, 8, 16, 30, 17, 17, 14], '7': [31, 1, 2, 4, 8, 8, 8], '8': [14, 17, 17, 14, 17, 17, 14],
	'9': [14, 17, 17, 15, 1, 2, 12],
	'.': [0, 0, 0, 0, 0, 12, 12], ',': [0, 0, 0, 0, 12, 4, 8], '-': [0, 0, 0, 31, 0, 0, 0],
	':': [0, 12, 12, 0, 12, 12, 0], '/': [1, 1, 2, 4, 8, 16, 16], '|': [4, 4, 4, 4, 4, 4, 4],
	'(': [2, 4, 8, 8, 8, 4, 2], ')': [8, 4, 2, 2, 2, 4, 8], '+': [0, 4, 4, 31, 4, 4, 0],
	'!': [4, 4, 4, 4, 4, 0, 4], '?': [14, 17, 1, 2, 4, 0, 4], '&': [12, 18, 20, 8, 21, 18, 13],
	"'": [4, 4, 8, 0, 0, 0, 0], _: [0, 0, 0, 0, 0, 0, 31]
}

export function normalizeText(text: string): string {
	return [...text.toUpperCase().replace(/[–—−]/g, '-').replace(/[‘’“”"]/g, "'")].map((char) => (FONT[char] === undefined ? '?' : char)).join('')
}

/**
 * Draws uppercase text with a 5x7 bitmap font; each glyph cell is 6 units wide.
 */
export function drawText(canvas: Canvas, text: string, x: number, y: number, scale: number, color: Rgb): void {
	let cursor = x
	for (const char of normalizeText(text)) {
		const rows = FONT[char] ?? FONT['?']!
		rows.forEach((bits, row) => {
			for (let column = 0; column < 5; column += 1) {
				if ((bits >> (4 - column)) & 1) canvas.rect(cursor + column * scale, y + row * scale, scale, scale, color)
			}
		})
		cursor += 6 * scale
	}
}

/**
 * Greedy word wrap. Overflowing text ends with `...` on the last line.
 */
export function wrapText(text: string, maxChars: number, maxLines: number): string[] {
	const lines: string[] = []
	let line = ''
	for (const word of normalizeText(text).split(/\s+/).filter((part) => part !== '')) {
		const next = line === '' ? word : `${line} ${word}`
		if (next.length <= maxChars) line = next
		else {
			if (line !== '') lines.push(line)
			line = word.slice(0, maxChars)
		}
	}
	if (line !== '') lines.push(line)
	if (lines.length <= maxLines) return lines
	const kept = lines.slice(0, maxLines)
	const last = kept[maxLines - 1] ?? ''
	kept[maxLines - 1] = `${last.slice(0, Math.max(0, maxChars - 3))}...`
	return kept
}

function inTriangle(px: number, py: number, p: readonly number[]): boolean {
	const [ax, ay, bx, by, cx, cy] = p as [number, number, number, number, number, number]
	const d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by)
	const d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy)
	const d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay)
	return !((d1 < 0 || d2 < 0 || d3 < 0) && (d1 > 0 || d2 > 0 || d3 > 0))
}

/**
 * Draws the Scantling triangle outline centred on (cx, cy) in a square of `size` pixels.
 */
export function drawLogo(canvas: Canvas, cx: number, cy: number, size: number, color: Rgb): void {
	const scale = size / 24
	const map = (x: number, y: number): [number, number] => [cx + (x - 12) * scale, cy + (y - 11.5) * scale]
	const outer = [...map(12, 4), ...map(21, 19), ...map(3, 19)]
	const [centreX, centreY] = map(12, 14) as [number, number]
	const inner = outer.map((value, index) => (index % 2 === 0 ? centreX + 0.58 * (value - centreX) : centreY + 0.58 * (value - centreY)))
	canvas.coverage(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2, (x, y) => inTriangle(x, y, outer) && !inTriangle(x, y, inner), color)
}

const BACKGROUND: Rgb = [9, 9, 11]
const WHITE: Rgb = [250, 250, 250]
const MUTED: Rgb = [161, 161, 170]
const ACCENT: Rgb = [196, 181, 253]

/**
 * A square app icon. `rounded` adds the corner radius used by browser tabs and the web manifest.
 */
export function renderIcon(size: number, rounded: boolean): Uint8Array {
	const canvas = new Canvas(size, size)
	const radius = rounded ? size * 0.22 : 0
	canvas.coverage(0, 0, size, size, (x, y) => {
		const dx = Math.max(radius - x, 0, x - (size - radius))
		const dy = Math.max(radius - y, 0, y - (size - radius))
		return dx * dx + dy * dy <= radius * radius
	}, BACKGROUND)
	drawLogo(canvas, size / 2, size / 2, size * 0.6, WHITE)
	return encodePng(size, size, canvas.data)
}

export interface OgInput {
	readonly brand: string
	readonly kicker: string
	readonly title: string
	readonly footer: string
}

/**
 * The 1200x630 social preview image for a page.
 */
export function renderOgImage(input: OgInput): Uint8Array {
	const canvas = new Canvas(1200, 630)
	canvas.gradient(BACKGROUND, [24, 24, 27])
	const border: Rgb = [39, 39, 42]
	canvas.rect(24, 24, 1152, 2, border)
	canvas.rect(24, 604, 1152, 2, border)
	canvas.rect(24, 24, 2, 582, border)
	canvas.rect(1174, 24, 2, 582, border)
	drawLogo(canvas, 112, 112, 56, WHITE)
	drawText(canvas, input.brand, 156, 91, 6, WHITE)
	drawText(canvas, input.kicker, 80, 200, 4, MUTED)
	canvas.rect(80, 244, 96, 6, ACCENT)
	wrapText(input.title, 19, 3).forEach((line, index) => drawText(canvas, line, 80, 282 + index * 80, 9, WHITE))
	drawText(canvas, input.footer, 80, 548, 4, MUTED)
	return encodePng(1200, 630, canvas.data)
}
