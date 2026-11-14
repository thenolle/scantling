import { expect, test } from 'bun:test'
import { encodePng, renderIcon, renderOgImage, wrapText } from '../scripts/png.ts'

const dimensions = (png: Uint8Array): { width: number; height: number } => {
	const view = new DataView(png.buffer, png.byteOffset)
	return { width: view.getUint32(16), height: view.getUint32(20) }
}

test('encodes a valid PNG container', () => {
	const png = encodePng(2, 2, new Uint8Array(16))
	expect([...png.subarray(0, 8)]).toEqual([137, 80, 78, 71, 13, 10, 26, 10])
	expect(dimensions(png)).toEqual({ width: 2, height: 2 })
	expect(String.fromCharCode(...png.subarray(png.length - 8, png.length - 4))).toBe('IEND')
})

test('renders a 1200x630 social image', () => {
	const png = renderOgImage({ brand: 'Scantling', kicker: 'Guide', title: 'Getting started', footer: 'example.github.io/scantling' })
	expect(dimensions(png)).toEqual({ width: 1200, height: 630 })
})

test('renders icons at the requested size', () => {
	expect(dimensions(renderIcon(32, true))).toEqual({ width: 32, height: 32 })
	expect(dimensions(renderIcon(180, false))).toEqual({ width: 180, height: 180 })
})

test('wraps text within the line limit', () => {
	expect(wrapText('Getting started with Scantling', 15, 3)).toEqual(['GETTING STARTED', 'WITH SCANTLING'])
	const long = wrapText('word '.repeat(40), 19, 3)
	expect(long).toHaveLength(3)
	expect(long.every((line) => line.length <= 19)).toBe(true)
	expect(long[2]?.endsWith('...')).toBe(true)
})
