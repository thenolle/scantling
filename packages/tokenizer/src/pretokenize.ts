export const PRETOKENIZER_ID = 'code-v1'

/** Pieces longer than this many bytes are split so that merging stays cheap. */
export const MAX_PRETOKEN_BYTES = 1024

const PATTERN = / ?[\p{L}\p{M}_]+| ?\p{N}| ?[^\s\p{L}\p{M}\p{N}_]+|\s+(?!\S)|\s+/gu

/**
 * Splits text into pieces that BPE merges never cross: identifiers and words, single digits,
 * punctuation runs and whitespace runs, each with at most one leading space. The pieces cover
 * the whole input, so joining them reproduces it.
 */
export function* pretokenize(text: string): Generator<string> {
	for (const match of text.matchAll(PATTERN)) yield match[0]
}

/** Splits bytes into chunks of at most `size` bytes. A chunk boundary may fall inside a UTF-8 character. */
export function* byteChunks(bytes: Uint8Array, size = MAX_PRETOKEN_BYTES): Generator<Uint8Array> {
	for (let start = 0; start < bytes.length; start += size) yield bytes.subarray(start, start + size)
}
