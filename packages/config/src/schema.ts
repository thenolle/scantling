export interface Issue {
	readonly path: string
	readonly message: string
}

export type ParseResult<T> =
	| { readonly ok: true; readonly value: T }
	| { readonly ok: false; readonly issues: readonly Issue[] }

export type SchemaKind = 'text' | 'number' | 'boolean' | 'choice' | 'list' | 'object'

export interface Schema<T> {
	readonly kind: SchemaKind
	readonly parse: (input: unknown, path: string) => ParseResult<T>
	readonly shape?: Readonly<Record<string, Schema<unknown>>>
	readonly fromText?: (text: string) => unknown
}

export type Infer<S> = S extends Schema<infer T> ? T : never

export interface NumberOptions {
	readonly min?: number
	readonly max?: number
	readonly integer?: boolean
}

export interface TextOptions {
	readonly minLength?: number
	readonly maxLength?: number
}

const ok = <T>(value: T): ParseResult<T> => ({ ok: true, value })
const fail = (path: string, message: string): ParseResult<never> => ({ ok: false, issues: [{ path, message }] })
const joinPath = (parent: string, key: string): string => (parent === '' ? key : `${parent}.${key}`)

/**
 * A required string.
 */
export function text(options: TextOptions = {}): Schema<string> {
	return {
		kind: 'text',
		fromText: (value) => value,
		parse: (input, path) => {
			if (input === undefined) return fail(path, 'is required')
			if (typeof input !== 'string') return fail(path, 'must be a string')
			if (options.minLength !== undefined && input.length < options.minLength) return fail(path, `must be at least ${options.minLength} characters`)
			if (options.maxLength !== undefined && input.length > options.maxLength) return fail(path, `must be at most ${options.maxLength} characters`)
			return ok(input)
		}
	}
}

/**
 * A required finite number with optional bounds and an integer constraint.
 */
export function number(options: NumberOptions = {}): Schema<number> {
	return {
		kind: 'number',
		fromText: (value) => (value.trim() === '' ? Number.NaN : Number(value)),
		parse: (input, path) => {
			if (input === undefined) return fail(path, 'is required')
			if (typeof input !== 'number' || !Number.isFinite(input)) return fail(path, 'must be a finite number')
			if (options.integer === true && !Number.isInteger(input)) return fail(path, 'must be an integer')
			if (options.min !== undefined && input < options.min) return fail(path, `must be at least ${options.min}`)
			if (options.max !== undefined && input > options.max) return fail(path, `must be at most ${options.max}`)
			return ok(input)
		}
	}
}

const TRUE_WORDS = new Set(['true', '1', 'yes', 'on'])
const FALSE_WORDS = new Set(['false', '0', 'no', 'off'])

/**
 * A required boolean. Environment variables accept true/false, 1/0, yes/no and on/off.
 */
export function bool(): Schema<boolean> {
	return {
		kind: 'boolean',
		fromText: (value) => {
			const word = value.trim().toLowerCase()
			if (TRUE_WORDS.has(word)) return true
			return FALSE_WORDS.has(word) ? false : value
		},
		parse: (input, path) => {
			if (input === undefined) return fail(path, 'is required')
			return typeof input === 'boolean' ? ok(input) : fail(path, 'must be true or false')
		}
	}
}

/**
 * A required string restricted to a fixed set of values.
 */
export function choice<const T extends readonly string[]>(values: T): Schema<T[number]> {
	return {
		kind: 'choice',
		fromText: (value) => value,
		parse: (input, path) => {
			if (input === undefined) return fail(path, 'is required')
			if (typeof input !== 'string' || !values.includes(input)) return fail(path, `must be one of ${values.join(', ')}`)
			return ok(input as T[number])
		}
	}
}

/**
 * A required array whose items all match `item`.
 */
export function list<T>(item: Schema<T>): Schema<T[]> {
	return {
		kind: 'list',
		parse: (input, path) => {
			if (input === undefined) return fail(path, 'is required')
			if (!Array.isArray(input)) return fail(path, 'must be a list')
			const values: T[] = []
			const issues: Issue[] = []
			input.forEach((entry: unknown, index) => {
				const result = item.parse(entry, `${path}[${index}]`)
				if (result.ok) values.push(result.value)
				else issues.push(...result.issues)
			})
			return issues.length > 0 ? { ok: false, issues } : ok(values)
		}
	}
}

/**
 * An object with a fixed set of keys. A missing object is treated as empty so that
 * defaults of its children apply. Unknown keys are reported.
 */
export function object<S extends Record<string, Schema<unknown>>>(shape: S): Schema<{ [K in keyof S]: Infer<S[K]> }> {
	return {
		kind: 'object',
		shape,
		parse: (input, path) => {
			const source = input === undefined ? {} : input
			if (typeof source !== 'object' || source === null || Array.isArray(source)) return fail(path, 'must be an object')
			const record = source as Record<string, unknown>
			const value: Record<string, unknown> = {}
			const issues: Issue[] = []
			for (const [key, child] of Object.entries(shape)) {
				const result = child.parse(record[key], joinPath(path, key))
				if (result.ok) value[key] = result.value
				else issues.push(...result.issues)
			}
			for (const key of Object.keys(record)) {
				if (!Object.hasOwn(shape, key)) issues.push({ path: joinPath(path, key), message: 'is not a known setting' })
			}
			return issues.length > 0 ? { ok: false, issues } : ok(value as { [K in keyof S]: Infer<S[K]> })
		}
	}
}

/**
 * Uses `fallback` when the input is missing.
 */
export function withDefault<T>(schema: Schema<T>, fallback: T): Schema<T> {
	return { ...schema, parse: (input, path) => (input === undefined ? ok(fallback) : schema.parse(input, path)) }
}

/**
 * Lists every leaf setting of a schema with its path.
 */
export function leaves(schema: Schema<unknown>, prefix: readonly string[] = []): { readonly path: readonly string[]; readonly schema: Schema<unknown> }[] {
	if (schema.shape === undefined) return [{ path: prefix, schema }]
	return Object.entries(schema.shape).flatMap(([key, child]) => leaves(child, [...prefix, key]))
}
