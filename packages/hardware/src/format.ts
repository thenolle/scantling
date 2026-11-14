export const MIB = 1024 * 1024
export const GIB = 1024 * MIB

/**
 * Formats a byte count using binary units, for example `7.5 GiB`.
 */
export function formatBytes(bytes: number): string {
	const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
	let value = Math.max(0, bytes)
	let unit = 0
	while (value >= 1024 && unit < units.length - 1) {
		value /= 1024
		unit += 1
	}
	return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit] ?? 'B'}`
}
