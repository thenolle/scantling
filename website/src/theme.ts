export type Theme = 'dark' | 'light'

const KEY = 'scantling-theme'

/**
 * Returns the theme currently applied to the document.
 */
export function currentTheme(): Theme {
	return document.documentElement.dataset['theme'] === 'light' ? 'light' : 'dark'
}

/**
 * Applies a theme and remembers the choice when storage is available.
 */
export function applyTheme(theme: Theme): void {
	document.documentElement.dataset['theme'] = theme
	try { localStorage.setItem(KEY, theme) } catch { return }
}

/**
 * Switches between dark and light and returns the new theme.
 */
export function toggleTheme(): Theme {
	const next = currentTheme() === 'dark' ? 'light' : 'dark'
	applyTheme(next)
	return next
}
