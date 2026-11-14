/**
 * Returns the element with the given id or throws when it is missing.
 */
export function byId<T extends HTMLElement = HTMLElement>(id: string): T {
	const element = document.getElementById(id)
	if (element === null) throw new Error(`Missing element #${id}`)
	return element as T
}

/**
 * Fetches a text resource, normalising line endings.
 */
export async function fetchText(url: string): Promise<string> {
	const response = await fetch(url, { cache: 'no-cache' })
	if (!response.ok) throw new Error(`${url} responded with ${response.status}`)
	return (await response.text()).replace(/\r\n/g, '\n')
}

/**
 * Reports whether keyboard input is currently going into a form control.
 */
export function isTyping(target: EventTarget | null): boolean {
	return (target instanceof HTMLElement && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)))
}
