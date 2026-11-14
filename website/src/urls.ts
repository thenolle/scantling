/**
 * The path prefix a site is served under, for example `/scantling/`, always ending in a slash.
 */
export function basePathOf(siteUrl: string): string {
	const path = new URL(siteUrl).pathname
	return path.endsWith('/') ? path : `${path}/`
}

/**
 * The URL path of a documentation page. The `index` page lives at the base path itself.
 */
export function routeHref(base: string, path: string): string {
	return path === 'index' ? base : `${base}${path}/`
}

/**
 * Maps a browser pathname back to a documentation page path.
 */
export function pathFromPathname(base: string, pathname: string): string {
	const rest = pathname.startsWith(base) ? pathname.slice(base.length) : pathname.replace(/^\//, '')
	const clean = rest.replace(/index\.html$/, '').replace(/^\/+|\/+$/g, '')
	return clean === '' ? 'index' : clean
}

/**
 * The absolute public URL of a page.
 */
export function absoluteUrl(siteUrl: string, path: string): string {
	return new URL(path === 'index' ? '' : `${path}/`, siteUrl).href
}

/**
 * File name stem of the social image for a page.
 */
export function ogName(path: string): string {
	return path.replace(/\//g, '-')
}
