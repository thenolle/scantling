import { existsSync, mkdirSync, rmSync } from 'node:fs'
import { join } from 'node:path'

const args = Bun.argv.slice(2)
const option = (name: string): string | undefined => (args.includes(name) ? args[args.indexOf(name) + 1] : undefined)
const user = option('--user') ?? 'thenolle'
const token = process.env['GITHUB_TOKEN']
const out = option('--out') ?? (token !== undefined && args.includes('--private') ? 'F:/corpus/github' : 'F:/corpus/github-others')

interface Repo { readonly name: string; readonly fork: boolean; readonly private: boolean; readonly size: number; readonly clone_url: string }

async function listRepos(): Promise<Repo[]> {
	const repos: Repo[] = []
	for (let page = 1; ; page += 1) {
		const useToken = token !== undefined && args.includes('--private')
		const url = useToken ? `https://api.github.com/user/repos?affiliation=owner&per_page=100&page=${page}` : `https://api.github.com/users/${user}/repos?per_page=100&page=${page}`
		const response = await fetch(url, { headers: { accept: 'application/vnd.github+json', 'user-agent': 'scantling-dataset', ...(useToken ? { authorization: `Bearer ${token}` } : {}) } })
		if (!response.ok) throw new Error(`GitHub answered ${response.status} for ${url}`)
		const batch = (await response.json()) as Repo[]
		repos.push(...batch)
		if (batch.length < 100) return repos
	}
}

mkdirSync(out, { recursive: true })
const repos = (await listRepos()).filter((repo) => (args.includes('--forks') || !repo.fork) && repo.size > 0)
console.log(`${repos.length} repositories (${repos.filter((repo) => repo.private).length} private)`)
for (const repo of repos) {
	const target = join(out, repo.name)
	if (existsSync(target)) {
		console.log(`skip ${repo.name} (already there)`)
		continue
	}
	const child = Bun.spawn(['git', 'clone', '--depth', '1', '--quiet', repo.clone_url, target], { stdout: 'inherit', stderr: 'inherit' })
	if ((await child.exited) !== 0) console.error(`failed ${repo.name}`)
	else rmSync(join(target, '.git'), { recursive: true, force: true })
}
