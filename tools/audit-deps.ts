import { resolve } from 'node:path'
import { loadPolicy } from './audit/policy.ts'
import { runAudit, writeBaseline } from './audit/run.ts'
import type { Finding } from './audit/types.ts'

const root = resolve(import.meta.dir, '..')

function format(finding: Finding): string {
  const where = finding.line === undefined ? finding.file : `${finding.file}:${finding.line}`
  return `  ${where}  ${finding.message}`
}

async function main(): Promise<void> {
  if (Bun.argv.includes('--write-baseline')) {
    const count = await writeBaseline(root)
    console.log(`Wrote tools/audit.baseline.json with ${count} packages. Review the diff before committing.`)
    return
  }
  const findings = await runAudit(root, await loadPolicy(root))
  if (findings.length === 0) {
    console.log('Dependency audit passed.')
    return
  }
  const rules = [...new Set(findings.map((finding) => finding.rule))]
  for (const rule of rules) {
    console.error(`\n[${rule}]`)
    for (const finding of findings.filter((item) => item.rule === rule)) console.error(format(finding))
  }
  console.error(`\nDependency audit failed with ${findings.length} finding(s).`)
  process.exitCode = 1
}

await main()
