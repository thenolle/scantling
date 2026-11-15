import { keepAwake } from '../packages/hardware/src/awake'

const args = Bun.argv.slice(2)
const hours = Number(args.includes('--hours') ? args[args.indexOf('--hours') + 1] : 4)
if (!(hours > 0 && hours <= 72)) throw new RangeError('--hours must be between 0 and 72')
const display = args.includes('--display')
const lock = keepAwake({ reason: 'Scantling keep-awake tool', display })
if (!lock.active) {
	console.error('Windows did not accept the request (or this is not Windows).')
	process.exit(1)
}
console.log(`Keeping the system awake for ${hours} h via ${lock.method}${display ? ' with the display on' : '; the display may still turn off'}. Ctrl+C releases it.`)
const stop = (): void => {
	lock.release()
	process.exit(0)
}
process.on('SIGINT', stop)
process.on('SIGTERM', stop)
const deadline = Date.now() + hours * 3_600_000
setInterval(() => {
	if (Date.now() >= deadline) stop()
}, 30_000)
