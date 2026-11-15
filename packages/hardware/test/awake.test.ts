import { expect, test } from 'bun:test'
import { keepAwake } from '../src/awake.ts'

test('the request can be taken and released twice without harm', () => {
	const lock = keepAwake({ reason: 'test' })
	if (process.platform === 'win32') expect(lock.active).toBe(true)
	else expect(lock.method).toBe('none')
	lock.release()
	lock.release()
})
