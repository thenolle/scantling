import { PROGRAMS } from './ops.ts'
import { PTX_TARGET, baseKernels } from './ptx.ts'

/** The complete PTX module: hand-written kernels plus every compiled program. */
export function moduleSource(): string {
	return [`.version 7.0\n.target ${PTX_TARGET}\n.address_size 64\n`, ...baseKernels(), ...Object.values(PROGRAMS).map((program) => program.text)].join('\n\n')
}
