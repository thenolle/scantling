/** Maps a zero-based optimizer step to a learning rate. */
export type LearningRateSchedule = (step: number) => number

export const constantRate = (rate: number): LearningRateSchedule => () => rate

export interface WarmupCosineOptions {
	readonly peak: number
	readonly warmupSteps: number
	readonly totalSteps: number
	/** The floor as a fraction of the peak. Defaults to 0.1. */
	readonly minRatio?: number
}

/**
 * Linear warmup to `peak` over `warmupSteps`, then cosine decay to `peak * minRatio` at `totalSteps`,
 * holding that floor afterwards: `min + (peak - min) * 0.5 * (1 + cos(pi * progress))`.
 */
export function warmupCosine(options: WarmupCosineOptions): LearningRateSchedule {
	const { peak, warmupSteps, totalSteps } = options
	const floor = peak * (options.minRatio ?? 0.1)
	if (!(peak > 0) || warmupSteps < 0 || totalSteps <= warmupSteps) throw new RangeError('Invalid warmup-cosine schedule')
	return (step) => {
		if (step < warmupSteps) return (peak * (step + 1)) / warmupSteps
		const progress = Math.min(1, (step - warmupSteps) / (totalSteps - warmupSteps))
		return floor + (peak - floor) * 0.5 * (1 + Math.cos(Math.PI * progress))
	}
}
