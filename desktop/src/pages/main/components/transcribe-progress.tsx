import { m } from '~/paraglide/messages.js'
import { Spinner } from '~/components/ui/spinner'
import { cn } from '~/lib/style'
import type { Job } from '../hooks/use-transcribe-queue'

/** Waiting for, or in, transcription. */
export function isTranscribing(job: Job) {
	return job.status === 'queued' || job.status === 'running'
}

/** "Transcribing · 42%" once the engine reports progress, "Transcribing..." before that, "Queued" while waiting. */
export function progressLabel(job: Job) {
	if (job.status === 'queued') return m.queued()
	return job.progress > 0 ? m.transcribingPercent({ percent: String(Math.round(job.progress)) }) : m.transcribing()
}

/** Thin bar filled to the job's progress; before the first report it pulses, while queued it stays empty. */
export function ProgressBar({ job, className }: { job: Job; className?: string }) {
	const known = job.status === 'running' && job.progress > 0
	const width = known ? `${Math.max(2, job.progress)}%` : job.status === 'running' ? '100%' : '0%'
	return (
		<div
			role="progressbar"
			aria-label={progressLabel(job)}
			aria-valuemin={0}
			aria-valuemax={100}
			aria-valuenow={known ? Math.round(job.progress) : undefined}
			className={cn('h-[3px] overflow-hidden rounded-full bg-muted-foreground/15', className)}>
			<div className={cn('aurora-bar h-full rounded-full transition-[width] duration-300 ease-out', !known && 'animate-pulse opacity-40')} style={{ width }} />
		</div>
	)
}

/** The progress line above a transcript that is still being made, aligned with its text column. */
export function JobProgressStrip({ job }: { job: Job }) {
	if (!isTranscribing(job)) return null
	return (
		<div role="status" className="mx-auto flex w-full max-w-[86ch] items-center gap-3 px-8 pt-3 text-[12px] xl:max-w-[96ch]">
			{job.status === 'running' && <Spinner className="h-3 w-3 shrink-0 text-muted-foreground" />}
			<span className="shrink-0 font-medium text-foreground tabular-nums">{progressLabel(job)}</span>
			<ProgressBar job={job} className="min-w-0 flex-1" />
		</div>
	)
}

/** The second line of a sidebar row whose transcript is being made: label and bar. */
export function RowProgress({ job }: { job: Job }) {
	return (
		<div className="mt-0.5">
			<div className="flex items-center gap-1.5 truncate text-[11px] text-muted-foreground tabular-nums">
				{job.status === 'running' && <Spinner className="h-3 w-3 shrink-0" />}
				<span className="truncate">{progressLabel(job)}</span>
			</div>
			<ProgressBar job={job} className="mt-1.5" />
		</div>
	)
}
