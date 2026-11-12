import { discardPrivateAudio } from '~/lib/privacy'
import type { Job } from './use-transcribe-queue'

/**
 * A run in progress keeps going while the user reads, edits or starts something else: opening a
 * transcript or starting a new one changes the view, not the queue.
 */

/** A transcript opened only to look at, not part of a run. */
export function isViewedOnly(job: Job) {
	return job.opened === true && job.status !== 'queued' && job.status !== 'running'
}

/** Jobs a run is made of: what the file rail lists, leaving out transcripts opened alongside. */
export function runJobs(jobs: Job[]) {
	return jobs.filter((job) => !isViewedOnly(job))
}

/**
 * The jobs that stay when the view moves on: the run's own while one goes on, none otherwise.
 * RAM-only audio of the jobs that go is discarded.
 */
export function keepForNewView(jobs: Job[], running: boolean) {
	const kept = running ? runJobs(jobs) : []
	for (const job of jobs) if (job.privateAudioPath && !kept.includes(job)) void discardPrivateAudio(job.privateAudioPath)
	return kept
}
