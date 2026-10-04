// @vitest-environment jsdom

import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { DEFAULT_AI, type AiSettings } from '~/lib/ai'
import type { Job, TranscribeQueue } from './use-transcribe-queue'
import { shouldAutoSummarize, shouldSummarizeForPrivacy, useSummaries } from './use-summaries'

const client = vi.hoisted(() => ({
	ask: vi.fn(),
	stream: vi.fn(),
	ai: undefined as AiSettings | undefined,
}))
vi.mock('~/lib/ai', async (importOriginal) => ({
	...(await importOriginal<typeof import('~/lib/ai')>()),
	createClient: () => client,
}))
vi.mock('~/providers/preference', () => ({ usePreferenceProvider: () => ({ ai: client.ai }) }))
vi.mock('sonner', () => ({ toast: { error: vi.fn() } }))

afterEach(() => {
	cleanup()
	vi.restoreAllMocks()
})
beforeEach(() => {
	vi.clearAllMocks()
	client.ai = { ...DEFAULT_AI, tasks: { ...DEFAULT_AI.tasks, summary: { ...DEFAULT_AI.tasks.summary, enabled: true } } }
	client.stream.mockImplementation(async (_prompt: string, token: (text: string) => void) => {
		token('Derived result')
	})
	vi.spyOn(console, 'error').mockImplementation(() => {})
})

const segments = [{ start: 0, stop: 1, text: 'New transcript' }]

function completed(overrides: Partial<Pick<Job, 'status' | 'hydrated' | 'summary' | 'segments'>> = {}) {
	return { status: 'done', hydrated: false, summary: undefined, segments, ...overrides } as Pick<Job, 'status' | 'hydrated' | 'summary' | 'segments'>
}

describe('automatic summaries', () => {
	it('is opt-in by default', () => {
		expect(DEFAULT_AI.tasks.summary.autoOnFinish).toBe(false)
	})

	it('runs only when a session job transitions to done', () => {
		expect(shouldAutoSummarize(completed(), 'running')).toBe(true)
		expect(shouldAutoSummarize(completed(), undefined)).toBe(false)
		expect(shouldAutoSummarize(completed(), 'done')).toBe(false)
	})

	it('never auto-runs for hydrated, empty, or already summarized jobs', () => {
		expect(shouldAutoSummarize(completed({ hydrated: true }), 'running')).toBe(false)
		expect(shouldAutoSummarize(completed({ segments: [] }), 'running')).toBe(false)
		expect(shouldAutoSummarize(completed({ summary: 'Existing' }), 'running')).toBe(false)
	})
})

describe('summaries for the delete-original-transcript option', () => {
	const recording = (overrides: Partial<Job> = {}) => makeJob({ source: 'record', ...overrides })

	it('covers a finished live recording and a file recording that just completed', () => {
		expect(shouldSummarizeForPrivacy(recording({ hydrated: true }), undefined)).toBe(true)
		expect(shouldSummarizeForPrivacy(recording(), 'running')).toBe(true)
	})

	it('leaves opened projects, other sources and finished work alone', () => {
		expect(shouldSummarizeForPrivacy(recording({ hydrated: true }), 'done')).toBe(false)
		expect(shouldSummarizeForPrivacy(makeJob({ hydrated: true }), undefined)).toBe(false)
		expect(shouldSummarizeForPrivacy(recording({ summary: 'Existing' }), 'running')).toBe(false)
		expect(shouldSummarizeForPrivacy(recording({ transcriptDeleted: true }), 'running')).toBe(false)
	})
})

function makeJob(overrides: Partial<Job> = {}): Job {
	return { id: 'job-1', name: 'Meeting', path: '', status: 'done', progress: 100, segments, ...overrides }
}

function makeQueue(job: Job): TranscribeQueue {
	return {
		jobs: [job],
		setJobSummary: vi.fn(async () => true),
		setJobThread: vi.fn(),
	} as unknown as TranscribeQueue
}

describe('AI output retention', () => {
	it('uses the retained summary as question context after raw text has been deleted', async () => {
		const job = makeJob({ segments: [], transcriptDeleted: true, summary: 'The agreed launch date is October 15.' })
		const queue = makeQueue(job)
		const { result } = renderHook(() => useSummaries(queue))
		await act(async () => {
			expect(await result.current.ask(job, 'What is the launch date?')).toBe(true)
		})

		expect(client.stream.mock.calls[0][0]).toContain(job.summary)
		expect(client.stream.mock.calls[0][0]).not.toContain('"""\n\n"""')
		expect(queue.setJobThread).toHaveBeenCalledExactlyOnceWith('job-1', [{ question: 'What is the launch date?', answer: 'Derived result' }])
	})

	it.each(['empty model output', 'model failure', 'durable save failure'])('does not report a successful summary after %s', async (failure) => {
		const job = makeJob()
		const queue = makeQueue(job)
		if (failure === 'empty model output')
			client.stream.mockImplementationOnce(async (_prompt: string, token: (text: string) => void) => {
				token(' \n ')
			})
		if (failure === 'model failure') client.stream.mockRejectedValueOnce(new Error('connection failed'))
		if (failure === 'durable save failure') vi.mocked(queue.setJobSummary).mockResolvedValueOnce(false)
		const { result } = renderHook(() => useSummaries(queue))
		await act(async () => {
			expect(await result.current.summarize(job)).toBe(false)
		})

		expect(job.segments).toBe(segments)
		expect(job.transcriptDeleted).toBeUndefined()
		if (failure !== 'durable save failure') expect(queue.setJobSummary).not.toHaveBeenCalled()
		expect(result.current.errors['job-1']).toBeTruthy()
		expect(result.current.pending['job-1']).toBeUndefined()
	})

	it('does not restore old history when summary redaction finishes during an in-flight question', async () => {
		const oldThread = [{ question: 'Old private question', answer: 'Old raw quote' }]
		const job = makeJob({ thread: oldThread })
		const queue = makeQueue(job)
		let release!: () => void
		const answering = new Promise<void>((resolve) => {
			release = resolve
		})
		client.stream.mockImplementation(async (prompt: string, token: (text: string) => void) => {
			if (prompt.includes('Question:')) await answering
			token('Derived result')
		})
		const { result, rerender } = renderHook(({ queue }) => useSummaries(queue), { initialProps: { queue } })
		let asked!: Promise<boolean>
		await act(async () => {
			asked = result.current.ask(job, 'New question')
			await Promise.resolve()
		})
		await act(async () => {
			expect(await result.current.summarize(job)).toBe(true)
		})
		rerender({ queue: { ...queue, jobs: [{ ...job, segments: [], summary: 'Derived result', transcriptDeleted: true, thread: undefined }] } })
		await act(async () => {
			release()
			expect(await asked).toBe(false)
		})

		expect(queue.setJobThread).not.toHaveBeenCalled()
		expect(result.current.answerDrafts['job-1']).toBeUndefined()
		expect(result.current.asking['job-1']).toBeUndefined()
	})
})
