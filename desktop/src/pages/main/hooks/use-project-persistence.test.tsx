// @vitest-environment jsdom

import { cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { Preference } from '~/providers/preference'
import type { SaveTranscriptResult } from '~/lib/transcripts-store'
import type { Job } from './use-transcribe-queue'

const store = vi.hoisted(() => ({
	saveTranscript: vi.fn(),
	renameTranscript: vi.fn(),
	updateTranscriptSegments: vi.fn(),
	updateTranscriptSpeakerNames: vi.fn(),
	updateTranscriptSummary: vi.fn(),
	updateTranscriptThread: vi.fn(),
	notifyTranscriptsChanged: vi.fn(),
}))
vi.mock('~/lib/transcripts-store', () => store)

import { useProjectPersistence } from './use-project-persistence'

const segments = [{ start: 0, stop: 1, text: 'Private raw transcript', speaker: 0 }]
const thread = [{ question: 'Private question', answer: 'Private answer' }]
const saved = { recordPath: '/projects/Original/transcript.vibe.json', mediaPath: '' }

afterEach(cleanup)
beforeEach(() => {
	vi.resetAllMocks()
	store.saveTranscript.mockResolvedValue(saved)
	store.renameTranscript.mockImplementation(async (_path: string, name: string) => ({
		path: `/projects/${name}/transcript.vibe.json`,
		name,
		createdAt: new Date(0),
		mediaPath: '',
	}))
	store.updateTranscriptSegments.mockResolvedValue(true)
	store.updateTranscriptSpeakerNames.mockResolvedValue(true)
	store.updateTranscriptSummary.mockResolvedValue(true)
	store.updateTranscriptThread.mockResolvedValue(true)
})

function setup(jobOverrides: Partial<Job> = {}, preferenceOverrides: Partial<Preference> = {}) {
	const job: Job = {
		id: 'job-1',
		name: 'Original',
		path: '',
		privateAudioPath: 'vibe-memory:recording-1',
		source: 'record',
		status: 'done',
		progress: 100,
		segments,
		speakerNames: { 0: 'Private speaker' },
		thread,
		...jobOverrides,
	}
	const jobs = { current: [job] }
	const preference = {
		current: {
			saveTranscripts: true,
			noAudioStorage: true,
			deleteTranscriptAfterAi: true,
			projectsPath: '/projects',
			modelOptions: { lang: 'en' },
			modelPath: '/models/model.bin',
			...preferenceOverrides,
		} as Preference,
	}
	const patch = vi.fn((id: string, changes: Partial<Job>) => {
		jobs.current = jobs.current.map((item) => (item.id === id ? { ...item, ...changes } : item))
	})
	const hook = renderHook(() => useProjectPersistence(jobs, preference, patch))
	return { ...hook, jobs, preference, patch, initial: job }
}

function deferred<T>() {
	let resolve!: (value: T) => void
	const promise = new Promise<T>((done) => {
		resolve = done
	})
	return { promise, resolve }
}

describe('project privacy persistence', () => {
	it('saves RAM-recording text without forwarding the private handle as a media path', async () => {
		const { result, jobs, initial } = setup()
		expect(await result.current.persist(initial, segments)).toBe(true)
		expect(store.saveTranscript).toHaveBeenCalledExactlyOnceWith(
			expect.objectContaining({
				sourcePath: '',
				moveSourceMedia: true,
				noAudioStorage: true,
				segments,
			}),
		)
		expect(JSON.stringify(store.saveTranscript.mock.calls)).not.toContain('vibe-memory:')
		expect(jobs.current[0].savedPath).toBe(saved.recordPath)
		expect(jobs.current[0].path).toBe('')
	})

	it('serializes the initial save and summary while preserving a title edited during slow I/O', async () => {
		const gate = deferred<SaveTranscriptResult | null>()
		store.saveTranscript.mockReturnValueOnce(gate.promise)
		const { result, jobs, initial } = setup()
		const saving = result.current.persist(initial, segments)
		await waitFor(() => expect(store.saveTranscript).toHaveBeenCalledTimes(1))
		result.current.previewJobName(initial.id, 'Final title')
		const summarizing = result.current.setJobSummary(initial.id, 'Derived result')
		expect(store.updateTranscriptSummary).not.toHaveBeenCalled()
		gate.resolve(saved)

		expect(await saving).toBe(true)
		expect(await summarizing).toBe(true)
		expect(store.saveTranscript).toHaveBeenCalledTimes(1)
		expect(store.renameTranscript).toHaveBeenCalledExactlyOnceWith(saved.recordPath, 'Final title')
		expect(store.updateTranscriptSummary).toHaveBeenCalledExactlyOnceWith('/projects/Final title/transcript.vibe.json', 'Derived result', true)
		expect(jobs.current[0]).toMatchObject({
			name: 'Final title',
			savedPath: '/projects/Final title/transcript.vibe.json',
			path: '',
			summary: 'Derived result',
			transcriptDeleted: true,
			segments: [],
		})
		expect(jobs.current[0].speakerNames).toBeUndefined()
		expect(jobs.current[0].thread).toBeUndefined()
	})

	it.each(['initial save', 'atomic summary replacement'])('keeps raw RAM state when the %s fails', async (stage) => {
		if (stage === 'initial save') store.saveTranscript.mockResolvedValueOnce(null)
		else store.updateTranscriptSummary.mockResolvedValueOnce(false)
		const { result, jobs } = setup(stage === 'initial save' ? {} : { savedPath: saved.recordPath })

		expect(await result.current.setJobSummary('job-1', 'Derived result')).toBe(false)
		expect(jobs.current[0]).toMatchObject({ segments, speakerNames: { 0: 'Private speaker' }, thread })
		expect(jobs.current[0].summary).toBeUndefined()
		expect(jobs.current[0].transcriptDeleted).toBeUndefined()
	})

	it('refuses an empty derived result before any save or redaction', async () => {
		const { result, jobs } = setup()
		expect(await result.current.setJobSummary('job-1', ' \n ')).toBe(false)
		expect(store.saveTranscript).not.toHaveBeenCalled()
		expect(store.updateTranscriptSummary).not.toHaveBeenCalled()
		expect(jobs.current[0].segments).toBe(segments)
	})

	it('redacts an intentionally RAM-only session without creating durable metadata', async () => {
		const { result, jobs, initial } = setup({}, { saveTranscripts: false })
		expect(await result.current.persist(initial, segments)).toBe(true)
		expect(await result.current.setJobSummary('job-1', 'Derived result')).toBe(true)
		expect(store.saveTranscript).not.toHaveBeenCalled()
		expect(store.updateTranscriptSummary).not.toHaveBeenCalled()
		expect(jobs.current[0]).toMatchObject({ summary: 'Derived result', segments: [], transcriptDeleted: true })
		expect(jobs.current[0].savedPath).toBeUndefined()
		expect(jobs.current[0].path).toBe('')
		expect(jobs.current[0].speakerNames).toBeUndefined()
		expect(jobs.current[0].thread).toBeUndefined()
	})

	it('keeps raw text when the delete-after-AI preference is off', async () => {
		const { result, jobs } = setup({ savedPath: saved.recordPath }, { deleteTranscriptAfterAi: false })
		expect(await result.current.setJobSummary('job-1', 'Derived result')).toBe(true)
		expect(store.updateTranscriptSummary).toHaveBeenCalledExactlyOnceWith(saved.recordPath, 'Derived result', false)
		expect(jobs.current[0]).toMatchObject({ segments, speakerNames: { 0: 'Private speaker' }, thread, summary: 'Derived result' })
		expect(jobs.current[0].transcriptDeleted).toBeUndefined()
	})

	it('drops queued raw edits and old history after a successful redaction', async () => {
		const gate = deferred<boolean>()
		store.updateTranscriptSummary.mockReturnValueOnce(gate.promise)
		const { result, jobs, initial } = setup({ savedPath: saved.recordPath })
		const summarizing = result.current.setJobSummary('job-1', 'Derived result')
		await waitFor(() => expect(store.updateTranscriptSummary).toHaveBeenCalledTimes(1))
		result.current.updateSegmentText('job-1', 0, 'Late raw edit')
		result.current.setSpeakerName('job-1', 0, 'Late speaker')
		result.current.setJobThread('job-1', [...thread, { question: 'Late question', answer: 'Raw answer' }])
		gate.resolve(true)
		expect(await summarizing).toBe(true)
		// Queued persist is also a barrier for the fire-and-forget edit writes above.
		expect(await result.current.persist(initial, segments)).toBe(false)

		expect(store.updateTranscriptSegments).not.toHaveBeenCalled()
		expect(store.updateTranscriptSpeakerNames).not.toHaveBeenCalled()
		expect(store.updateTranscriptThread).not.toHaveBeenCalled()
		expect(jobs.current[0]).toMatchObject({ summary: 'Derived result', segments: [], transcriptDeleted: true })
		expect(jobs.current[0].speakerNames).toBeUndefined()
		expect(jobs.current[0].thread).toBeUndefined()
	})
})
