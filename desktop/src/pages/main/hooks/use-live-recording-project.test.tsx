// @vitest-environment jsdom

import { cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { Preference } from '~/providers/preference'
import type { LiveRecordingResult } from '~/pages/home/hooks/use-live-recording'
import type { SaveTranscriptResult } from '~/lib/transcripts-store'
import type { TranscribeQueue } from './use-transcribe-queue'
import { useLiveRecordingProject } from './use-live-recording-project'

const mocks = vi.hoisted(() => ({ save: vi.fn(), notify: vi.fn(), toast: vi.fn(), preference: {} as Preference }))
vi.mock('~/providers/preference', () => ({ usePreferenceProvider: () => mocks.preference }))
vi.mock('~/lib/transcripts-store', () => ({ saveTranscript: mocks.save, notifyTranscriptsChanged: mocks.notify, TRANSCRIPT_VERSION: 1 }))
vi.mock('sonner', () => ({ toast: { error: mocks.toast } }))
vi.mock('~/paraglide/messages.js', () => ({ m: { error: () => 'Error', projectFromRecord: () => 'Recording' } }))

const segments = [{ start: 12, stop: 145, text: 'Transcript with no audio file' }]
const finished: LiveRecordingResult = {
	name: 'Meeting',
	segments,
	language: 'zh',
	modelPath: '/models/captured.audio8.json',
	createdAt: new Date('2026-10-04T18:00:00Z'),
	processingTimeSec: 7,
}
const saved = { recordPath: '/projects/Meeting/transcript.vibe.json', mediaPath: '' }
function deferred<T>() {
	let resolve!: (value: T) => void
	const promise = new Promise<T>((done) => {
		resolve = done
	})
	return { promise, resolve }
}
function setup() {
	const hydrate = vi.fn()
	const transcribeJob = vi.fn()
	const enqueue = vi.fn()
	const onShown = vi.fn()
	const queue = { hydrate, transcribeJob, enqueue } as unknown as TranscribeQueue
	const hook = renderHook(() => useLiveRecordingProject(queue, onShown))
	return { ...hook, hydrate, transcribeJob, enqueue, onShown }
}

beforeEach(() => {
	vi.resetAllMocks()
	mocks.preference = {
		saveTranscripts: true,
		projectsPath: '/projects',
		noAudioStorage: false,
		autoTranscribeAfterRecording: true,
		modelOptions: { lang: 'en' },
		modelPath: '/models/changed-since-capture.bin',
	} as Preference
	mocks.save.mockResolvedValue(saved)
})
afterEach(cleanup)

describe('live transcript-only project completion', () => {
	it('saves only text, hydrates without audio, and never retriggers transcription', async () => {
		const { result, hydrate, transcribeJob, enqueue, onShown } = setup()
		await result.current(finished, () => true)
		expect(mocks.save).toHaveBeenCalledExactlyOnceWith({
			name: 'Record-Meeting',
			sourcePath: '',
			noAudioStorage: true,
			projectsPath: '/projects',
			segments,
			language: 'zh',
			modelPath: finished.modelPath,
			createdAt: finished.createdAt,
		})
		expect(hydrate).toHaveBeenCalledExactlyOnceWith(
			expect.objectContaining({
				name: 'Record-Meeting',
				sourcePath: '',
				language: 'zh',
				modelPath: finished.modelPath,
				segments,
			}),
			saved.recordPath,
			null,
			'record',
		)
		expect(transcribeJob).not.toHaveBeenCalled()
		expect(enqueue).not.toHaveBeenCalled()
		expect(onShown).toHaveBeenCalledOnce()
		expect(mocks.notify).toHaveBeenCalledOnce()
	})

	it('keeps the result visible in RAM and never writes when transcript saving is disabled', async () => {
		mocks.preference.saveTranscripts = false
		const { result, hydrate } = setup()
		await result.current(finished, () => true)
		expect(mocks.save).not.toHaveBeenCalled()
		expect(mocks.notify).not.toHaveBeenCalled()
		expect(hydrate).toHaveBeenCalledWith(expect.objectContaining({ sourcePath: '', segments }), '', null, 'record')
	})

	it.each(['null', 'rejection'])('preserves a visible transcript and shows an error when saving fails (%s)', async (failure) => {
		if (failure === 'null') mocks.save.mockResolvedValueOnce(null)
		else mocks.save.mockRejectedValueOnce(new Error('Disk unavailable'))
		const { result, hydrate, onShown } = setup()
		await result.current(finished, () => true)
		expect(hydrate).toHaveBeenCalledWith(expect.objectContaining({ segments }), '', null, 'record')
		expect(onShown).toHaveBeenCalledOnce()
		expect(mocks.toast).toHaveBeenCalledWith('Error', expect.objectContaining({ description: expect.stringContaining('visible in memory') }))
	})

	it('cannot replace a newer capture or clear its panel when an old disk write completes', async () => {
		const saving = deferred<SaveTranscriptResult | null>()
		mocks.save.mockReturnValueOnce(saving.promise)
		const { result, hydrate, onShown } = setup()
		let current = true
		const completion = result.current(finished, () => current)
		current = false
		saving.resolve(saved)
		await completion
		expect(mocks.notify).toHaveBeenCalledOnce()
		expect(hydrate).not.toHaveBeenCalled()
		expect(onShown).not.toHaveBeenCalled()
		expect(mocks.toast).not.toHaveBeenCalled()
	})
})
