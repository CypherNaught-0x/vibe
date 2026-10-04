// @vitest-environment jsdom

import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { Preference } from '~/providers/preference'
import type { TranscribeQueue } from './use-transcribe-queue'

const mocks = vi.hoisted(() => ({
	invoke: vi.fn<(command: string, args?: Record<string, unknown>) => Promise<unknown>>(),
	emit: vi.fn(),
	listen: vi.fn(),
	preference: {} as Preference,
	errorModal: vi.fn(),
	toast: { error: vi.fn(), warning: vi.fn(), success: vi.fn() },
	started: vi.fn(),
	cancelled: vi.fn(),
	failed: vi.fn(),
	succeeded: vi.fn(),
	gpuOutOfMemory: vi.fn(),
	rememberGpuOutOfMemory: vi.fn(),
	startKeepAwake: vi.fn(),
	stopKeepAwake: vi.fn(),
	exportTranscript: vi.fn(),
	saveTranscript: vi.fn(),
	renameTranscript: vi.fn(),
	updateTranscriptSegments: vi.fn(),
	updateTranscriptSpeakerNames: vi.fn(),
	updateTranscriptSummary: vi.fn(),
	updateTranscriptThread: vi.fn(),
	notifyTranscriptsChanged: vi.fn(),
}))
vi.mock('@tauri-apps/api/core', () => ({ invoke: mocks.invoke }))
vi.mock('@tauri-apps/api/event', () => ({ emit: mocks.emit, listen: mocks.listen }))
vi.mock('~/providers/preference', () => ({ usePreferenceProvider: () => mocks.preference }))
vi.mock('~/providers/error-modal', async () => ({
	ErrorModalContext: (await import('react')).createContext({ setState: mocks.errorModal }),
}))
vi.mock('sonner', () => ({ toast: mocks.toast }))
// The diarization model counts as installed, so setup goes straight on to the shared options.
vi.mock('~/providers/model-gates', () => ({ useModelGates: () => ({ ensureDiarizeModel: async () => true }) }))
vi.mock('~/lib/analytics', () => ({
	trackTranscribeStarted: mocks.started,
	trackTranscribeCancelled: mocks.cancelled,
	trackTranscribeFailed: mocks.failed,
	trackTranscribeSucceeded: mocks.succeeded,
	trackGpuOutOfMemory: mocks.gpuOutOfMemory,
}))
vi.mock('~/lib/keep-awake', () => ({ KEEP_AWAKE: { queue: 'queue' }, startKeepAwake: mocks.startKeepAwake, stopKeepAwake: mocks.stopKeepAwake }))
vi.mock('~/lib/gpu-memory', () => ({ gpuOutOfMemoryBefore: () => false, rememberGpuOutOfMemory: mocks.rememberGpuOutOfMemory }))
vi.mock('~/lib/notify', () => ({ notify: vi.fn() }))
vi.mock('~/lib/app', () => ({ openPath: vi.fn() }))
vi.mock('~/lib/auto-export', () => ({
	exportTranscript: mocks.exportTranscript,
	tauriExportIo: vi.fn(),
	sharedFolder: () => null,
	summarizeResults: () => ({ exported: 0, skipped: 0, fallback: 0, failed: 0 }),
}))
vi.mock('~/lib/transcripts-store', () => ({
	saveTranscript: mocks.saveTranscript,
	renameTranscript: mocks.renameTranscript,
	updateTranscriptSegments: mocks.updateTranscriptSegments,
	updateTranscriptSpeakerNames: mocks.updateTranscriptSpeakerNames,
	updateTranscriptSummary: mocks.updateTranscriptSummary,
	updateTranscriptThread: mocks.updateTranscriptThread,
	notifyTranscriptsChanged: mocks.notifyTranscriptsChanged,
}))

import { reconcileProjectName, useTranscribeQueue } from './use-transcribe-queue'

const segments = [{ start: 0, stop: 1, text: 'Decoded audio' }]
const files = [{ name: 'First.wav', path: '/media/first.wav' }, { name: 'Second.wav', path: '/media/second.wav' }]
const recordings = [
	{ name: 'First.wav', path: 'vibe-memory:first', source: 'record' as const },
	{ name: 'Second.wav', path: 'vibe-memory:second', source: 'record' as const },
]

afterEach(() => {
	cleanup()
	vi.restoreAllMocks()
})
beforeEach(() => {
	vi.resetAllMocks()
	mocks.preference = {
		modelPath: '/models/test.audio8.json', gpuDevice: -1, noGpu: false, unloadTimeoutMinutes: 5,
		modelOptions: { lang: 'en' }, modelMetadata: null, stableTimestampsEnabled: false, diarizeEnabled: false,
		noAudioStorage: true, saveTranscripts: true, projectsPath: '/projects', autoExport: { enabled: false },
		soundOnFinish: false, focusOnFinish: false,
	} as Preference
	mocks.listen.mockResolvedValue(vi.fn())
	mocks.emit.mockResolvedValue(undefined)
	mocks.invoke.mockImplementation(async (command) => {
		if (command === 'load_model') return 'loaded'
		if (command === 'get_models_folder') return '/models'
		if (command === 'transcribe') return { segments }
		if (command === 'discard_private_audio') return
		throw new Error(`Unexpected command: ${command}`)
	})
	mocks.saveTranscript.mockImplementation(async ({ name, sourcePath }: { name: string; sourcePath: string }) => ({
		recordPath: `/projects/${name}/transcript.vibe.json`, mediaPath: sourcePath,
	}))
})

function deferred<T>() {
	let resolve!: (value: T) => void
	let reject!: (error: unknown) => void
	const promise = new Promise<T>((done, fail) => { resolve = done; reject = fail })
	return { promise, resolve, reject }
}

function calls(command: string) { return mocks.invoke.mock.calls.filter(([name]) => name === command) }
function transcribedPaths() { return calls('transcribe').map(([, args]) => (args?.options as { path: string }).path) }
async function finished(result: { current: TranscribeQueue }) {
	await waitFor(() => expect(result.current.running).toBe(false))
	if (result.current.jobs.length > 1 && result.current.jobs.some((job) => job.status === 'done')) {
		await waitFor(() => expect(result.current.batchSummary).not.toBeNull())
	}
}

function deferSetup(stage: 'model loading' | 'shared-option setup') {
	const gate = deferred<string>()
	const command = stage === 'model loading' ? 'load_model' : 'get_models_folder'
	if (stage === 'shared-option setup') mocks.preference.diarizeEnabled = true
	const normal = mocks.invoke.getMockImplementation()!
	mocks.invoke.mockImplementation((name, args) => name === command ? gate.promise : normal(name, args))
	return { gate, command }
}

describe('project name reconciliation', () => {
	it('keeps reconciling when the live title changes during a slow rename', async () => {
		let liveName = 'Second draft'
		const rename = vi.fn(async (_path: string, name: string) => {
			if (name === 'Second draft') liveName = 'Final title'
			return {
				path: `/projects/${name}/transcript.vibe.json`,
				mediaPath: `/projects/${name}/audio.wav`,
				name,
				createdAt: new Date(0),
			}
		})

		const result = await reconcileProjectName(
			'Original',
			{ recordPath: '/projects/Original/transcript.vibe.json', mediaPath: '/projects/Original/audio.wav' },
			() => liveName,
			rename,
		)

		expect(rename.mock.calls.map(([, name]) => name)).toEqual(['Second draft', 'Final title'])
		expect(result).toEqual({
			recordPath: '/projects/Final title/transcript.vibe.json',
			mediaPath: '/projects/Final title/audio.wav',
		})
	})
})

describe('queue cancellation during setup', () => {
	it.each(['model loading', 'shared-option setup'] as const)('cancels every pending private handle during %s without starting inference', async (stage) => {
		const { gate, command } = deferSetup(stage)
		const { result } = renderHook(useTranscribeQueue)
		act(() => result.current.enqueue(recordings))
		await waitFor(() => expect(calls(command)).toHaveLength(1))
		expect(result.current.activeId).toBeNull()
		act(() => result.current.cancelAll())
		expect(result.current.jobs.map((job) => job.status)).toEqual(['cancelled', 'cancelled'])
		expect(result.current.jobs.every((job) => job.privateAudioPath === undefined)).toBe(true)
		// Files arriving while the stopped setup unwinds must also be released.
		act(() => result.current.enqueue([{ name: 'Late.wav', path: 'vibe-memory:late', source: 'record' }]))
		await act(async () => { gate.resolve(stage === 'model loading' ? 'loaded' : '/models') })
		await finished(result)
		expect(result.current.jobs.map((job) => job.status)).toEqual(['cancelled', 'cancelled', 'cancelled'])
		expect(result.current.jobs.every((job) => job.privateAudioPath === undefined)).toBe(true)
		expect(calls('discard_private_audio').map(([, args]) => args?.path)).toEqual(['vibe-memory:first', 'vibe-memory:second', 'vibe-memory:late'])
		expect(calls('transcribe')).toHaveLength(0)
		expect(mocks.saveTranscript).not.toHaveBeenCalled()
		expect(mocks.errorModal).not.toHaveBeenCalled()
		expect(result.current.isAborting).toBe(false)
	})

	it.each(['model loading', 'shared-option setup'] as const)('cancels only the first job during %s and preserves the later queue', async (stage) => {
		const { gate, command } = deferSetup(stage)
		const { result } = renderHook(useTranscribeQueue)
		act(() => result.current.enqueue(recordings))
		await waitFor(() => expect(calls(command)).toHaveLength(1))
		act(() => { result.current.cancelCurrent(); result.current.cancelCurrent() })
		expect(result.current.jobs.map((job) => job.status)).toEqual(['cancelled', 'queued'])
		expect(result.current.jobs[0].privateAudioPath).toBeUndefined()
		expect(result.current.jobs[1].privateAudioPath).toBe('vibe-memory:second')
		expect(mocks.emit).not.toHaveBeenCalled()
		await act(async () => { gate.resolve(stage === 'model loading' ? 'loaded' : '/models') })
		await finished(result)
		expect(transcribedPaths()).toEqual(['vibe-memory:second'])
		expect(result.current.jobs.map((job) => job.status)).toEqual(['cancelled', 'done'])
		expect(result.current.jobs.every((job) => job.privateAudioPath === undefined)).toBe(true)
		expect(calls('discard_private_audio').map(([, args]) => args?.path)).toEqual(['vibe-memory:first', 'vibe-memory:second'])
		expect(mocks.saveTranscript).toHaveBeenCalledTimes(1)
		expect(mocks.saveTranscript.mock.calls[0][0]).toMatchObject({ sourcePath: '', segments })
		expect(mocks.cancelled).toHaveBeenCalledTimes(1)
	})

	it.each(['resolves', 'rejects'] as const)('never invokes transcription or reports a load error after the only job is cancelled and loading %s', async (outcome) => {
		const { gate } = deferSetup('model loading')
		const { result } = renderHook(useTranscribeQueue)
		act(() => result.current.enqueue([recordings[0]]))
		act(() => result.current.cancelCurrent())
		await act(async () => {
			if (outcome === 'resolves') gate.resolve('loaded')
			else gate.reject(new Error('startup failed after cancellation'))
		})
		await finished(result)
		expect(result.current.jobs[0].status).toBe('cancelled')
		expect(calls('transcribe')).toHaveLength(0)
		expect(calls('get_models_folder')).toHaveLength(0)
		expect(calls('discard_private_audio')).toHaveLength(1)
		expect(mocks.errorModal).not.toHaveBeenCalled()
		expect(mocks.toast.error).not.toHaveBeenCalled()
	})
})

describe('queue run behaviour', () => {
	it('loads once and transcribes a normal batch in order', async () => {
		const { gate } = deferSetup('model loading')
		const { result } = renderHook(useTranscribeQueue)
		act(() => result.current.enqueue(files))
		expect(calls('transcribe')).toHaveLength(0)
		await act(async () => { gate.resolve('loaded') })
		await finished(result)
		expect(calls('load_model')).toHaveLength(1)
		expect(transcribedPaths()).toEqual(files.map((file) => file.path))
		expect(result.current.jobs.map((job) => job.status)).toEqual(['done', 'done'])
		expect(mocks.saveTranscript).toHaveBeenCalledTimes(2)
		expect(result.current.batchSummary?.total).toBe(2)
		expect(mocks.stopKeepAwake).toHaveBeenCalledTimes(1)
	})

	it('still retries GPU exhaustion on the CPU before continuing the batch', async () => {
		let attempts = 0
		const normal = mocks.invoke.getMockImplementation()!
		mocks.invoke.mockImplementation((command, args) => {
			if (command === 'transcribe' && attempts++ === 0) return Promise.reject({ code: 'gpu_out_of_memory', message: 'GPU allocation failed' })
			return normal(command, args)
		})
		const { result } = renderHook(useTranscribeQueue)
		act(() => result.current.enqueue(files))
		await finished(result)
		expect(transcribedPaths()).toEqual([files[0].path, files[0].path, files[1].path])
		expect(calls('load_model').map(([, args]) => args?.noGpu)).toEqual([false, true])
		expect(result.current.jobs.map((job) => job.status)).toEqual(['done', 'done'])
		expect(mocks.rememberGpuOutOfMemory).toHaveBeenCalledWith(mocks.preference.modelPath)
	})

	it.each([
		{ action: 'current', reloads: true }, { action: 'current', reloads: false },
		{ action: 'all', reloads: true }, { action: 'all', reloads: false },
	] as const)('does not retry a cancelled job during CPU reload ($action, reloads=$reloads)', async ({ action, reloads }) => {
		const gate = deferred<string>()
		let loads = 0
		let attempts = 0
		const normal = mocks.invoke.getMockImplementation()!
		vi.spyOn(console, 'error').mockImplementation(() => {})
		mocks.invoke.mockImplementation((command, args) => {
			if (command === 'load_model' && ++loads === 2) return gate.promise
			if (command === 'transcribe' && attempts++ === 0) return Promise.reject({ code: 'gpu_out_of_memory', message: 'GPU allocation failed' })
			return normal(command, args)
		})
		const { result } = renderHook(useTranscribeQueue)
		act(() => result.current.enqueue(files))
		await waitFor(() => expect(calls('load_model')).toHaveLength(2))
		act(() => { if (action === 'all') result.current.cancelAll(); else result.current.cancelCurrent() })
		await act(async () => { if (reloads) gate.resolve('loaded'); else gate.reject(new Error('CPU load failed')) })
		await finished(result)
		expect(transcribedPaths()).toEqual(action === 'all' ? [files[0].path] : files.map((file) => file.path))
		expect(result.current.jobs.map((job) => job.status)).toEqual(action === 'all' ? ['cancelled', 'cancelled'] : ['cancelled', 'done'])
		expect(mocks.errorModal).not.toHaveBeenCalled()
		expect(mocks.failed).not.toHaveBeenCalled()
	})

	it('still persists partial results from an already invoked, aborted transcription without exporting them', async () => {
		const gate = deferred<{ segments: typeof segments }>()
		const normal = mocks.invoke.getMockImplementation()!
		mocks.invoke.mockImplementation((command, args) => command === 'transcribe' ? gate.promise : normal(command, args))
		mocks.preference.autoExport.enabled = true
		const { result } = renderHook(useTranscribeQueue)
		act(() => result.current.enqueue([recordings[0]]))
		await waitFor(() => expect(calls('transcribe')).toHaveLength(1))
		act(() => result.current.cancelCurrent())
		expect(mocks.emit).toHaveBeenCalledWith('abort_transcribe')
		await act(async () => { gate.resolve({ segments }) })
		await finished(result)
		await waitFor(() => expect(mocks.saveTranscript).toHaveBeenCalledTimes(1))
		expect(result.current.jobs[0]).toMatchObject({ status: 'done', segments, privateAudioPath: undefined })
		expect(mocks.saveTranscript.mock.calls[0][0]).toMatchObject({ sourcePath: '', segments })
		expect(mocks.exportTranscript).not.toHaveBeenCalled()
		expect(mocks.cancelled).toHaveBeenCalledTimes(1)
		expect(calls('discard_private_audio')).toHaveLength(1)
	})
})
