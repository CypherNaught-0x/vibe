// @vitest-environment jsdom

import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useState } from 'react'
import type { Preference } from '~/providers/preference'
import { SessionProvider, useSession, type Session } from '../session'
import type { LiveRecordingEvent } from '~/pages/home/hooks/use-live-recording'

const mocks = vi.hoisted(() => ({
	invoke: vi.fn(),
	listen: vi.fn(),
	emit: vi.fn(),
	save: vi.fn(),
	hydrate: vi.fn(),
	transcribeJob: vi.fn(),
	toast: vi.fn(),
	preference: {} as Preference,
	shortcut: { isShortcutRecording: false, setNormalRecordingActive: vi.fn(), setRecordingBusy: vi.fn() },
}))
vi.mock('@tauri-apps/api/core', () => ({ invoke: mocks.invoke }))
vi.mock('@tauri-apps/api/event', () => ({ listen: mocks.listen, emit: mocks.emit }))
vi.mock('@tauri-apps/api', () => ({ webviewWindow: { getCurrentWebviewWindow: () => ({ show: vi.fn(), setFocus: vi.fn() }) } }))
vi.mock('@tauri-apps/plugin-dialog', () => ({ message: vi.fn(), open: vi.fn() }))
vi.mock('react-router-dom', () => ({ useNavigate: () => vi.fn() }))
vi.mock('~/providers/preference', () => ({ usePreferenceProvider: () => mocks.preference }))
vi.mock('~/providers/recording-shortcut', () => ({ useRecordingShortcut: () => mocks.shortcut }))
vi.mock('~/providers/hotkey', () => ({ hotkeyRecordingActive: false }))
vi.mock('~/lib/config-store', () => ({ usePersisted: (_key: string, fallback: unknown) => useState(fallback) }))
vi.mock('~/lib/model', () => ({ cleanupPartialDownloads: vi.fn(), listInstalledModels: async () => [{ path: '/models/model.audio8.json', valid: true }] }))
vi.mock('~/lib/keep-awake', () => ({ KEEP_AWAKE: { record: 'record' }, startKeepAwake: vi.fn(), stopKeepAwake: vi.fn() }))
vi.mock('~/lib/use-confirm-exit', () => ({ useConfirmExit: vi.fn() }))
vi.mock('~/lib/transcripts-store', () => ({ saveTranscript: mocks.save, notifyTranscriptsChanged: vi.fn(), TRANSCRIPT_VERSION: 1 }))
vi.mock('~/pages/home/hooks/use-audio-download', () => ({ useAudioDownload: () => ({}) }))
vi.mock('./use-drop-target', () => ({ useDropTarget: () => false }))
vi.mock('./use-summaries', () => ({ useSummaries: () => ({}) }))
vi.mock('./use-transcribe-queue', () => ({
	useTranscribeQueue: () => ({
		jobs: [],
		running: false,
		hydrate: mocks.hydrate,
		transcribeJob: mocks.transcribeJob,
		enqueue: vi.fn(),
		reset: vi.fn(),
	}),
}))
vi.mock('sonner', () => ({ toast: { error: mocks.toast, warning: vi.fn() } }))
vi.mock('~/paraglide/messages.js', () => ({ m: { error: () => 'Error', liveTranscriptionInfo: () => 'Audio8 requires en or zh' } }))

let session: Session
let listeners: Map<string, (event: { payload: unknown }) => void>
function Probe() {
	session = useSession()
	return null
}
function liveEvent(payload: Partial<LiveRecordingEvent> & Pick<LiveRecordingEvent, 'type'>) {
	listeners.get('live_transcription')?.({ payload: { sessionId: 'session-1', name: 'Meeting', ...payload } })
}
beforeEach(() => {
	vi.resetAllMocks()
	vi.useFakeTimers()
	listeners = new Map()
	mocks.preference = {
		homeTab: 'record',
		liveTranscription: true,
		modelPath: '/models/model.audio8.json',
		modelOptions: { lang: 'en', n_threads: 4 },
		advancedTranscribeOptions: { includeSubFolders: false },
		setModelPath: vi.fn(),
		saveTranscripts: true,
		projectsPath: '/projects',
	} as unknown as Preference
	mocks.shortcut.isShortcutRecording = false
	mocks.listen.mockImplementation(async (name: string, callback) => {
		listeners.set(name, callback)
		return () => listeners.delete(name)
	})
	mocks.invoke.mockImplementation(async (command: string) => {
		if (command === 'get_audio_devices') return [{ id: 'mic', name: 'Microphone', isInput: true, isDefault: true }]
		if (command === 'get_models_folder') return '/models'
		if (command === 'start_live_recording') {
			liveEvent({ type: 'loading' })
			return 'session-1'
		}
	})
	mocks.emit.mockResolvedValue(undefined)
	mocks.save.mockResolvedValue({ recordPath: '/projects/transcript.vibe.json', mediaPath: '' })
})
afterEach(() => {
	cleanup()
	vi.useRealTimers()
})

async function setup() {
	await act(async () => {
		render(
			<SessionProvider>
				<Probe />
			</SessionProvider>,
		)
	})
	await act(async () => {
		await session.recording.startRecord()
	})
}

describe('live recording session integration', () => {
	it('does not count model loading/benchmarking time and freezes the capture timer during flush', async () => {
		await setup()
		act(() => vi.advanceTimersByTime(5000))
		expect(session.recordElapsed).toBe(0)
		act(() => liveEvent({ type: 'loading', stage: 'calibrating' }))
		act(() => vi.advanceTimersByTime(3000))
		expect(session.recordElapsed).toBe(0)
		act(() => liveEvent({ type: 'ready' }))
		act(() => vi.advanceTimersByTime(2500))
		expect(session.recordElapsed).toBe(2)
		await act(async () => {
			await session.recording.stopRecord()
		})
		act(() => vi.advanceTimersByTime(3000))
		expect(session.recordElapsed).toBe(2)
	})

	it('ignores legacy finish/error channels for a live request and hydrates native text exactly once', async () => {
		await setup()
		await act(async () => {
			listeners.get('record_finish')?.({ payload: { path: '/stale.wav', name: 'Stale legacy finish' } })
			listeners.get('record_error')?.({ payload: 'Stale legacy error' })
		})
		expect(mocks.save).not.toHaveBeenCalled()
		expect(mocks.toast).not.toHaveBeenCalled()
		expect(session.recording.isRecording).toBe(true)
		const segments = [{ start: 0, stop: 100, text: 'Live transcript' }]
		await act(async () => {
			liveEvent({ type: 'finished', segments, processingTimeSec: 1 })
		})
		expect(mocks.hydrate).toHaveBeenCalledExactlyOnceWith(
			expect.objectContaining({ sourcePath: '', segments }),
			'/projects/transcript.vibe.json',
			null,
			'record',
		)
		expect(mocks.transcribeJob).not.toHaveBeenCalled()
		expect(session.recording.isRecording).toBe(false)
	})

	it('keeps shortcut recording on the existing capture clock and recording panel', async () => {
		mocks.shortcut.isShortcutRecording = true
		await act(async () => {
			render(
				<SessionProvider>
					<Probe />
				</SessionProvider>,
			)
		})
		act(() => vi.advanceTimersByTime(2500))
		expect(session.recording.isRecording).toBe(true)
		expect(session.recording.isLiveRecording).toBe(false)
		expect(session.recordElapsed).toBe(2)
		expect(session.panel).toBe('record')
		expect(mocks.invoke.mock.calls.some(([command]) => command === 'start_live_recording')).toBe(false)
	})
})
