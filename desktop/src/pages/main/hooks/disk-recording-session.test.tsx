// @vitest-environment jsdom

import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useState } from 'react'
import type { Preference } from '~/providers/preference'
import { SessionProvider, useSession, type Session } from '../session'

const mocks = vi.hoisted(() => ({
	invoke: vi.fn(),
	listen: vi.fn(),
	emit: vi.fn(),
	save: vi.fn(),
	hydrate: vi.fn(),
	transcribeJob: vi.fn(),
	toast: vi.fn(),
	stopKeepAwake: vi.fn(),
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
vi.mock('~/lib/permissions', () => ({ ensureSystemAudioPermission: async () => true }))
vi.mock('~/lib/config-store', () => ({ usePersisted: (_key: string, fallback: unknown) => useState(fallback) }))
vi.mock('~/lib/model', () => ({ cleanupPartialDownloads: vi.fn(), listInstalledModels: async () => [{ path: '/models/model.bin', valid: true }] }))
vi.mock('~/lib/keep-awake', () => ({ KEEP_AWAKE: { record: 'record' }, startKeepAwake: vi.fn(), stopKeepAwake: mocks.stopKeepAwake }))
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
vi.mock('~/paraglide/messages.js', () => ({ m: { error: () => 'Error', recordingRecoveredWarning: () => 'Recovered' } }))

let session: Session
let listeners: Map<string, (event: { payload: unknown }) => unknown>
function Probe() {
	session = useSession()
	return null
}

/** Fire a backend event and let its async handler run as far as it can. */
async function fire(name: string, payload: unknown = null) {
	await act(async () => {
		await listeners.get(name)?.({ payload })
	})
}

beforeEach(() => {
	vi.resetAllMocks()
	vi.useFakeTimers()
	listeners = new Map()
	mocks.preference = {
		homeTab: 'record',
		liveTranscription: false,
		modelPath: '/models/model.bin',
		modelOptions: { lang: 'en', n_threads: 4 },
		advancedTranscribeOptions: { includeSubFolders: false },
		setModelPath: vi.fn(),
		setHomeTab: vi.fn(),
		saveTranscripts: true,
		noAudioStorage: false,
		projectsPath: '/projects',
		recordingFormat: 'opus',
	} as unknown as Preference
	mocks.listen.mockImplementation(async (name: string, callback) => {
		listeners.set(name, callback)
		return () => listeners.delete(name)
	})
	mocks.invoke.mockImplementation(async (command: string) => {
		if (command === 'get_audio_devices') return [{ id: 'mic', name: 'Microphone', isInput: true, isDefault: true }]
		if (command === 'get_models_folder') return '/models'
	})
	mocks.emit.mockResolvedValue(undefined)
	mocks.hydrate.mockReturnValue('job-1')
})
afterEach(() => {
	cleanup()
	vi.useRealTimers()
})

async function startRecording() {
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

describe('disk recording stop and save', () => {
	it('stores the chosen format and stays in a saving state until the project is written', async () => {
		await startRecording()
		expect(mocks.invoke).toHaveBeenCalledWith('start_record', expect.objectContaining({ recordingFormat: 'opus' }))
		act(() => vi.advanceTimersByTime(3000))

		await act(async () => {
			await session.recording.stopRecord()
			await session.recording.stopRecord()
		})
		// One stop reaches the backend; the second click is swallowed while saving.
		expect(mocks.emit.mock.calls.filter(([name]) => name === 'stop_record')).toHaveLength(1)
		expect(session.recording.isSaving).toBe(true)
		expect(mocks.shortcut.setRecordingBusy).toHaveBeenLastCalledWith(true)

		let finishSave: (value: unknown) => void = () => {}
		mocks.save.mockReturnValue(new Promise((resolve) => (finishSave = resolve)))
		await fire('record_stopping')
		const finished = fire('record_finish', { path: '/tmp/2026-11-11 10-00-00.opus', name: '2026-11-11 10-00-00.opus' })
		await act(async () => {
			await Promise.resolve()
		})
		// The project save is still running: still saving, clock frozen, panel kept open.
		act(() => vi.advanceTimersByTime(3000))
		expect(session.recording.isSaving).toBe(true)
		expect(session.recording.isRecording).toBe(true)
		expect(session.recordElapsed).toBe(3)
		expect(session.panel).toBe('record')
		expect(mocks.save).toHaveBeenCalledWith(expect.objectContaining({ sourcePath: '/tmp/2026-11-11 10-00-00.opus', moveSourceMedia: true }))

		finishSave({ recordPath: '/projects/p/transcript.vibe.json', mediaPath: '/projects/p/audio.opus' })
		await finished
		expect(session.recording.isSaving).toBe(false)
		expect(session.recording.isRecording).toBe(false)
		expect(session.panel).toBe('none')
		expect(mocks.hydrate).toHaveBeenCalledWith(expect.objectContaining({ sourcePath: '/projects/p/audio.opus' }), '/projects/p/transcript.vibe.json', '/projects/p/audio.opus', 'record')
		expect(mocks.stopKeepAwake).toHaveBeenCalledWith('record')
		expect(mocks.shortcut.setRecordingBusy).toHaveBeenLastCalledWith(false)
	})

	it('shows saving for a stop that came from elsewhere and clears it when the save fails', async () => {
		await startRecording()
		// e.g. the global shortcut: the backend announces the accepted stop.
		await fire('record_stopping')
		expect(session.recording.isSaving).toBe(true)
		mocks.save.mockResolvedValue(null)
		await fire('record_finish', { path: '/tmp/capture.opus', name: 'capture.opus' })
		expect(mocks.toast).toHaveBeenCalled()
		expect(session.recording.isSaving).toBe(false)
		expect(session.recording.isRecording).toBe(false)
	})

	it('clears saving when finishing throws or the backend reports an error', async () => {
		await startRecording()
		await act(async () => {
			await session.recording.stopRecord()
		})
		mocks.save.mockRejectedValue(new Error('disk full'))
		await fire('record_finish', { path: '/tmp/capture.opus', name: 'capture.opus' })
		expect(session.recording.isSaving).toBe(false)

		await act(async () => {
			await session.recording.startRecord()
		})
		await act(async () => {
			await session.recording.stopRecord()
		})
		expect(session.recording.isSaving).toBe(true)
		await fire('record_error', { message: 'Recording stopped without any captured audio' })
		expect(session.recording.isSaving).toBe(false)
		expect(session.recording.isRecording).toBe(false)
	})

	it('releases the stop button when the backend never acknowledges the stop', async () => {
		await startRecording()
		await act(async () => {
			await session.recording.stopRecord()
		})
		act(() => vi.advanceTimersByTime(9_000))
		expect(session.recording.isSaving).toBe(true)
		act(() => vi.advanceTimersByTime(1_000))
		expect(session.recording.isSaving).toBe(false)
		// An acknowledged stop is never timed out, however long encoding and saving take.
		await act(async () => {
			await session.recording.stopRecord()
		})
		await fire('record_stopping')
		act(() => vi.advanceTimersByTime(60_000))
		expect(session.recording.isSaving).toBe(true)
	})
})
