// @vitest-environment jsdom

import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useState } from 'react'
import type { AudioDevice } from '~/lib/audio'
import type { Preference } from '~/providers/preference'
import { useLiveRecording, type LiveRecordingEvent } from './use-live-recording'
import { useRecording } from './use-recording'

const mocks = vi.hoisted(() => ({
	invoke: vi.fn(),
	listen: vi.fn(),
	emit: vi.fn(),
	startKeepAwake: vi.fn(),
	stopKeepAwake: vi.fn(),
	permission: vi.fn(),
	toast: vi.fn(),
	preference: {} as Preference,
}))
vi.mock('@tauri-apps/api/core', () => ({ invoke: mocks.invoke }))
vi.mock('@tauri-apps/api/event', () => ({ listen: mocks.listen, emit: mocks.emit }))
vi.mock('~/lib/permissions', () => ({ ensureSystemAudioPermission: mocks.permission }))
vi.mock('~/lib/keep-awake', () => ({ KEEP_AWAKE: { record: 'record' }, startKeepAwake: mocks.startKeepAwake, stopKeepAwake: mocks.stopKeepAwake }))
vi.mock('~/providers/preference', () => ({ usePreferenceProvider: () => mocks.preference }))
vi.mock('~/lib/config-store', () => ({ usePersisted: (_key: string, fallback: unknown) => useState(fallback) }))
vi.mock('sonner', () => ({ toast: { error: mocks.toast } }))
vi.mock('~/paraglide/messages.js', () => ({ m: { error: () => 'Error', liveTranscriptionInfo: () => 'Audio8 requires en or zh' } }))

const microphone = { id: 'mic', name: 'Microphone', isInput: true, isDefault: true } as AudioDevice
const output = { id: 'output', name: 'System audio', isInput: false, isDefault: true } as AudioDevice
let callbacks: Set<(event: { payload: LiveRecordingEvent }) => void>
let starts = 0
function event(payload: Partial<LiveRecordingEvent> & Pick<LiveRecordingEvent, 'type'>) {
	for (const callback of [...callbacks]) callback({ payload: { sessionId: 'session-1', name: 'Meeting', ...payload } })
}
function deferred<T>() {
	let resolve!: (value: T) => void
	const promise = new Promise<T>((done) => {
		resolve = done
	})
	return { promise, resolve }
}

beforeEach(() => {
	vi.resetAllMocks()
	callbacks = new Set()
	starts = 0
	mocks.preference = {
		liveTranscription: true,
		modelPath: '/models/model.audio8.json',
		modelOptions: { lang: 'en', n_threads: 4 },
		gpuDevice: 2,
		noGpu: false,
		unloadTimeoutMinutes: 5,
		noAudioStorage: false,
		homeTab: 'record',
	} as Preference
	mocks.permission.mockResolvedValue(true)
	mocks.emit.mockResolvedValue(undefined)
	mocks.listen.mockImplementation(async (_name, callback) => {
		callbacks.add(callback)
		return vi.fn(() => callbacks.delete(callback))
	})
	mocks.invoke.mockImplementation(async (command: string) => {
		if (command === 'get_audio_devices') return [microphone]
		if (command === 'start_live_recording') {
			const sessionId = `session-${++starts}`
			event({ type: 'loading', sessionId })
			return sessionId
		}
	})
})
afterEach(cleanup)

async function setup() {
	const onFinished = vi.fn().mockResolvedValue(undefined)
	const onBeforeStart = vi.fn()
	const hook = renderHook(() => useLiveRecording(onBeforeStart, onFinished))
	await act(async () => {
		await hook.result.current.start([microphone], '  Meeting  ')
	})
	return { ...hook, onFinished, onBeforeStart }
}

describe('managed live recording UI lifecycle', () => {
	it('installs the listener before native startup, passes explicit settings, and never preloads the model', async () => {
		const { result, onBeforeStart } = await setup()
		expect(mocks.listen).toHaveBeenCalledWith('live_transcription', expect.any(Function))
		expect(mocks.listen.mock.invocationCallOrder[0]).toBeLessThan(mocks.invoke.mock.invocationCallOrder[0])
		expect(mocks.invoke).toHaveBeenCalledExactlyOnceWith('start_live_recording', {
			devices: [microphone],
			recordingName: 'Meeting',
			language: 'en',
			threads: 4,
			modelPath: '/models/model.audio8.json',
			gpuDevice: 2,
			noGpu: false,
			unloadTimeoutMinutes: 5,
		})
		expect(onBeforeStart).toHaveBeenCalledOnce()
		expect(result.current.state).toMatchObject({ phase: 'loading', startedAt: null })
		expect(result.current.isRecording).toBe(true)
	})

	it('can stop before listener registration resolves without starting native capture', async () => {
		const registration = deferred<() => void>()
		const unlisten = vi.fn()
		mocks.listen.mockReturnValueOnce(registration.promise)
		const { result } = renderHook(() => useLiveRecording(vi.fn()))
		let starting!: Promise<void>
		act(() => {
			starting = result.current.start([microphone], '')
		})
		expect(result.current.isRecording).toBe(true)
		await act(async () => {
			await result.current.stop()
		})
		await act(async () => {
			registration.resolve(unlisten)
			await starting
		})
		expect(mocks.invoke).not.toHaveBeenCalled()
		expect(mocks.emit).not.toHaveBeenCalled()
		expect(unlisten).toHaveBeenCalledOnce()
		expect(result.current.isRecording).toBe(false)
		expect(result.current.state).toBeNull()
	})

	it('cancels a requested capture while system-audio permission is pending', async () => {
		const permission = deferred<boolean>()
		mocks.permission.mockReturnValueOnce(permission.promise)
		const { result } = renderHook(() => useLiveRecording(vi.fn()))
		let starting!: Promise<void>
		act(() => {
			starting = result.current.start([output], '')
		})
		await act(async () => {
			await result.current.stop()
		})
		await act(async () => {
			permission.resolve(true)
			await starting
		})
		expect(mocks.invoke).not.toHaveBeenCalled()
		expect(result.current.state).toBeNull()
	})

	it('remembers Stop while invocation has not installed native listeners yet', async () => {
		const invocation = deferred<string>()
		mocks.invoke.mockReturnValueOnce(invocation.promise)
		const { result } = renderHook(() => useLiveRecording(vi.fn()))
		let starting!: Promise<void>
		act(() => {
			starting = result.current.start([microphone], '')
		})
		await waitFor(() => expect(mocks.invoke).toHaveBeenCalledOnce())
		await act(async () => {
			await result.current.stop()
		})
		expect(mocks.emit).not.toHaveBeenCalled()
		await act(async () => {
			event({ type: 'loading' })
			invocation.resolve('session-1')
			await starting
		})
		expect(mocks.emit).toHaveBeenCalledExactlyOnceWith('stop_record', { sessionId: 'session-1' })
		act(() => event({ type: 'cancelled' }))
		expect(result.current.state).toBeNull()
		expect(result.current.isRecording).toBe(false)
	})

	it('stops an unmounted pending invocation after its ID resolves without stopping a newer session', async () => {
		const invocation = deferred<string>()
		mocks.invoke.mockReturnValueOnce(invocation.promise).mockImplementationOnce(async () => {
			event({ type: 'loading', sessionId: 'new-session' })
			return 'new-session'
		})
		const stoppedSessions: string[] = []
		mocks.emit.mockImplementation(async (_name: string, payload: { sessionId: string }) => {
			// Simulate native's new session filtering; the older task has already released its lease.
			if (payload.sessionId === 'new-session') stoppedSessions.push(payload.sessionId)
		})
		const old = renderHook(() => useLiveRecording(vi.fn()))
		let starting!: Promise<void>
		act(() => {
			starting = old.result.current.start([microphone], '')
		})
		await waitFor(() => expect(mocks.invoke).toHaveBeenCalledOnce())
		old.unmount()
		expect(mocks.emit).not.toHaveBeenCalled()
		const newer = renderHook(() => useLiveRecording(vi.fn()))
		await act(async () => {
			await newer.result.current.start([microphone], '')
		})
		act(() => event({ type: 'ready', sessionId: 'new-session' }))
		await act(async () => {
			invocation.resolve('old-session')
			await starting
		})
		expect(mocks.emit).toHaveBeenCalledExactlyOnceWith('stop_record', { sessionId: 'old-session' })
		expect(stoppedSessions).toEqual([])
		expect(newer.result.current.isRecording).toBe(true)
		expect(newer.result.current.state?.phase).toBe('ready')
	})

	it('uses the known session ID on unmount and repeats the scoped stop when pending IPC resolves', async () => {
		const invocation = deferred<string>()
		mocks.invoke.mockReturnValueOnce(invocation.promise)
		const { result, unmount } = renderHook(() => useLiveRecording(vi.fn()))
		let starting!: Promise<void>
		act(() => {
			starting = result.current.start([microphone], '')
		})
		await waitFor(() => expect(mocks.invoke).toHaveBeenCalledOnce())
		act(() => event({ type: 'loading' }))
		unmount()
		expect(mocks.emit).toHaveBeenCalledExactlyOnceWith('stop_record', { sessionId: 'session-1' })
		await act(async () => {
			invocation.resolve('session-1')
			await starting
		})
		expect(mocks.emit.mock.calls).toEqual([
			['stop_record', { sessionId: 'session-1' }],
			['stop_record', { sessionId: 'session-1' }],
		])
	})

	it('does not resend a stop after a terminal event while the initial IPC result is still pending', async () => {
		const invocation = deferred<string>()
		mocks.invoke.mockReturnValueOnce(invocation.promise)
		const { result, unmount } = renderHook(() => useLiveRecording(vi.fn()))
		let starting!: Promise<void>
		act(() => {
			starting = result.current.start([microphone], '')
		})
		await waitFor(() => expect(mocks.invoke).toHaveBeenCalledOnce())
		act(() => {
			event({ type: 'loading' })
			event({ type: 'cancelled' })
		})
		unmount()
		await act(async () => {
			invocation.resolve('session-1')
			await starting
		})
		expect(mocks.emit).not.toHaveBeenCalled()
	})

	it('starts the capture clock only at ready and keeps bounded buffer/device calibration visible', async () => {
		const { result } = await setup()
		act(() => event({ type: 'loading', stage: 'calibrating' }))
		expect(result.current.state).toMatchObject({ phase: 'calibrating', startedAt: null })
		const profile = { window_tokens: 128, pcm_capacity_samples: 32000, step_seconds: 0.05, realtime: true, device: 'mps' }
		act(() => event({ type: 'calibration', ...profile }))
		expect(result.current.state).toMatchObject({ phase: 'calibrating', calibration: profile, startedAt: null })
		act(() => event({ type: 'ready' }))
		expect(result.current.state?.phase).toBe('ready')
		expect(result.current.state?.startedAt).toEqual(expect.any(Number))
		await act(async () => {
			await result.current.stop()
		})
		expect(result.current.state?.phase).toBe('finishing')
		expect(result.current.state?.stoppedAt).toEqual(expect.any(Number))
		expect(result.current.isRecording).toBe(true)
	})

	it('replaces cumulative partial text per segment and converts committed bounds without rescaling', async () => {
		const { result, onFinished } = await setup()
		act(() => {
			event({ type: 'ready' })
			event({ type: 'partial', text: 'Hello', segment_index: 0 })
			event({ type: 'partial', text: 'Hello world', segment_index: 0 })
		})
		expect(result.current.state?.partial).toBe('Hello world')
		act(() => {
			event({ type: 'segment', segment: { start: 12, stop: 145, text: 'Hello world' }, segment_index: 0 })
			event({ type: 'partial', text: 'stale', segment_index: 0 })
			event({ type: 'partial', text: 'Next thought', segment_index: 1 })
		})
		expect(result.current.state?.segments).toEqual([{ start: 12, stop: 145, text: 'Hello world' }])
		expect(result.current.state?.partial).toBe('Next thought')
		const segments = [{ start: 12, stop: 145, text: 'Final authoritative text' }]
		await act(async () => {
			event({ type: 'finished', segments, processingTimeSec: 8 })
		})
		expect(onFinished).toHaveBeenCalledWith(expect.objectContaining({ segments, processingTimeSec: 8, language: 'en' }), expect.any(Function))
		expect(result.current.isRecording).toBe(false)
		expect(result.current.state).toMatchObject({ phase: 'finished', segments, partial: '' })
	})

	it('retains committed and partial captions after a native failure', async () => {
		const { result, onFinished } = await setup()
		act(() => {
			event({ type: 'ready' })
			event({ type: 'segment', start: 0, end: 100, text: 'Committed', segment_index: 0 })
			event({ type: 'partial', text: 'Unfinished', segment_index: 1 })
			event({ type: 'error', message: 'Backend disconnected' })
		})
		expect(result.current.state).toMatchObject({ phase: 'error', partial: 'Unfinished', segments: [{ start: 0, stop: 100, text: 'Committed' }] })
		expect(result.current.isRecording).toBe(false)
		expect(onFinished).not.toHaveBeenCalled()
		expect(mocks.toast).toHaveBeenCalledWith('Error', expect.objectContaining({ description: 'Backend disconnected' }))
	})

	it('ignores old-session terminal/partial events when a new capture begins', async () => {
		const { result, onFinished } = await setup()
		const oldCallback = [...callbacks][0]
		act(() => event({ type: 'cancelled' }))
		await act(async () => {
			await result.current.start([microphone], 'Second')
		})
		act(() => {
			oldCallback({ payload: { sessionId: 'session-1', name: 'Old', type: 'finished', segments: [] } })
			event({ type: 'partial', sessionId: 'session-1', text: 'Old text' })
			event({ type: 'ready', sessionId: 'session-2' })
			event({ type: 'partial', sessionId: 'session-2', text: 'New text', segment_index: 0 })
		})
		expect(result.current.isRecording).toBe(true)
		expect(result.current.state).toMatchObject({ phase: 'ready', partial: 'New text' })
		expect(onFinished).not.toHaveBeenCalled()
	})

	it('allows retrying Stop after a failed event dispatch instead of abandoning active audio', async () => {
		const { result } = await setup()
		act(() => event({ type: 'ready' }))
		mocks.emit.mockRejectedValueOnce('Dispatch failed')
		await act(async () => {
			await result.current.stop()
		})
		expect(result.current.isRecording).toBe(true)
		expect(result.current.state?.phase).toBe('ready')
		await act(async () => {
			await result.current.stop()
		})
		expect(mocks.emit).toHaveBeenCalledTimes(2)
	})

	it.each(['auto', 'fr'])('does not send an unsupported or implicit language (%s)', async (lang) => {
		mocks.preference.modelOptions.lang = lang
		const { result } = renderHook(() => useLiveRecording(vi.fn()))
		await act(async () => {
			await result.current.start([microphone], '')
		})
		expect(mocks.invoke).not.toHaveBeenCalled()
		expect(result.current.isRecording).toBe(false)
	})
})

describe('manual recording routing', () => {
	it.each([
		{ liveTranscription: false, modelPath: '/models/model.audio8.json' },
		{ liveTranscription: true, modelPath: '/models/ggml.bin' },
	])('keeps legacy recording for $modelPath with live=$liveTranscription', async (overrides) => {
		Object.assign(mocks.preference, overrides, { noAudioStorage: true })
		const { result } = renderHook(() => useRecording(vi.fn()))
		await waitFor(() => expect(result.current.inputDevice).toEqual(microphone))
		await act(async () => {
			await result.current.startRecord()
		})
		expect(mocks.invoke).toHaveBeenCalledWith('start_record', { devices: [microphone], recordingName: null, noAudioStorage: true })
		expect(mocks.invoke.mock.calls.some(([command]) => command === 'start_live_recording')).toBe(false)
		await act(async () => {
			await result.current.stopRecord()
		})
		expect(mocks.emit).toHaveBeenCalledWith('stop_record')
	})

	it('routes Audio8 opt-in to live recording and handles a same-render Stop through request state', async () => {
		const registration = deferred<() => void>()
		mocks.listen.mockReturnValueOnce(registration.promise)
		const { result } = renderHook(() => useRecording(vi.fn()))
		await waitFor(() => expect(result.current.inputDevice).toEqual(microphone))
		let starting!: Promise<void>
		await act(async () => {
			starting = result.current.startRecord()
			await result.current.stopRecord()
		})
		await act(async () => {
			registration.resolve(vi.fn())
			await starting
		})
		expect(mocks.invoke.mock.calls.some(([command]) => command === 'start_live_recording')).toBe(false)
		expect(mocks.emit).not.toHaveBeenCalled()
		expect(result.current.isRecording).toBe(false)
	})
})
