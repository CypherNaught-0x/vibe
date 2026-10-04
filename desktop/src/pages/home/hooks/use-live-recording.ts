import { invoke } from '@tauri-apps/api/core'
import { emit, listen, type UnlistenFn } from '@tauri-apps/api/event'
import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { m } from '~/paraglide/messages.js'
import type { AudioDevice } from '~/lib/audio'
import { KEEP_AWAKE, startKeepAwake, stopKeepAwake } from '~/lib/keep-awake'
import { ensureSystemAudioPermission } from '~/lib/permissions'
import type { Segment } from '~/lib/transcript'
import { usePreferenceProvider } from '~/providers/preference'

export interface LiveCalibration {
	window_tokens: number
	pcm_capacity_samples: number
	step_seconds: number
	realtime: boolean
	device?: string
}

/** Native committed and final segment bounds are centiseconds, never seconds. */
export interface LiveRecordingEvent extends Partial<LiveCalibration> {
	sessionId: string
	name: string
	type: 'loading' | 'calibration' | 'ready' | 'partial' | 'segment' | 'finished' | 'cancelled' | 'error'
	profile?: LiveCalibration
	stage?: string
	segment_index?: number
	text?: string
	start?: number
	end?: number
	segment?: Segment
	segments?: Segment[]
	processingTimeSec?: number
	message?: string
}

export interface LiveRecordingState {
	phase: 'loading' | 'calibrating' | 'ready' | 'finishing' | 'finished' | 'error'
	name: string
	segments: Segment[]
	partial: string
	partialIndex: number | null
	calibration?: LiveCalibration
	startedAt: number | null
	stoppedAt: number | null
	error?: string
}

export interface LiveRecordingResult {
	name: string
	segments: Segment[]
	processingTimeSec?: number
	language: 'en' | 'zh'
	modelPath: string
	createdAt: Date
}

export type LiveRecordingCompletion = (result: LiveRecordingResult, isCurrent: () => boolean) => Promise<void>

interface Request {
	generation: number
	sessionId: string | null
	invoked: boolean
	installed: boolean
	stopRequested: boolean
	stopSent: boolean
	terminal: boolean
	unlisten?: UnlistenFn
	state: LiveRecordingState
}

export function supportsLiveRecording(modelPath: string | null) {
	return !!modelPath?.toLowerCase().endsWith('.audio8.json')
}

/** A requested capture exists before permissions/listener/native startup finish, so Stop cannot be lost. */
export function useLiveRecording(onBeforeStart: () => void, onFinished?: LiveRecordingCompletion) {
	const preference = usePreferenceProvider()
	const callbacks = useRef({ onBeforeStart, onFinished })
	callbacks.current = { onBeforeStart, onFinished }
	const [state, setState] = useState<LiveRecordingState | null>(null)
	const [isRecording, setIsRecording] = useState(false)
	const active = useRef<Request | null>(null)
	const generation = useRef(0)
	const previousSessions = useRef(new Set<string>())

	function current(request: Request) {
		return active.current === request && generation.current === request.generation
	}

	function update(request: Request, changes: Partial<LiveRecordingState>) {
		if (!current(request)) return
		request.state = { ...request.state, ...changes }
		setState(request.state)
	}

	function release(request: Request) {
		request.unlisten?.()
		request.unlisten = undefined
		if (request.sessionId) previousSessions.current.add(request.sessionId)
		if (!current(request)) return
		active.current = null
		setIsRecording(false)
		void stopKeepAwake(KEEP_AWAKE.record)
	}

	function fail(request: Request, error: unknown) {
		if (!current(request)) return
		request.terminal = true
		const message = typeof error === 'object' && error !== null && 'message' in error ? String(error.message) : String(error)
		update(request, { phase: 'error', error: message, stoppedAt: request.state.stoppedAt ?? Date.now() })
		release(request)
		toast.error(m.error(), { description: message, position: 'bottom-center' })
	}

	async function sendStop(request: Request) {
		if (!current(request) || !request.installed || !request.sessionId || request.stopSent || request.terminal) return
		request.stopSent = true
		try {
			await emit('stop_record', { sessionId: request.sessionId })
		} catch (error) {
			// Keep the active capture stoppable if dispatch itself failed; don't pretend audio stopped.
			request.stopSent = false
			request.stopRequested = false
			update(request, { phase: request.state.startedAt ? 'ready' : request.state.calibration ? 'calibrating' : 'loading', stoppedAt: null })
			if (current(request)) toast.error(m.error(), { description: String(error), position: 'bottom-center' })
		}
	}

	async function start(devices: AudioDevice[], recordingName: string) {
		if (active.current) return
		const language = preference.modelOptions.lang
		if (!supportsLiveRecording(preference.modelPath) || (language !== 'en' && language !== 'zh')) {
			toast.error(m.liveTranscriptionInfo(), { position: 'bottom-center' })
			return
		}
		const modelPath = preference.modelPath!
		const createdAt = new Date()
		const request: Request = {
			generation: ++generation.current,
			sessionId: null,
			invoked: false,
			installed: false,
			stopRequested: false,
			stopSent: false,
			terminal: false,
			state: { phase: 'loading', name: recordingName, segments: [], partial: '', partialIndex: null, startedAt: null, stoppedAt: null },
		}
		active.current = request
		setState(request.state)
		setIsRecording(true)
		callbacks.current.onBeforeStart()
		const completed = callbacks.current.onFinished
		try {
			if (devices.some((device) => !device.isInput) && !(await ensureSystemAudioPermission())) {
				if (current(request)) {
					setState(null)
					release(request)
				}
				return
			}
			if (!current(request)) return
			void startKeepAwake(KEEP_AWAKE.record)
			const unlisten = await listen<LiveRecordingEvent>('live_transcription', ({ payload }) => {
				if (!current(request) || request.terminal || !request.invoked || previousSessions.current.has(payload.sessionId)) return
				// Native emits loading before resolving the command: bind provisionally, then verify its returned ID.
				if (!request.sessionId) {
					if (payload.type !== 'loading') return
					request.sessionId = payload.sessionId
					request.installed = true
				}
				if (payload.sessionId !== request.sessionId) return
				update(request, { name: payload.name })
				switch (payload.type) {
					case 'loading':
						if (!request.stopRequested) update(request, { phase: payload.stage === 'calibrating' ? 'calibrating' : 'loading' })
						break
					case 'calibration':
						update(request, {
							...(!request.stopRequested ? { phase: 'calibrating' as const } : {}),
							calibration: payload.profile ?? (payload as LiveCalibration),
						})
						break
					case 'ready':
						update(request, { phase: request.stopRequested ? 'finishing' : 'ready', startedAt: Date.now() })
						break
					case 'partial': {
						const index = payload.segment_index ?? request.state.segments.length
						if (index < request.state.segments.length || index < (request.state.partialIndex ?? 0)) break
						update(request, { partial: payload.text ?? '', partialIndex: index })
						break
					}
					case 'segment': {
						const index = payload.segment_index ?? request.state.segments.length
						const segment = payload.segment ?? { start: payload.start!, stop: payload.end!, text: payload.text ?? '' }
						if (index !== request.state.segments.length || !Number.isFinite(segment.start) || !Number.isFinite(segment.stop)) break
						update(request, {
							segments: [...request.state.segments, segment],
							...(request.state.partialIndex === null || request.state.partialIndex <= index ? { partial: '', partialIndex: null } : {}),
						})
						break
					}
					case 'cancelled':
						request.terminal = true
						setState(null)
						release(request)
						return
					case 'error':
						// Native currently supplies only a message; streamed committed and partial text survive.
						if (payload.segments) update(request, { segments: payload.segments })
						fail(request, payload.message ?? m.error())
						return
					case 'finished': {
						request.terminal = true
						const segments = payload.segments ?? request.state.segments
						update(request, { phase: 'finishing', segments, partial: '', partialIndex: null })
						request.unlisten?.()
						request.unlisten = undefined
						void stopKeepAwake(KEEP_AWAKE.record)
						void (async () => {
							try {
								await completed?.(
									{ name: payload.name, segments, processingTimeSec: payload.processingTimeSec, language, modelPath, createdAt },
									() => current(request),
								)
								update(request, { phase: 'finished' })
								release(request)
							} catch (error) {
								fail(request, error)
							}
						})()
						return
					}
				}
				if (request.stopRequested) void sendStop(request)
			})
			if (!current(request)) {
				unlisten()
				return
			}
			request.unlisten = unlisten
			request.invoked = true
			// Native owns model loading and benchmarking; never await load_model on the UI path.
			const sessionId = await invoke<string>('start_live_recording', {
				devices,
				recordingName: recordingName.trim() || null,
				language,
				threads: preference.modelOptions.n_threads ?? null,
				modelPath,
				gpuDevice: preference.gpuDevice,
				noGpu: preference.noGpu,
				unloadTimeoutMinutes: preference.unloadTimeoutMinutes,
			})
			if (!current(request)) {
				// Unmount may precede native IPC dispatch. Its listeners now exist, so repeat the stop
				// with the returned ID without affecting a newer capture (or any legacy recording).
				if (!request.terminal) await emit('stop_record', { sessionId })
				return
			}
			if (request.sessionId && request.sessionId !== sessionId) throw new Error('Live recording session ID mismatch')
			request.sessionId = sessionId
			request.installed = true
			if (request.stopRequested) await sendStop(request)
		} catch (error) {
			fail(request, error)
		}
	}

	async function stop() {
		const request = active.current
		if (!request || request.terminal) return
		request.stopRequested = true
		if (!request.invoked) {
			request.terminal = true
			setState(null)
			release(request)
			return
		}
		update(request, { phase: 'finishing', stoppedAt: Date.now() })
		await sendStop(request)
	}

	function clear() {
		if (active.current) return
		generation.current += 1
		setState(null)
	}

	useEffect(
		() => () => {
			const request = active.current
			generation.current += 1
			active.current = null
			if (request) {
				request.unlisten?.()
				if (request.invoked && request.sessionId && !request.terminal) {
					void emit('stop_record', { sessionId: request.sessionId }).catch(() => console.error('Failed to stop an unmounted live recording'))
				}
				void stopKeepAwake(KEEP_AWAKE.record)
			}
		},
		[],
	)

	return { state, isRecording, isActive: () => active.current !== null, start, stop, clear }
}
