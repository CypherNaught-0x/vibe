import { invoke } from '@tauri-apps/api/core'
import { emit, listen, type UnlistenFn } from '@tauri-apps/api/event'
import * as webview from '@tauri-apps/api/webviewWindow'
import { useCallback, useContext, useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import successSound from '~/assets/success.mp3'
import { m } from '~/paraglide/messages.js'
import { trackGpuOutOfMemory, trackTranscribeCancelled, trackTranscribeFailed, trackTranscribeStarted, trackTranscribeSucceeded } from '~/lib/analytics'
import { KEEP_AWAKE, startKeepAwake, stopKeepAwake } from '~/lib/keep-awake'
import { validPath } from '~/lib/media'
import { exportTranscript, sharedFolder, summarizeResults, tauriExportIo, type AutoExportResult } from '~/lib/auto-export'
import { openPath } from '~/lib/app'
import { notify } from '~/lib/notify'
import { autoProjectName } from '~/lib/project-name'
import { gpuOutOfMemoryBefore, rememberGpuOutOfMemory } from '~/lib/gpu-memory'
import { fatalRunError, isGpuOutOfMemory, isUserError, serverErrorCodes } from '~/lib/server-errors'
import type { Segment, SpeakerNames, Transcript } from '~/lib/transcript'
import type { AiThreadEntry, TranscriptRecord } from '~/lib/transcripts-store'
import { discardPrivateAudio, isPrivateAudioPath } from '~/lib/privacy'
import { useProjectPersistence } from './use-project-persistence'
import { keepForNewView } from './viewed-jobs'
export { reconcileProjectName } from './use-project-persistence'
import type { NamedPath, ProjectSource } from '~/lib/types'
import { ErrorModalContext } from '~/providers/error-modal'
import { useModelGates } from '~/providers/model-gates'
import { hasNativeDiarization, needsModelsFolder, speakerAndTimingOptions, withoutUnsupportedOptions } from '~/lib/model'
import { type Preference, usePreferenceProvider } from '~/providers/preference'

export type JobStatus = 'queued' | 'running' | 'done' | 'error' | 'cancelled'

/** One transcription input. `projectName` re-runs an existing project under the title it already has. */
export interface EnqueueItem extends NamedPath {
	/** Title of the project this input came from; used verbatim, without a source prefix. */
	projectName?: string
	/** User folder to save into; undefined uses the folder selected for new transcriptions. */
	folder?: string | null
}

export interface Job {
	id: string
	name: string
	path: string
	/** Unconsumed RAM recording handle; never exposed to the player or persisted. */
	privateAudioPath?: string
	/** Raw text was intentionally deleted; do not automatically retranscribe retained audio. */
	transcriptDeleted?: boolean
	/** User folder this new project is saved into, fixed when it was queued. */
	folder?: string | null
	/** Origin of this newly created project. */
	source?: ProjectSource
	/** Opened from Recents to read or edit, not queued as part of a run. */
	opened?: boolean
	status: JobStatus
	/** 0..100, only meaningful while running */
	progress: number
	segments: Segment[]
	error?: string
	/** wall clock seconds the transcription took */
	seconds?: number
	/** path of the .vibe.json this job was saved to / loaded from, when persisted */
	savedPath?: string
	/** true when the job was loaded from the store rather than transcribed in this session */
	hydrated?: boolean
	/** Last AI summary of this transcript, when one was made. */
	summary?: string
	/** Questions asked about this transcript and their answers, oldest first. */
	thread?: AiThreadEntry[]
	/** Names the user gave the diarized speakers, by zero-based speaker index. */
	speakerNames?: SpeakerNames
	/** What auto-export did with this transcript, when it ran. */
	exported?: AutoExportResult
}

/** A run of two or more files: what batch users watch instead of the transcripts. */
export interface BatchProgress {
	total: number
	done: number
	startedAt: number
	/** Seconds per finished file so far, for the time estimate; null until one finished. */
	secondsPerFile: number | null
}

export interface BatchSummary {
	total: number
	seconds: number
	exported: number
	skipped: number
	fallback: number
	failed: number
	/** The one folder everything went to, when there is one. */
	folder: string | null
	/** Jobs whose export did not go as planned, for "Show details". */
	exceptions: { name: string; detail: string }[]
}

export interface TranscribeQueue {
	jobs: Job[]
	activeId: string | null
	selectedId: string | null
	selectedJob: Job | null
	running: boolean
	isAborting: boolean
	hasResults: boolean
	selectJob: (id: string) => void
	enqueue: (files: EnqueueItem[]) => void
	/** Transcribe the media already attached to a hydrated project with no transcript. */
	transcribeJob: (jobId: string) => void
	/**
	 * Show a transcript loaded from the store as a finished job, in place of the session but beside
	 * a run in progress. `audioPath` is the media copy kept in the project folder, when it still has one; it becomes the
	 * job's path so the player keeps working after the original file moved away.
	 */
	hydrate: (record: TranscriptRecord, savedPath: string, audioPath?: string | null, source?: ProjectSource) => string | null
	/** Inline edit of one segment's text. Persists to the job's saved file when it has one. */
	updateSegmentText: (jobId: string, segmentIndex: number, text: string) => void
	/** Rename the visible project and its persisted record, when one exists. */
	renameJob: (jobId: string, name: string) => Promise<boolean>
	/** Update the visible title while the inline rename editor is open. */
	previewJobName: (jobId: string, name: string) => void
	/** Attach an AI summary to a job. Persists to the job's saved file when it has one. */
	setJobSummary: (jobId: string, summary: string) => Promise<boolean>
	/** Replace the question-and-answer thread of a job. Persists like the summary. */
	setJobThread: (jobId: string, thread: AiThreadEntry[]) => void
	/** Name one diarized speaker; an empty name restores "Speaker N". Persists like the segment edits. */
	setSpeakerName: (jobId: string, speaker: number, name: string) => void
	/** Attribute one line to a speaker: a line diarization missed, or one it got wrong. */
	setSegmentSpeaker: (jobId: string, segmentIndex: number, speaker: number) => void
	/** A saved project moved on disk (sidebar move/rename): point jobs holding it at the new place. */
	relocateSaved: (fromPath: string, toPath: string, mediaPath?: string) => void
	cancelCurrent: () => void
	cancelAll: () => void
	reset: () => void
	/** Progress across a run of several files, null for a single file or when idle. */
	batch: BatchProgress | null
	/** The finished card's content, until dismissed. */
	batchSummary: BatchSummary | null
	dismissBatchSummary: () => void
	/** Finished jobs auto-export has not written yet: the count behind "Export all". */
	unexportedCount: number
	/** Export every finished job that has nothing written yet, with the current settings. */
	exportFinished: () => Promise<void>
}

let jobCounter = 0
function nextJobId() {
	jobCounter += 1
	return `job-${jobCounter}`
}

/** Options shared by every file of a run (diarize / vad model paths). */
async function buildSharedOptions(preference: Preference) {
	const capabilities = preference.modelMetadata?.capabilities
	const modelsFolder = needsModelsFolder(preference, capabilities) ? await invoke<string>('get_models_folder') : null
	return speakerAndTimingOptions(preference, capabilities, modelsFolder)
}

/** Reload the current model on the CPU. False when even that failed, so the original error stands. */
async function reloadOnCpu(preference: Preference): Promise<boolean> {
	try {
		await invoke('load_model', {
			modelPath: preference.modelPath,
			gpuDevice: preference.gpuDevice,
			noGpu: true,
			unloadTimeoutMinutes: preference.unloadTimeoutMinutes,
		})
		return true
	} catch (error) {
		console.error('reloading on the CPU after the GPU ran out of memory failed', error)
		return false
	}
}

function errorParts(error: unknown) {
	const object = typeof error === 'object' && error !== null ? (error as { code?: string; message?: string }) : null
	return { code: object?.code, message: object?.message || String(error) }
}

/**
 * One sequential transcription queue: the model is loaded once, then every queued file runs
 * in order. Segments of the currently running file stream in live through `new_segment`.
 */
export function useTranscribeQueue(): TranscribeQueue {
	const preference = usePreferenceProvider()
	const preferenceRef = useRef(preference)
	const { setState: setErrorModal } = useContext(ErrorModalContext)
	const { ensureDiarizeModel } = useModelGates()

	const [jobs, setJobs] = useState<Job[]>([])
	const jobsRef = useRef<Job[]>([])
	const [activeId, setActiveId] = useState<string | null>(null)
	const activeIdRef = useRef<string | null>(null)
	const preparingIdRef = useRef<string | null>(null)
	const [selectedId, setSelectedId] = useState<string | null>(null)
	const selectedIdRef = useRef<string | null>(null)
	const pinnedRef = useRef(false)
	const [running, setRunning] = useState(false)
	const runningRef = useRef(false)
	const [isAborting, setIsAborting] = useState(false)
	const abortCurrentRef = useRef(false)
	const abortAllRef = useRef(false)

	useEffect(() => {
		preferenceRef.current = preference
	}, [preference])

	const commit = useCallback((next: Job[]) => {
		jobsRef.current = next
		setJobs(next)
	}, [])

	const patch = useCallback(
		(id: string, changes: Partial<Job>) => {
			commit(jobsRef.current.map((job) => (job.id === id ? { ...job, ...changes } : job)))
		},
		[commit],
	)

	const select = useCallback((id: string | null) => {
		selectedIdRef.current = id
		setSelectedId(id)
	}, [])

	const selectJob = useCallback(
		(id: string) => {
			pinnedRef.current = true
			select(id)
		},
		[select],
	)

	const { persist, updateSegmentText, setSegmentSpeaker, renameJob, previewJobName, setJobSummary, setJobThread, setSpeakerName, relocateSaved } = useProjectPersistence(
		jobsRef,
		preferenceRef,
		patch,
	)

	useEffect(() => {
		const unlisteners: Promise<UnlistenFn>[] = []

		unlisteners.push(
			listen<number>('transcribe_progress', ({ payload }) => {
				const id = activeIdRef.current
				if (id && payload >= 0 && payload <= 100) patch(id, { progress: payload })
			}),
		)
		unlisteners.push(
			listen<Segment>('new_segment', ({ payload }) => {
				const id = activeIdRef.current
				if (!id) return
				commit(jobsRef.current.map((job) => (job.id === id ? { ...job, segments: [...job.segments, payload] } : job)))
			}),
		)

		return () => {
			unlisteners.forEach((promise) => promise.then((unlisten) => unlisten()))
		}
	}, [commit, patch])

	const discardPendingAudio = useCallback(() => {
		for (const job of jobsRef.current) {
			if (job.privateAudioPath && job.id !== activeIdRef.current) void discardPrivateAudio(job.privateAudioPath)
		}
	}, [])

	useEffect(() => () => discardPendingAudio(), [discardPendingAudio])

	const failPending = useCallback(
		(message: string) => {
			discardPendingAudio()
			commit(
				jobsRef.current.map((job) =>
					job.status === 'queued' || job.status === 'running' ? { ...job, privateAudioPath: undefined, status: 'error', error: message } : job,
				),
			)
		},
		[commit, discardPendingAudio],
	)

	const [batch, setBatch] = useState<BatchProgress | null>(null)
	const [batchSummary, setBatchSummary] = useState<BatchSummary | null>(null)
	const runResultsRef = useRef<Map<string, AutoExportResult>>(new Map())

	/**
	 * Write a finished transcript in the auto-export formats. Never throws and never touches
	 * the job's status: the transcription succeeded whatever happens to the files.
	 */
	const autoExportJob = useCallback(
		async (job: Job, segments: Segment[]): Promise<AutoExportResult | null> => {
			const current = preferenceRef.current
			if (!current.autoExport.enabled || segments.length === 0) return null
			const result = await exportTranscript(
				{ name: job.name, path: job.path, segments, summary: job.summary, speakerNames: job.speakerNames },
				current.autoExport,
				{
					direction: current.textAreaDirection,
					theme: current.exportOptions.theme,
					showTimestamps: current.exportOptions.showTimestamps,
					showSpeakers: current.exportOptions.showSpeakers,
					speakerLabel: m.speakerPrefix(),
					labels: { transcript: m.exportTranscript(), summary: m.exportSummary() },
				},
				tauriExportIo(current.projectsPath),
				{ skipped: (names) => m.autoExportSkipped({ names }), fallback: m.autoExportFallback() },
			)
			patch(job.id, { exported: result })
			runResultsRef.current.set(job.id, result)
			return result
		},
		[patch],
	)

	const runLoop = useCallback(async () => {
		if (runningRef.current) return
		runningRef.current = true
		setRunning(true)
		abortAllRef.current = false
		abortCurrentRef.current = false
		preparingIdRef.current = jobsRef.current.find((job) => job.status === 'queued')?.id ?? null
		setIsAborting(false)
		startKeepAwake(KEEP_AWAKE.queue)
		let completedAny = false
		const runStartedAt = Date.now()
		const runTotal = jobsRef.current.filter((job) => job.status === 'queued').length
		const runIds = new Set(jobsRef.current.filter((job) => job.status === 'queued').map((job) => job.id))
		let runDone = 0
		let runSeconds = 0
		runResultsRef.current = new Map()
		if (runTotal > 1) {
			setBatchSummary(null)
			setBatch({ total: runTotal, done: 0, startedAt: runStartedAt, secondsPerFile: null })
		}

		try {
			const current = preferenceRef.current
			if (!current.modelPath) {
				failPending(m.noModelSelected())
				toast.error(m.noModelSelected(), { position: 'bottom-center' })
				return
			}

			// Once the GPU runs out of memory the rest of the run stays on the CPU, and so does
			// this model on every later launch: a loaded model that fits can still overflow once
			// the working buffers of a long file or diarization arrive.
			let onCpu = current.noGpu || gpuOutOfMemoryBefore(current.modelPath)
			let shared: Record<string, unknown>
			try {
				const loadResult = await invoke<string>('load_model', {
					modelPath: current.modelPath,
					gpuDevice: current.gpuDevice,
					noGpu: onCpu,
					unloadTimeoutMinutes: current.unloadTimeoutMinutes,
				})
				if (abortAllRef.current || !jobsRef.current.some((job) => job.status === 'queued')) return
				if (loadResult === 'gpu_fallback') toast.warning(m.gpuFallbackToCpu(), { position: 'bottom-center', duration: 8000 })
				// Speaker recognition stays on across updates while its model can change name with one, so
				// an existing user may have it on with the new model missing: ask for it now rather than
				// transcribing without speakers. Declining, or a failed download, turns the setting off and
				// the run goes on.
				let options = current
				if (current.diarizeEnabled && !hasNativeDiarization(current.modelMetadata?.capabilities)) {
					const ready = await ensureDiarizeModel().catch((error) => {
						console.error('diarization model download failed:', error)
						toast.error(String(error), { position: 'bottom-center' })
						return false
					})
					if (!ready) {
						current.setDiarizeEnabled(false)
						options = { ...current, diarizeEnabled: false }
					}
				}
				shared = await buildSharedOptions(options)
			} catch (error) {
				if (abortAllRef.current || !jobsRef.current.some((job) => job.status === 'queued')) return
				const { message } = errorParts(error)
				failPending(message)
				setErrorModal?.({ log: message, open: true })
				return
			}

			preparingIdRef.current = null
			while (!abortAllRef.current) {
				const next = jobsRef.current.find((job) => job.status === 'queued')
				if (!next) break

				activeIdRef.current = next.id
				setActiveId(next.id)
				if (!pinnedRef.current) select(next.id)
				patch(next.id, { status: 'running', progress: 0, segments: [], error: undefined })
				abortCurrentRef.current = false

				const startedAt = performance.now()
				trackTranscribeStarted('main', next.path)
				try {
					const result = await invoke<Transcript>('transcribe', {
						options: {
							path: next.privateAudioPath ?? next.path,
							...withoutUnsupportedOptions(preferenceRef.current.modelOptions, preferenceRef.current.modelMetadata?.capabilities),
							...shared,
							no_audio_storage: preferenceRef.current.noAudioStorage,
						},
					})
					const seconds = Math.round((performance.now() - startedAt) / 1000)
					patch(next.id, { status: 'done', progress: 100, segments: result.segments, seconds, privateAudioPath: undefined })
					completedAny = true
					runDone += 1
					runSeconds += seconds
					if (runTotal > 1) setBatch({ total: runTotal, done: runDone, startedAt: runStartedAt, secondsPerFile: runSeconds / runDone })
					void persist(next, result.segments)
					if (!abortCurrentRef.current && !abortAllRef.current) void autoExportJob(next, result.segments)
					// An abort resolves as a success carrying the partial segments, so only these flags tell them apart.
					if (abortCurrentRef.current || abortAllRef.current) {
						trackTranscribeCancelled('main', next.path)
					} else {
						trackTranscribeSucceeded('main', { durationSeconds: seconds, segmentsCount: result.segments.length })
					}
				} catch (error) {
					if (abortCurrentRef.current || abortAllRef.current) {
						patch(next.id, { status: 'cancelled', progress: 0 })
						trackTranscribeCancelled('main', next.path)
					} else {
						const { code, message } = errorParts(error)
						if (!next.privateAudioPath && !onCpu && isGpuOutOfMemory(code, message) && (await reloadOnCpu(current))) {
							onCpu = true
							rememberGpuOutOfMemory(current.modelPath)
							trackGpuOutOfMemory('main', { signal: code === serverErrorCodes.GPU_OUT_OF_MEMORY ? 'error_code' : 'process_death' })
							toast.warning(m.gpuFallbackToCpu(), { position: 'bottom-center', duration: 8000 })
							// Back to the queue for one more try; on the CPU an overflow is the plain error.
							const cancelled = abortCurrentRef.current || abortAllRef.current
							patch(next.id, { status: cancelled ? 'cancelled' : 'queued', progress: 0 })
							if (cancelled) trackTranscribeCancelled('main', next.path)
							continue
						}
						if (abortCurrentRef.current || abortAllRef.current) {
							patch(next.id, { status: 'cancelled', progress: 0 })
							trackTranscribeCancelled('main', next.path)
							continue
						}
						patch(next.id, { status: 'error', progress: 0, error: message })
						const userError = Boolean(code && isUserError(code))
						trackTranscribeFailed('main', next.path, { errorMessage: message, userError })
						if (userError) {
							toast.error(`${m.error()}: ${message}`, { position: 'bottom-center' })
						} else {
							setErrorModal?.({ log: message, open: true })
							// A dead sidecar or an unloaded model fails every following file the same way.
							const fatal = fatalRunError(code, message)
							if (fatal) {
								toast.error(fatal === 'no_model' ? m.noModelLoadedBatchStopped() : m.transcribeEngineStoppedBatchStopped(), {
									position: 'bottom-center',
								})
								failPending(message)
								break
							}
						}
					}
				} finally {
					if (next.privateAudioPath) {
						void discardPrivateAudio(next.privateAudioPath)
						patch(next.id, { privateAudioPath: undefined })
					}
					abortCurrentRef.current = false
					setIsAborting(false)
				}
			}
		} finally {
			if (abortAllRef.current) {
				discardPendingAudio()
				commit(jobsRef.current.map((job) => (job.status === 'queued' ? { ...job, privateAudioPath: undefined, status: 'cancelled' } : job)))
			}
			preparingIdRef.current = null
			stopKeepAwake(KEEP_AWAKE.queue)
			activeIdRef.current = null
			setActiveId(null)
			runningRef.current = false
			setRunning(false)
			setIsAborting(false)

			setBatch(null)
			if (completedAny && runTotal > 1) {
				// Export results may still be landing; wait for the ones this run started.
				await new Promise((resolve) => setTimeout(resolve, 50))
				const results = [...runIds].map((id) => runResultsRef.current.get(id)).filter((r): r is AutoExportResult => Boolean(r))
				const counts = summarizeResults(results)
				const summary: BatchSummary = {
					total: runTotal,
					seconds: Math.round((Date.now() - runStartedAt) / 1000),
					...counts,
					folder: sharedFolder(results),
					exceptions: jobsRef.current
						.filter((job) => runIds.has(job.id) && job.exported && job.exported.status !== 'exported')
						.map((job) => ({ name: job.name, detail: job.exported?.detail ?? job.exported?.status ?? '' })),
				}
				setBatchSummary(summary)
				const exportedLine = preferenceRef.current.autoExport.enabled
					? m.batchFinishedExported({ count: String(counts.exported + counts.fallback) })
					: m.batchFinishedFiles({ count: String(runDone) })
				void notify(m.batchFinishedTitle(), exportedLine)
			}
			if (completedAny) {
				const finished = preferenceRef.current
				if (finished.soundOnFinish) new Audio(successSound).play()
				if (finished.focusOnFinish) {
					webview.getCurrentWebviewWindow().unminimize()
					webview.getCurrentWebviewWindow().setFocus()
				}
			}

			// A file enqueued while the loop was winding down would otherwise stay queued forever.
			if (!abortAllRef.current && jobsRef.current.some((job) => job.status === 'queued')) void runLoop()
		}
	}, [autoExportJob, commit, discardPendingAudio, ensureDiarizeModel, failPending, patch, persist, select, setErrorModal])

	const unexportedCount = jobs.filter((job) => job.status === 'done' && job.segments.length > 0 && !job.exported).length

	/** "Export all": catch up every finished job with nothing written, with the current settings. */
	const exportFinished = useCallback(async () => {
		const pending = jobsRef.current.filter((job) => job.status === 'done' && job.segments.length > 0 && !job.exported)
		const results: AutoExportResult[] = []
		for (const job of pending) {
			const result = await autoExportJob(job, job.segments)
			if (result) results.push(result)
		}
		if (results.length === 0) return
		const counts = summarizeResults(results)
		const folder = sharedFolder(results)
		toast.success(m.batchFinishedExported({ count: String(counts.exported + counts.fallback) }), {
			position: 'bottom-center',
			...(folder ? { action: { label: m.openFolder(), onClick: () => void openPath({ name: '', path: folder }) } } : {}),
		})
	}, [autoExportJob])

	const dismissBatchSummary = useCallback(() => setBatchSummary(null), [])

	const transcribeJob = useCallback(
		(jobId: string) => {
			const job = jobsRef.current.find((candidate) => candidate.id === jobId)
			if (
				!job?.hydrated ||
				job.transcriptDeleted ||
				(!job.savedPath && !job.privateAudioPath) ||
				(!job.path && !job.privateAudioPath) ||
				!['done', 'error', 'cancelled'].includes(job.status) ||
				job.segments.length > 0
			)
				return
			pinnedRef.current = true
			commit(jobsRef.current.map((candidate) => (candidate.id === jobId ? { ...candidate, status: 'queued', progress: 0, error: undefined } : candidate)))
			select(jobId)
			void runLoop()
		},
		[commit, runLoop, select],
	)

	const enqueue = useCallback(
		(files: EnqueueItem[]) => {
			const accepted = files.filter((file) => (isPrivateAudioPath(file.path) ? file.source === 'record' : validPath(file.path.toLowerCase())))
			if (accepted.length === 0) {
				if (files.length > 0) toast.error(m.supportsFormats(), { position: 'bottom-center' })
				return
			}

			const created: Job[] = accepted.map((file) => ({
				id: nextJobId(),
				// A re-transcribed project keeps its own title: prefixing it again would grow
				// "File-multi" into "File-File-multi" on every run.
				name: file.projectName ?? autoProjectName(file.name, file.source ?? 'file'),
				path: isPrivateAudioPath(file.path) ? '' : file.path,
				privateAudioPath: isPrivateAudioPath(file.path) ? file.path : undefined,
				source: file.source ?? 'file',
				folder: file.folder !== undefined ? file.folder : preferenceRef.current.transcriptFolder,
				status: 'queued',
				progress: 0,
				segments: [],
			}))
			commit([...jobsRef.current, ...created])
			if (!selectedIdRef.current) select(created[0].id)
			void runLoop()
		},
		[commit, runLoop, select],
	)

	/** Load a saved transcript as the whole session: one finished job the done view can render. */
	const hydrate = useCallback(
		(record: TranscriptRecord, savedPath: string, audioPath?: string | null, source?: ProjectSource) => {
			pinnedRef.current = true
			// Already in the session (say, still transcribing): show that job, not a stale copy.
			const existing = savedPath ? jobsRef.current.find((other) => other.savedPath === savedPath) : undefined
			if (existing) {
				select(existing.id)
				return existing.id
			}
			const kept = keepForNewView(jobsRef.current, runningRef.current)
			const id = nextJobId()
			const mediaPath = audioPath || record.sourcePath
			const privateAudioPath = isPrivateAudioPath(mediaPath) ? mediaPath : undefined
			const job: Job = {
				id,
				name: record.name,
				// The project folder's own copy of the media, else the original path it came from.
				path: privateAudioPath ? '' : mediaPath,
				privateAudioPath,
				transcriptDeleted: record.transcriptDeleted,
				source,
				opened: source !== 'record',
				status: 'done',
				progress: 100,
				segments: record.segments,
				savedPath,
				hydrated: true,
				summary: record.summary,
				thread: record.thread,
				speakerNames: record.speakerNames,
			}
			commit([...kept, job])
			select(job.id)
			return id
		},
		[commit, select],
	)

	const cancelCurrent = useCallback(() => {
		if (!activeIdRef.current) {
			// Setup has no inference listener yet: cancel its first job locally,
			// without setting an abort flag that could spill into the next file.
			const preparing = jobsRef.current.find((job) => job.id === preparingIdRef.current && job.status === 'queued')
			if (!preparing) return
			if (preparing.privateAudioPath) void discardPrivateAudio(preparing.privateAudioPath)
			patch(preparing.id, { status: 'cancelled', progress: 0, privateAudioPath: undefined })
			trackTranscribeCancelled('main', preparing.path)
			return
		}
		abortCurrentRef.current = true
		setIsAborting(true)
		emit('abort_transcribe')
	}, [patch])

	const cancelAll = useCallback(() => {
		abortAllRef.current = true
		abortCurrentRef.current = true
		setIsAborting(true)
		emit('abort_transcribe')
		discardPendingAudio()
		commit(
			jobsRef.current.map((job) =>
				job.status === 'queued' || (job.privateAudioPath && job.id !== activeIdRef.current)
					? { ...job, privateAudioPath: undefined, status: 'cancelled' }
					: job,
			),
		)
	}, [commit, discardPendingAudio])

	/** Clear the view for a new transcription. A run in progress keeps going in the background. */
	const reset = useCallback(() => {
		const kept = keepForNewView(jobsRef.current, runningRef.current)
		// Pinned, so the next file of the run does not pull the view away from the start screen.
		pinnedRef.current = kept.length > 0
		commit(kept)
		select(null)
	}, [commit, select])

	const selectedJob = jobs.find((job) => job.id === selectedId) ?? null
	const hasResults = jobs.some((job) => job.segments.length > 0 || Boolean(job.summary))

	return {
		jobs,
		activeId,
		selectedId,
		selectedJob,
		running,
		isAborting,
		hasResults,
		selectJob,
		enqueue,
		transcribeJob,
		hydrate,
		relocateSaved,
		updateSegmentText,
		renameJob,
		previewJobName,
		setJobSummary,
		setJobThread,
		setSpeakerName,
		setSegmentSpeaker,
		cancelCurrent,
		cancelAll,
		reset,
		batch,
		batchSummary,
		dismissBatchSummary,
		unexportedCount,
		exportFinished,
	}
}
