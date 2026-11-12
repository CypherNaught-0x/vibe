import { useCallback, useRef, type MutableRefObject } from 'react'
import { createSerialExecutor } from '~/lib/privacy'
import type { Segment, SpeakerNames } from '~/lib/transcript'
import {
	notifyTranscriptsChanged,
	renameTranscript,
	saveTranscript,
	updateTranscriptSegments,
	updateTranscriptSpeakerNames,
	updateTranscriptSummary,
	updateTranscriptThread,
	type AiThreadEntry,
	type SaveTranscriptResult,
	type TranscriptEntry,
} from '~/lib/transcripts-store'
import type { Preference } from '~/providers/preference'
import type { Job } from './use-transcribe-queue'

/** Re-read the live title after every rename so slow disk I/O cannot restore a stale draft. */
export async function reconcileProjectName(
	initialName: string,
	saved: SaveTranscriptResult,
	latestName: () => string | undefined,
	rename: (path: string, name: string) => Promise<TranscriptEntry | null> = renameTranscript,
	onRenamed?: (saved: SaveTranscriptResult) => void,
) {
	let current = saved
	let appliedName = initialName
	for (;;) {
		const desiredName = latestName()
		if (!desiredName || desiredName === appliedName) return current
		const renamed = await rename(current.recordPath, desiredName)
		if (!renamed) return current
		current = { recordPath: renamed.path, mediaPath: renamed.mediaPath ?? current.mediaPath }
		appliedName = desiredName
		onRenamed?.(current)
	}
}

/** All project writes, including the first autosave, share one per-job operation queue. */
export function useProjectPersistence(
	jobs: MutableRefObject<Job[]>,
	preference: MutableRefObject<Preference>,
	patch: (id: string, changes: Partial<Job>) => void,
) {
	const serial = useRef(createSerialExecutor()).current
	const find = useCallback((id: string) => jobs.current.find((job) => job.id === id), [jobs])

	const save = useCallback(
		async (job: Job, segments: Segment[]): Promise<boolean> => {
			const current = preference.current
			if (job.savedPath || !current.saveTranscripts) return true
			const saved = await saveTranscript({
				name: job.name,
				sourcePath: job.path,
				projectsPath: current.projectsPath,
				folder: job.folder,
				moveSourceMedia: job.source === 'record' || job.source === 'url',
				noAudioStorage: current.noAudioStorage,
				segments,
				language: current.modelOptions.lang,
				modelPath: current.modelPath,
			})
			if (!saved) return false
			patch(job.id, { savedPath: saved.recordPath, path: saved.mediaPath })
			const final = await reconcileProjectName(
				job.name,
				saved,
				() => find(job.id)?.name,
				renameTranscript,
				(step) => patch(job.id, { savedPath: step.recordPath, path: step.mediaPath }),
			)
			patch(job.id, { savedPath: final.recordPath, path: final.mediaPath })
			notifyTranscriptsChanged()
			return true
		},
		[find, patch, preference],
	)

	const persist = useCallback(
		(job: Job, segments: Segment[]) =>
			serial(job.id, async () => {
				const current = find(job.id)
				if (!current || current.transcriptDeleted || !segments.length) return false
				if (!current.savedPath) return save(current, segments)
				const saved = await updateTranscriptSegments(current.savedPath, segments)
				if (saved) notifyTranscriptsChanged()
				return saved
			}),
		[find, save, serial],
	)

	const writeSegments = useCallback(
		(id: string, segments: Segment[]) => {
			void serial(id, async () => {
				const job = find(id)
				if (job?.savedPath && !job.transcriptDeleted) await updateTranscriptSegments(job.savedPath, segments)
			})
		},
		[find, serial],
	)

	const updateSegmentText = useCallback(
		(id: string, index: number, text: string) => {
			const job = find(id)
			const segment = job?.segments[index]
			if (!job || job.transcriptDeleted || !segment || segment.text === text) return
			const segments = job.segments.map((item, position) => (position === index ? { ...item, text } : item))
			patch(id, { segments })
			writeSegments(id, segments)
		},
		[find, patch, writeSegments],
	)

	const setSegmentSpeaker = useCallback(
		(id: string, index: number, speaker: number) => {
			const job = find(id)
			const segment = job?.segments[index]
			if (!job || job.transcriptDeleted || !segment || segment.speaker === speaker) return
			const segments = job.segments.map((item, position) => (position === index ? { ...item, speaker } : item))
			patch(id, { segments })
			writeSegments(id, segments)
		},
		[find, patch, writeSegments],
	)

	const renameJob = useCallback(
		(id: string, name: string) =>
			serial(id, async () => {
				const next = name.trim()
				const job = find(id)
				if (!job || !next) return false
				if (!job.savedPath) {
					patch(id, { name: next })
					return true
				}
				const renamed = await renameTranscript(job.savedPath, next)
				if (!renamed) return false
				const latest = find(id)
				patch(id, {
					...(latest?.name === job.name || latest?.name === next ? { name: next } : {}),
					savedPath: renamed.path,
					path: renamed.mediaPath ?? job.path,
				})
				notifyTranscriptsChanged()
				return true
			}),
		[find, patch, serial],
	)

	const previewJobName = useCallback((id: string, name: string) => patch(id, { name }), [patch])

	const setJobSummary = useCallback(
		(id: string, summary: string) =>
			serial(id, async () => {
				let job = find(id)
				if (!job || !summary.trim()) return false
				const deleteRaw = preference.current.deleteTranscriptAfterAi
				if (!job.savedPath && preference.current.saveTranscripts) {
					if (!(await save(job, job.segments))) return false
					job = find(id)
					if (!job?.savedPath) return false
				}
				if (job.savedPath && !(await updateTranscriptSummary(job.savedPath, summary, deleteRaw))) return false
				// Only the durable atomic replacement (or an intentionally RAM-only session) permits redaction.
				patch(id, { summary, ...(deleteRaw ? { segments: [], speakerNames: undefined, thread: undefined, transcriptDeleted: true } : {}) })
				if (job.savedPath) notifyTranscriptsChanged()
				return true
			}),
		[find, patch, preference, save, serial],
	)

	const setJobThread = useCallback(
		(id: string, thread: AiThreadEntry[]) => {
			const wasDeleted = find(id)?.transcriptDeleted
			void serial(id, async () => {
				const job = find(id)
				if (!job || job.transcriptDeleted !== wasDeleted) return
				patch(id, { thread })
				if (job.savedPath) await updateTranscriptThread(job.savedPath, thread, Boolean(job.transcriptDeleted))
			})
		},
		[find, patch, serial],
	)

	const setSpeakerName = useCallback(
		(id: string, speaker: number, name: string) => {
			const job = find(id)
			if (!job || job.transcriptDeleted) return
			const next = name.trim()
			if ((job.speakerNames?.[speaker] ?? '') === next) return
			const speakerNames: SpeakerNames = { ...job.speakerNames }
			if (next) speakerNames[speaker] = next
			else delete speakerNames[speaker]
			patch(id, { speakerNames })
			void serial(id, async () => {
				const current = find(id)
				if (current?.savedPath && !current.transcriptDeleted) await updateTranscriptSpeakerNames(current.savedPath, speakerNames)
			})
		},
		[find, patch, serial],
	)

	return { persist, updateSegmentText, setSegmentSpeaker, renameJob, previewJobName, setJobSummary, setJobThread, setSpeakerName }
}
