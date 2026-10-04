import { useRef } from 'react'
import { toast } from 'sonner'
import { m } from '~/paraglide/messages.js'
import { autoProjectName } from '~/lib/project-name'
import { notifyTranscriptsChanged, saveTranscript, TRANSCRIPT_VERSION, type TranscriptRecord } from '~/lib/transcripts-store'
import type { LiveRecordingCompletion } from '~/pages/home/hooks/use-live-recording'
import { usePreferenceProvider } from '~/providers/preference'
import type { TranscribeQueue } from './use-transcribe-queue'

/** A live finish is already transcribed. Persist text only and never enqueue it for a second pass. */
export function useLiveRecordingProject(queue: TranscribeQueue, onShown: () => void): LiveRecordingCompletion {
	const preference = usePreferenceProvider()
	const latest = useRef({ preference, hydrate: queue.hydrate, onShown })
	latest.current = { preference, hydrate: queue.hydrate, onShown }

	return async (result, isCurrent) => {
		const { preference: current } = latest.current
		const name = autoProjectName(result.name, 'record')
		const record: TranscriptRecord = {
			version: TRANSCRIPT_VERSION,
			name,
			sourcePath: '',
			createdAt: result.createdAt.toISOString(),
			language: result.language,
			modelPath: result.modelPath,
			segments: result.segments,
		}
		let savedPath = ''
		let saveFailed = false
		if (current.saveTranscripts) {
			try {
				const saved = await saveTranscript({
					name,
					sourcePath: '',
					noAudioStorage: true,
					projectsPath: current.projectsPath,
					segments: record.segments,
					language: record.language,
					modelPath: record.modelPath,
					createdAt: result.createdAt,
				})
				savedPath = saved?.recordPath ?? ''
				saveFailed = !saved
				if (saved) notifyTranscriptsChanged()
			} catch (error) {
				console.error('Failed to save live transcript:', error)
				saveFailed = true
			}
		}
		// Slow disk completion cannot replace a newer capture/session or clear its panel.
		if (!isCurrent()) return
		latest.current.hydrate(record, savedPath, null, 'record')
		latest.current.onShown()
		if (saveFailed) {
			toast.error(m.error(), {
				description: 'Failed to save the live transcript; it remains visible in memory.',
				position: 'bottom-center',
			})
		}
	}
}
