import { invoke } from '@tauri-apps/api/core'

/** Native recording handle, never a filesystem or playback path. */
export function isPrivateAudioPath(path: string) {
	return path.startsWith('vibe-memory:')
}

/** Release an unconsumed recording (safe even if transcription already consumed it). */
export async function discardPrivateAudio(path: string) {
	if (!isPrivateAudioPath(path)) return
	try {
		await invoke('discard_private_audio', { path })
	} catch (error) {
		console.error('failed to discard private audio:', error)
	}
}

/** Serialize read/modify/write operations; a failure must not poison later operations. */
export function createSerialExecutor() {
	const pending = new Map<string, Promise<unknown>>()
	return <T>(key: string, operation: () => Promise<T>): Promise<T> => {
		const previous = pending.get(key) ?? Promise.resolve()
		const current = previous.catch(() => undefined).then(operation)
		pending.set(key, current)
		const cleanup = () => {
			if (pending.get(key) === current) pending.delete(key)
		}
		void current.then(cleanup, cleanup)
		return current
	}
}
