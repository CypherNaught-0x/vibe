import * as pathApi from '@tauri-apps/api/path'
import * as fs from '@tauri-apps/plugin-fs'
import {
	basename,
	isProjectRecordPath,
	isUserFolderName,
	listTranscriptsAndFolders,
	readTranscript,
	serializeMetadata,
	stampPattern,
	TRANSCRIPT_EXTENSION,
	TRANSCRIPT_FILENAME,
	toFolderName,
	transcriptsFolder,
	writeRecordAtomic,
} from './transcripts-store'

/**
 * User folders: named subfolders of the transcript store that group projects.
 *
 * A folder is nothing but a directory one level below the store root, so it is just as visible (and
 * movable) in the file manager. Projects move between folders by moving their project folder (or a
 * legacy flat file); records keep no folder field, the location is the single source of truth.
 * Every operation is best-effort and reports failure as null/false, like the rest of the store.
 */

export interface MovedTranscript {
	/** New absolute path of the record. */
	path: string
	/** Stable playback path after the move: the project's media copy, else the original source. */
	mediaPath: string
}

/** Insert `-<attempt>` before the trailing stamp (and extension), so retries stay sortable. */
function withAttempt(name: string, attempt: number) {
	if (attempt === 1) return name
	const extension = name.endsWith(TRANSCRIPT_EXTENSION) ? TRANSCRIPT_EXTENSION : ''
	const stem = extension ? name.slice(0, -extension.length) : name
	const match = stem.match(stampPattern)
	const at = match?.index ?? stem.length
	return `${stem.slice(0, at)}-${attempt}${stem.slice(at)}${extension}`
}

async function rootOf(projectsPath?: string | null) {
	return transcriptsFolder(projectsPath)
}

/**
 * Create a user folder. Creating one that already exists is fine and returns its name.
 * @returns the (sanitized) folder name, or null when the name is unusable or creation failed.
 */
export async function createFolder(projectsPath: string | null | undefined, name: string): Promise<string | null> {
	const folder = toFolderName(name)
	if (!folder || !isUserFolderName(folder)) return null
	try {
		const target = await pathApi.join(await rootOf(projectsPath), folder)
		if (await fs.exists(target)) {
			// A project of that name is not a folder the user can file into.
			if (await fs.exists(await pathApi.join(target, TRANSCRIPT_FILENAME))) return null
			return folder
		}
		await fs.mkdir(target, { recursive: true })
		return folder
	} catch (error) {
		console.warn('failed to create transcript folder:', name, error)
		return null
	}
}

/**
 * Rename a user folder. Fails (null) when another folder or project already has the new name, so two
 * folders never silently merge.
 */
export async function renameFolder(projectsPath: string | null | undefined, from: string, to: string): Promise<string | null> {
	const next = toFolderName(to)
	if (!next || !isUserFolderName(next)) return null
	if (next === from) return from
	try {
		const root = await rootOf(projectsPath)
		const source = await pathApi.join(root, from)
		const target = await pathApi.join(root, next)
		// Case-only renames on case-insensitive disks report the target as existing; allow those.
		if (next.toLowerCase() !== from.toLowerCase() && (await fs.exists(target))) return null
		await fs.rename(source, target)
		return next
	} catch (error) {
		console.warn('failed to rename transcript folder:', from, error)
		return null
	}
}

/**
 * Remove a user folder without losing anything: its transcripts move back to the top level first,
 * then the folder is removed only if that left it empty (files the user keeps there survive).
 * @returns whether the folder is gone.
 */
export async function deleteFolder(projectsPath: string | null | undefined, folder: string): Promise<boolean> {
	try {
		const { entries } = await listTranscriptsAndFolders(projectsPath)
		for (const entry of entries) {
			if (entry.folder === folder) await moveTranscript(projectsPath, entry.path, null)
		}
		const target = await pathApi.join(await rootOf(projectsPath), folder)
		const remaining = await fs.readDir(target)
		if (remaining.some((item) => item.name !== '.DS_Store')) return false
		await fs.remove(target, { recursive: true })
		return true
	} catch (error) {
		console.warn('failed to delete transcript folder:', folder, error)
		return false
	}
}

/**
 * Move one transcript into a user folder, or back to the top level with `folder = null`. Project
 * folders move whole (media copy included); legacy flat files move alone. The name and timestamp
 * stay; a clash with a project already in the destination gets a `-2`, `-3`… suffix.
 * @returns the new location, or null when nothing moved.
 */
export function moveTranscript(projectsPath: string | null | undefined, path: string, folder: string | null): Promise<MovedTranscript | null> {
	return serializeMetadata(path, () => moveTranscriptRecord(projectsPath, path, folder))
}

async function moveTranscriptRecord(projectsPath: string | null | undefined, path: string, folder: string | null): Promise<MovedTranscript | null> {
	try {
		const record = await readTranscript(path)
		if (!record) return null
		const root = await rootOf(projectsPath)
		const destination = folder ? await pathApi.join(root, toFolderName(folder)) : root
		if (!(await fs.exists(destination))) await fs.mkdir(destination, { recursive: true })

		const project = isProjectRecordPath(path)
		const source = project ? await pathApi.dirname(path) : path
		if ((await pathApi.dirname(source)) === destination) return null

		const name = basename(source)
		let target = ''
		for (let attempt = 1; ; attempt += 1) {
			const candidate = await pathApi.join(destination, withAttempt(name, attempt))
			if (await fs.exists(candidate)) continue
			try {
				await fs.rename(source, candidate)
				target = candidate
				break
			} catch (error) {
				if (await fs.exists(candidate)) continue
				throw error
			}
		}

		if (!project) return { path: target, mediaPath: record.sourcePath }

		const recordPath = await pathApi.join(target, TRANSCRIPT_FILENAME)
		if (!record.audioFile) return { path: recordPath, mediaPath: record.sourcePath }
		const mediaPath = await pathApi.join(target, record.audioFile)
		// Vibe-created media lives only in the project, so its record points inside the folder.
		if (record.sourcePath === (await pathApi.join(source, record.audioFile))) {
			try {
				await writeRecordAtomic(recordPath, { ...record, sourcePath: mediaPath })
			} catch (error) {
				// The move itself succeeded; playback still resolves through `audioFile`.
				console.warn('failed to update the moved transcript source path:', recordPath, error)
			}
		}
		return { path: recordPath, mediaPath }
	} catch (error) {
		console.warn('failed to move transcript:', path, error)
		return null
	}
}
