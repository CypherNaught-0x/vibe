import * as pathApi from '@tauri-apps/api/path'
import * as dialog from '@tauri-apps/plugin-dialog'
import { useMemo, useState } from 'react'
import { toast } from 'sonner'
import { m } from '~/paraglide/messages.js'
import { createFolder, deleteFolder, moveTranscript, renameFolder } from '~/lib/transcript-folders'
import { notifyTranscriptsChanged, transcriptsFolder, type TranscriptEntry } from '~/lib/transcripts-store'
import type { RecentsItem } from '../components/recents-sidebar'
import { useOpenFolders } from '../components/recents-folders'
import { useSession } from '../session'

/**
 * Folder state and actions for the recents sidebar: which folders are open, the folder being
 * created, the flattened list the virtualizer renders, and every create / rename / delete / move.
 * Jobs of the open session that point into a moved project are re-pointed, so edits keep saving.
 */
export function useRecentsFolders(entries: TranscriptEntry[], folders: string[], query: string, refresh: () => void) {
	const { queue, preference, startNew } = useSession()
	const openFolders = useOpenFolders()
	/** A folder name being typed; `moving` is the transcript to file into it once it exists. */
	const [creating, setCreating] = useState<{ moving?: TranscriptEntry } | null>(null)
	const projectsPath = preference.projectsPath
	const targetFolder = preference.transcriptFolder

	const items = useMemo<RecentsItem[]>(() => {
		const needle = query.trim().toLowerCase()
		const matches = (entry: TranscriptEntry) => !needle || entry.name.toLowerCase().includes(needle)
		const list: RecentsItem[] = creating ? [{ kind: 'new-folder' }] : []
		for (const folder of folders) {
			const inside = entries.filter((entry) => entry.folder === folder)
			const folderMatches = !!needle && folder.toLowerCase().includes(needle)
			const shown = folderMatches ? inside : inside.filter(matches)
			if (needle && !folderMatches && shown.length === 0) continue
			// A search opens every folder holding a match; otherwise the user's choice holds.
			const open = needle ? true : openFolders.open.has(folder)
			list.push({ kind: 'folder', folder, count: inside.length, open })
			if (open) for (const entry of shown) list.push({ kind: 'entry', entry, nested: true })
		}
		for (const entry of entries) if (entry.folder === undefined && matches(entry)) list.push({ kind: 'entry', entry, nested: false })
		return list
	}, [creating, entries, folders, openFolders.open, query])

	async function move(entry: TranscriptEntry, folder: string | null) {
		const moved = await moveTranscript(projectsPath, entry.path, folder)
		if (moved) {
			queue.relocateSaved(entry.path, moved.path, moved.mediaPath)
			if (folder) openFolders.expand(folder)
		}
		notifyTranscriptsChanged()
	}

	async function create(name: string) {
		const moving = creating?.moving
		setCreating(null)
		const folder = await createFolder(projectsPath, name)
		if (!folder) {
			toast.error(m.folderCreateFailed(), { position: 'bottom-center' })
			return
		}
		openFolders.expand(folder)
		if (moving) await move(moving, folder)
		else refresh()
	}

	async function rename(folder: string, name: string) {
		const next = await renameFolder(projectsPath, folder, name)
		if (!next) {
			toast.error(m.folderCreateFailed(), { position: 'bottom-center' })
			return
		}
		if (next === folder) return
		// Session jobs opened from inside the folder now live under its new name.
		const root = await transcriptsFolder(projectsPath)
		const from = await pathApi.join(root, folder)
		const to = await pathApi.join(root, next)
		const under = (path: string) => path.startsWith(from) && /^[/\\]/.test(path.slice(from.length))
		for (const job of queue.jobs) {
			if (!job.savedPath || !under(job.savedPath)) continue
			queue.relocateSaved(job.savedPath, to + job.savedPath.slice(from.length), job.path && under(job.path) ? to + job.path.slice(from.length) : undefined)
		}
		openFolders.rename(folder, next)
		if (targetFolder === folder) preference.setTranscriptFolder(next)
		notifyTranscriptsChanged()
	}

	async function remove(folder: string) {
		const confirmed = await dialog.ask(m.deleteFolderBody({ name: folder }), { title: m.deleteFolder(), kind: 'warning' })
		if (!confirmed) return
		// Through `move`, so open jobs follow their projects back to the top level.
		for (const entry of entries) if (entry.folder === folder) await move(entry, null)
		if (!(await deleteFolder(projectsPath, folder))) toast.warning(m.folderKept(), { position: 'bottom-center' })
		if (targetFolder === folder) preference.setTranscriptFolder(null)
		notifyTranscriptsChanged()
	}

	function newIn(folder: string) {
		preference.setTranscriptFolder(folder)
		openFolders.expand(folder)
		startNew()
	}

	return {
		items,
		targetFolder,
		clearTarget: () => preference.setTranscriptFolder(null),
		startCreating: (moving?: TranscriptEntry) => setCreating({ moving }),
		cancelCreating: () => setCreating(null),
		create: (name: string) => void create(name),
		move: (entry: TranscriptEntry, folder: string | null) => void move(entry, folder),
		rename: (folder: string, name: string) => void rename(folder, name),
		remove: (folder: string) => void remove(folder),
		toggle: openFolders.toggle,
		newIn,
	}
}
