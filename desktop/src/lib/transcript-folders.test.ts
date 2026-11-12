import { beforeEach, describe, expect, it, vi } from 'vitest'

const disk = vi.hoisted(() => ({
	files: new Map<string, string | Uint8Array>(),
	directories: new Set<string>(),
	failCopies: false,
	failWrites: 0,
	failTemporaryRenames: 0,
	rejectReplacement: false,
	failBackupRename: false,
	failBackupRemoval: false,
	failRemovalPath: '',
	failReadDir: false,
}))

vi.mock('@tauri-apps/api/path', async () => {
	const path = await import('node:path')
	return {
		documentDir: vi.fn(async () => '/documents'),
		join: vi.fn(async (...parts: string[]) => path.posix.join(...parts)),
		dirname: vi.fn(async (value: string) => path.posix.dirname(value)),
		basename: vi.fn(async (value: string) => path.posix.basename(value)),
	}
})

vi.mock('@tauri-apps/plugin-fs', () => ({
	exists: vi.fn(async (path: string) => disk.files.has(path) || disk.directories.has(path)),
	mkdir: vi.fn(async (path: string, options?: { recursive?: boolean }) => {
		if (disk.directories.has(path) && !options?.recursive) throw new Error(`exists: ${path}`)
		disk.directories.add(path)
	}),
	writeTextFile: vi.fn(async (path: string, value: string) => {
		if (disk.failWrites > 0) {
			disk.failWrites -= 1
			throw new Error('disk full')
		}
		disk.files.set(path, value)
	}),
	readTextFile: vi.fn(async (path: string) => {
		const value = disk.files.get(path)
		if (typeof value !== 'string') throw new Error(`not text: ${path}`)
		return value
	}),
	copyFile: vi.fn(async (source: string, target: string) => {
		if (disk.failCopies) throw new Error('copy failed')
		const value = disk.files.get(source)
		if (value === undefined) throw new Error(`missing: ${source}`)
		disk.files.set(target, value)
	}),
	rename: vi.fn(async (source: string, target: string) => {
		if (source.includes('.tmp-') && disk.failTemporaryRenames > 0) {
			disk.failTemporaryRenames -= 1
			throw new Error('temporary publish failed')
		}
		if (disk.rejectReplacement && disk.files.has(target)) throw new Error(`cannot replace: ${target}`)
		if (disk.failBackupRename && target.includes('.bak-')) throw new Error('backup rename failed')
		if (disk.directories.has(source)) {
			if (disk.directories.has(target)) throw new Error(`exists: ${target}`)
			disk.directories.delete(source)
			disk.directories.add(target)
			for (const directory of [...disk.directories]) {
				if (!directory.startsWith(`${source}/`)) continue
				disk.directories.delete(directory)
				disk.directories.add(`${target}${directory.slice(source.length)}`)
			}
			for (const [path, value] of [...disk.files]) {
				if (!path.startsWith(`${source}/`)) continue
				disk.files.delete(path)
				disk.files.set(`${target}${path.slice(source.length)}`, value)
			}
			return
		}
		const value = disk.files.get(source)
		if (value === undefined) throw new Error(`missing: ${source}`)
		disk.files.set(target, value)
		disk.files.delete(source)
	}),
	remove: vi.fn(async (path: string, options?: { recursive?: boolean }) => {
		if (disk.failRemovalPath === path) throw new Error('artifact cleanup failed')
		if (disk.failBackupRemoval && path.includes('.bak-')) throw new Error('backup cleanup failed')
		disk.files.delete(path)
		if (options?.recursive) {
			for (const file of [...disk.files.keys()]) if (file.startsWith(`${path}/`)) disk.files.delete(file)
			for (const directory of [...disk.directories]) if (directory === path || directory.startsWith(`${path}/`)) disk.directories.delete(directory)
		}
	}),
	readDir: vi.fn(async (path: string) => {
		if (disk.failReadDir) throw new Error('directory listing failed')
		const prefix = `${path}/`
		return [...new Set([...disk.files.keys(), ...disk.directories])]
			.filter((entry) => entry.startsWith(prefix) && !entry.slice(prefix.length).includes('/'))
			.map((entry) => ({
				name: entry.slice(prefix.length),
				isFile: disk.files.has(entry),
				isDirectory: disk.directories.has(entry),
				isSymlink: false,
			}))
	}),
}))

import { createFolder, deleteFolder, moveTranscript, renameFolder } from './transcript-folders'
import { listTranscriptsAndFolders, readTranscript, saveTranscript } from './transcripts-store'

// Folder tests never exercise the failure switches, but the shared fs mock reads them.
const root = '/documents/Vibe'
const createdAt = new Date(2026, 7, 26, 10, 20, 30)
const segments = [{ start: 0, stop: 100, text: 'Hello' }]

beforeEach(() => {
	vi.clearAllMocks()
	disk.files.clear()
	disk.directories.clear()
	disk.directories.add('/documents')
})

async function recordWithAudio(name: string, folder?: string | null) {
	disk.files.set('/tmp/recording.wav', 'audio')
	const saved = await saveTranscript({ name, sourcePath: '/tmp/recording.wav', moveSourceMedia: true, segments, createdAt, folder })
	if (!saved) throw new Error('save failed')
	return saved
}

describe('transcript folders', () => {
	it('lists empty folders and files projects saved into a folder under it', async () => {
		expect(await createFolder(null, 'Clients')).toBe('Clients')
		expect(await createFolder(null, 'Q3.planning')).toBe('Q3.planning')
		const saved = await recordWithAudio('Call', 'Clients')
		expect(saved.recordPath).toBe(`${root}/Clients/Call-20260826-102030/transcript.vibe.json`)
		await recordWithAudio('Standup')

		const listing = await listTranscriptsAndFolders(null)
		expect(listing.folders).toEqual(['Clients', 'Q3.planning'])
		expect(listing.entries.map((entry) => [entry.name, entry.folder])).toEqual(
			expect.arrayContaining([
				['Call', 'Clients'],
				['Standup', undefined],
			]),
		)
	})

	it('moves a project into a folder and back, keeping its media path inside the project', async () => {
		await createFolder(null, 'Clients')
		const saved = await recordWithAudio('Call')
		const moved = await moveTranscript(null, saved.recordPath, 'Clients')
		expect(moved).toEqual({
			path: `${root}/Clients/Call-20260826-102030/transcript.vibe.json`,
			mediaPath: `${root}/Clients/Call-20260826-102030/audio.wav`,
		})
		expect((await readTranscript(moved!.path))?.sourcePath).toBe(moved!.mediaPath)

		const back = await moveTranscript(null, moved!.path, null)
		expect(back?.path).toBe(saved.recordPath)
		expect(await moveTranscript(null, back!.path, null)).toBeNull()
	})

	it('suffixes a clashing project instead of overwriting it', async () => {
		await createFolder(null, 'Clients')
		const inFolder = await recordWithAudio('Call', 'Clients')
		const atTop = await recordWithAudio('Call')
		const moved = await moveTranscript(null, atTop.recordPath, 'Clients')
		expect(moved?.path).toBe(`${root}/Clients/Call-2-20260826-102030/transcript.vibe.json`)
		expect(disk.files.has(inFolder.recordPath)).toBe(true)
	})

	it('refuses to rename onto an existing folder', async () => {
		await createFolder(null, 'A')
		await createFolder(null, 'B')
		expect(await renameFolder(null, 'A', 'B')).toBeNull()
		expect(await renameFolder(null, 'A', 'C')).toBe('C')
		expect((await listTranscriptsAndFolders(null)).folders).toEqual(['B', 'C'])
	})

	it('deleting a folder moves its transcripts back to the top level', async () => {
		await createFolder(null, 'Clients')
		await recordWithAudio('Call', 'Clients')
		expect(await deleteFolder(null, 'Clients')).toBe(true)
		const listing = await listTranscriptsAndFolders(null)
		expect(listing.folders).toEqual([])
		expect(listing.entries.map((entry) => [entry.name, entry.folder])).toEqual([['Call', undefined]])
	})

	it('keeps a folder that still holds files Vibe did not put there', async () => {
		await createFolder(null, 'Clients')
		disk.files.set(`${root}/Clients/notes.txt`, 'mine')
		expect(await deleteFolder(null, 'Clients')).toBe(false)
		expect(disk.files.has(`${root}/Clients/notes.txt`)).toBe(true)
	})
})
