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

import * as pathApi from '@tauri-apps/api/path'
import * as fs from '@tauri-apps/plugin-fs'
import {
	readTranscript,
	renameTranscript,
	resolveProjectAudio,
	saveTranscript,
	updateTranscriptSegments,
	updateTranscriptSpeakerNames,
	updateTranscriptSummary,
	updateTranscriptThread,
	type TranscriptRecord,
} from './transcripts-store'

const createdAt = new Date(2026, 7, 26, 10, 20, 30)
const segments = [{ start: 0, stop: 100, text: 'Hello' }]

beforeEach(() => {
	vi.clearAllMocks()
	disk.files.clear()
	disk.directories.clear()
	disk.failCopies = false
	disk.failWrites = 0
	disk.failTemporaryRenames = 0
	disk.rejectReplacement = false
	disk.failBackupRename = false
	disk.failBackupRemoval = false
	disk.failRemovalPath = ''
	disk.failReadDir = false
})

describe('project media ownership', () => {
	it('persists an untranscribed recording as a playable empty project', async () => {
		disk.files.set('/temp/meeting.wav', new Uint8Array([1, 2, 3]))

		const saved = await saveTranscript({
			name: 'Record-meeting',
			sourcePath: '/temp/meeting.wav',
			projectsPath: '/media/projects',
			moveSourceMedia: true,
			createdAt,
			segments: [],
		})

		const record = await readTranscript(saved!.recordPath)
		expect(record?.segments).toEqual([])
		expect(record?.sourcePath).toBe(saved?.mediaPath)
		expect(disk.files.has(saved!.mediaPath)).toBe(true)
		expect(disk.files.has('/temp/meeting.wav')).toBe(false)
	})

	it('moves Vibe-created media into the selected projects folder', async () => {
		disk.files.set('/temp/capture.wav', new Uint8Array([1, 2, 3]))

		const recordPath = await saveTranscript({
			name: 'Record-capture',
			sourcePath: '/temp/capture.wav',
			projectsPath: '/media/projects',
			moveSourceMedia: true,
			createdAt,
			segments,
		})

		expect(recordPath?.recordPath).toBe('/media/projects/Record-capture-20260826-102030/transcript.vibe.json')
		expect(recordPath?.mediaPath).toBe('/media/projects/Record-capture-20260826-102030/audio.wav')
		expect(disk.files.has('/temp/capture.wav')).toBe(false)
		const record = await readTranscript(recordPath!.recordPath)
		const audioPath = record && (await resolveProjectAudio(recordPath!.recordPath, record))
		expect(audioPath).toBe('/media/projects/Record-capture-20260826-102030/audio.wav')
		expect(record?.sourcePath).toBe(audioPath)
	})

	it.each(['opus', 'm4a', 'mp3', 'flac'])('keeps the extension of a %s recording when moving it into its project', async (extension) => {
		disk.files.set(`/temp/2026-08-26 10-20-30.${extension}`, new Uint8Array([1]))

		const saved = await saveTranscript({
			name: 'Record-capture',
			sourcePath: `/temp/2026-08-26 10-20-30.${extension}`,
			projectsPath: '/media/projects',
			moveSourceMedia: true,
			createdAt,
			segments,
		})

		expect(saved?.mediaPath).toBe(`/media/projects/Record-capture-20260826-102030/audio.${extension}`)
		const record = await readTranscript(saved!.recordPath)
		expect(record && (await resolveProjectAudio(saved!.recordPath, record))).toBe(saved?.mediaPath)
	})

	it('copies imported media and leaves the user-owned source untouched', async () => {
		disk.files.set('/user/interview.m4a', new Uint8Array([4, 5, 6]))

		const recordPath = await saveTranscript({
			name: 'File-interview',
			sourcePath: '/user/interview.m4a',
			projectsPath: '/media/projects',
			createdAt,
			segments,
		})

		expect(disk.files.has('/user/interview.m4a')).toBe(true)
		const record = await readTranscript(recordPath!.recordPath)
		expect(record?.sourcePath).toBe('/user/interview.m4a')
		expect(await resolveProjectAudio(recordPath!.recordPath, record!)).toBe('/media/projects/File-interview-20260826-102030/audio.m4a')
	})

	it('keeps staged media and removes the incomplete project when its copy fails', async () => {
		disk.files.set('/temp/capture.wav', new Uint8Array([1]))
		disk.failCopies = true

		const saved = await saveTranscript({
			name: 'Record-capture',
			sourcePath: '/temp/capture.wav',
			projectsPath: '/media/projects',
			moveSourceMedia: true,
			createdAt,
			segments,
		})

		expect(saved).toBeNull()
		expect(disk.files.has('/temp/capture.wav')).toBe(true)
		expect(disk.directories.has('/media/projects/Record-capture-20260826-102030')).toBe(false)
	})

	it('keeps staged media when final metadata cannot be written', async () => {
		disk.files.set('/temp/capture.wav', new Uint8Array([1]))
		disk.failWrites = 1

		const saved = await saveTranscript({
			name: 'Record-capture',
			sourcePath: '/temp/capture.wav',
			projectsPath: '/media/projects',
			moveSourceMedia: true,
			createdAt,
			segments,
		})

		expect(saved).toBeNull()
		expect(disk.files.has('/temp/capture.wav')).toBe(true)
	})

	it('saves imported metadata against the external source when copying fails', async () => {
		disk.files.set('/user/interview.m4a', new Uint8Array([1]))
		disk.failCopies = true

		const saved = await saveTranscript({
			name: 'File-interview',
			sourcePath: '/user/interview.m4a',
			projectsPath: '/media/projects',
			createdAt,
			segments,
		})

		expect(saved?.mediaPath).toBe('/user/interview.m4a')
		expect((await readTranscript(saved!.recordPath))?.sourcePath).toBe('/user/interview.m4a')
	})

	it('atomically reserves distinct folders for concurrent same-second saves', async () => {
		disk.files.set('/user/one.wav', new Uint8Array([1]))
		disk.files.set('/user/two.wav', new Uint8Array([2]))
		const input = { name: 'Meeting', projectsPath: '/media/projects', createdAt, segments }

		const [first, second] = await Promise.all([
			saveTranscript({ ...input, sourcePath: '/user/one.wav' }),
			saveTranscript({ ...input, sourcePath: '/user/two.wav' }),
		])

		expect(first?.recordPath).not.toBe(second?.recordPath)
		expect(new Set([first?.recordPath, second?.recordPath])).toEqual(
			new Set(['/media/projects/Meeting-20260826-102030/transcript.vibe.json', '/media/projects/Meeting-2-20260826-102030/transcript.vibe.json']),
		)
	})

	it('renames project media paths and avoids an occupied target', async () => {
		disk.files.set('/temp/capture.wav', new Uint8Array([1]))
		const saved = await saveTranscript({
			name: 'Original',
			sourcePath: '/temp/capture.wav',
			projectsPath: '/media/projects',
			moveSourceMedia: true,
			createdAt,
			segments,
		})
		disk.directories.add('/media/projects/Renamed-20260826-102030')

		const renamed = await renameTranscript(saved!.recordPath, 'Renamed')
		const record = await readTranscript(renamed!.path)

		expect(renamed?.path).toBe('/media/projects/Renamed-2-20260826-102030/transcript.vibe.json')
		expect(record?.sourcePath).toBe('/media/projects/Renamed-2-20260826-102030/audio.wav')
		expect(renamed?.mediaPath).toBe(record?.sourcePath)
	})

	it('rolls the project folder back when renamed metadata cannot be written', async () => {
		disk.files.set('/temp/capture.wav', new Uint8Array([1]))
		const saved = await saveTranscript({
			name: 'Original',
			sourcePath: '/temp/capture.wav',
			projectsPath: '/media/projects',
			moveSourceMedia: true,
			createdAt,
			segments,
		})
		disk.failWrites = 1

		expect(await renameTranscript(saved!.recordPath, 'Broken')).toBeNull()
		expect(await readTranscript(saved!.recordPath)).not.toBeNull()
		expect(disk.directories.has('/media/projects/Broken-20260826-102030')).toBe(false)
	})
})

describe('private audio persistence', () => {
	it('skips imported audio copies without touching the user-owned original', async () => {
		const original = new Uint8Array([4, 5, 6])
		disk.files.set('/user/interview.m4a', original)
		const saved = await saveTranscript({
			name: 'Private interview',
			sourcePath: '/user/interview.m4a',
			projectsPath: '/media/projects',
			noAudioStorage: true,
			createdAt,
			segments,
		})

		expect(saved).not.toBeNull()
		expect(fs.copyFile).not.toHaveBeenCalled()
		expect(fs.remove).not.toHaveBeenCalledWith('/user/interview.m4a')
		expect(disk.files.get('/user/interview.m4a')).toBe(original)
		const record = await readTranscript(saved!.recordPath)
		expect(record?.audioFile).toBeUndefined()
		expect(record?.sourcePath).toBe('/user/interview.m4a')
		expect(await resolveProjectAudio(saved!.recordPath, record!)).toBeNull()
		expect([...disk.files.keys()].filter((path) => path.startsWith('/media/projects/'))).toEqual([saved!.recordPath])
	})

	it.each([false, true])('never passes a RAM handle to path/fs APIs or persists it (privacy preference %s)', async (noAudioStorage) => {
		const handle = 'vibe-memory:recording.wav'
		const saved = await saveTranscript({
			name: 'RAM recording',
			sourcePath: handle,
			projectsPath: '/media/projects',
			moveSourceMedia: true,
			noAudioStorage,
			createdAt,
			segments,
		})

		expect(saved?.mediaPath).toBe('')
		const record = await readTranscript(saved!.recordPath)
		expect(record?.sourcePath).toBe('')
		expect(record?.audioFile).toBeUndefined()
		expect(await resolveProjectAudio(saved!.recordPath, record!)).toBeNull()
		expect(disk.files.get(saved!.recordPath)).not.toContain(handle)
		expect(fs.copyFile).not.toHaveBeenCalled()
		for (const api of [...Object.values(pathApi), ...Object.values(fs)]) {
			if (!vi.isMockFunction(api)) continue
			expect(api.mock.calls.flat()).not.toContain(handle)
		}
	})

	it('drops Vibe-owned staging audio only after saving metadata without a playback path', async () => {
		disk.files.set('/temp/capture.wav', new Uint8Array([1]))
		const saved = await saveTranscript({
			name: 'Private capture',
			sourcePath: '/temp/capture.wav',
			projectsPath: '/media/projects',
			moveSourceMedia: true,
			noAudioStorage: true,
			createdAt,
			segments,
		})

		expect(saved?.mediaPath).toBe('')
		expect((await readTranscript(saved!.recordPath))?.sourcePath).toBe('')
		expect(fs.copyFile).not.toHaveBeenCalled()
		expect(disk.files.has('/temp/capture.wav')).toBe(false)
		expect(vi.mocked(fs.rename).mock.invocationCallOrder[0]).toBeLessThan(vi.mocked(fs.remove).mock.invocationCallOrder[0])
	})

	it('reports failed initial atomic publication and retains staging audio', async () => {
		disk.files.set('/temp/capture.wav', new Uint8Array([1]))
		disk.failTemporaryRenames = 1
		const saved = await saveTranscript({
			name: 'Private capture',
			sourcePath: '/temp/capture.wav',
			projectsPath: '/media/projects',
			moveSourceMedia: true,
			noAudioStorage: true,
			createdAt,
			segments,
		})

		expect(saved).toBeNull()
		expect([...disk.files.keys()]).toEqual(['/temp/capture.wav'])
		expect(disk.directories.has('/media/projects/Private capture-20260826-102030')).toBe(false)
	})
})

const recordPath = '/media/projects/Meeting/transcript.vibe.json'
const rawRecord: TranscriptRecord = {
	version: 1,
	name: 'Meeting',
	sourcePath: '/user/interview.m4a',
	createdAt: createdAt.toISOString(),
	language: 'en',
	modelPath: '/models/model.bin',
	audioFile: 'audio.m4a',
	segments,
	speakerNames: { 0: 'Private speaker' },
	summary: 'Earlier derived result',
	thread: [{ question: 'Private question', answer: 'Raw quoted answer' }],
}

function seedRawRecord() {
	const original = JSON.stringify(rawRecord)
	disk.files.set(recordPath, original)
	return original
}

function durableRecord(): TranscriptRecord {
	return JSON.parse(disk.files.get(recordPath) as string)
}

describe('atomic transcript redaction', () => {
	it('durably keeps the derived result and project metadata while deleting raw text, speakers and history', async () => {
		seedRawRecord()
		expect(await updateTranscriptSummary(recordPath, 'Derived result', true)).toBe(true)

		const record = durableRecord()
		expect(record).toEqual({
			version: 1,
			name: 'Meeting',
			sourcePath: rawRecord.sourcePath,
			createdAt: rawRecord.createdAt,
			language: 'en',
			modelPath: rawRecord.modelPath,
			audioFile: 'audio.m4a',
			segments: [],
			summary: 'Derived result',
			transcriptDeleted: true,
		})
		expect(await readTranscript(recordPath)).toMatchObject({ segments: [], summary: 'Derived result', transcriptDeleted: true })
		expect([...disk.files.keys()]).toEqual([recordPath])
	})

	it('preserves raw content when redaction is not requested', async () => {
		seedRawRecord()
		expect(await updateTranscriptSummary(recordPath, 'Derived result')).toBe(true)
		expect(durableRecord()).toEqual({ ...rawRecord, summary: 'Derived result' })
	})

	it.each(['', ' \n\t '])('rejects empty AI output without modifying the durable raw record (%j)', async (summary) => {
		const original = seedRawRecord()
		expect(await updateTranscriptSummary(recordPath, summary, true)).toBe(false)
		expect(disk.files.get(recordPath)).toBe(original)
		expect(fs.writeTextFile).not.toHaveBeenCalled()
	})

	it('serializes redaction before late segment, speaker and old-thread writes so none resurrect raw content', async () => {
		seedRawRecord()
		const results = await Promise.all([
			updateTranscriptSummary(recordPath, 'Derived result', true),
			updateTranscriptSegments(recordPath, [{ start: 0, stop: 1, text: 'Late raw edit' }]),
			updateTranscriptSpeakerNames(recordPath, { 0: 'Late speaker' }),
			updateTranscriptThread(recordPath, rawRecord.thread!),
		])

		expect(results).toEqual([true, false, false, false])
		expect(durableRecord()).toMatchObject({ segments: [], summary: 'Derived result', transcriptDeleted: true })
		expect(durableRecord().speakerNames).toBeUndefined()
		expect(durableRecord().thread).toBeUndefined()
		expect(fs.writeTextFile).toHaveBeenCalledTimes(1)
	})

	it('allows an explicitly post-redaction question without restoring the deleted history', async () => {
		seedRawRecord()
		await updateTranscriptSummary(recordPath, 'Derived result', true)
		const thread = [{ question: 'About the summary?', answer: 'Derived answer' }]
		expect(await updateTranscriptThread(recordPath, thread, true)).toBe(true)
		expect(durableRecord()).toMatchObject({ segments: [], summary: 'Derived result', transcriptDeleted: true, thread })
	})

	it.each([
		[
			'temporary write',
			() => {
				disk.failWrites = 1
			},
		],
		[
			'temporary publish and retry',
			() => {
				disk.failTemporaryRenames = 2
			},
		],
		[
			'backup creation',
			() => {
				disk.rejectReplacement = true
				disk.failBackupRename = true
			},
		],
		[
			'raw backup cleanup',
			() => {
				disk.rejectReplacement = true
				disk.failBackupRemoval = true
			},
		],
	] as const)('does not report successful deletion when %s fails', async (_stage, fail) => {
		const original = seedRawRecord()
		fail()
		expect(await updateTranscriptSummary(recordPath, 'Derived result', true)).toBe(false)
		expect(disk.files.get(recordPath)).toBe(original)
		expect(await readTranscript(recordPath)).toMatchObject({ segments, speakerNames: rawRecord.speakerNames, thread: rawRecord.thread })
		expect([...disk.files.keys()]).toEqual([recordPath])
	})

	it('removes the raw backup before reporting success on platforms that cannot replace existing files', async () => {
		seedRawRecord()
		disk.rejectReplacement = true
		expect(await updateTranscriptSummary(recordPath, 'Derived result', true)).toBe(true)
		expect(durableRecord()).toMatchObject({ segments: [], summary: 'Derived result', transcriptDeleted: true })
		expect([...disk.files.keys()]).toEqual([recordPath])
		expect(fs.remove).toHaveBeenCalledWith(expect.stringContaining('.bak-'))
	})

	it('purges raw artifacts from earlier failed cleanup before publishing a later redaction', async () => {
		const original = seedRawRecord()
		disk.rejectReplacement = true
		disk.failBackupRemoval = true
		expect(await updateTranscriptSummary(recordPath, 'Earlier successful summary')).toBe(true)
		const staleBackup = [...disk.files.keys()].find((path) => path.startsWith(`${recordPath}.bak-`))!
		expect(staleBackup).toBeTruthy()
		expect(disk.files.get(staleBackup)).toBe(original)
		const staleTemporary = `${recordPath}.tmp-123456789-2`
		disk.files.set(staleTemporary, original)
		disk.rejectReplacement = false
		disk.failBackupRemoval = false
		vi.mocked(fs.remove).mockClear()
		vi.mocked(fs.writeTextFile).mockClear()

		expect(await updateTranscriptSummary(recordPath, 'Derived result', true)).toBe(true)
		expect([...disk.files.keys()]).toEqual([recordPath])
		expect(durableRecord()).toMatchObject({ segments: [], summary: 'Derived result', transcriptDeleted: true })
		for (const artifact of [staleBackup, staleTemporary]) {
			const removal = vi.mocked(fs.remove).mock.calls.findIndex(([path]) => path === artifact)
			expect(removal).toBeGreaterThanOrEqual(0)
			expect(vi.mocked(fs.remove).mock.invocationCallOrder[removal]).toBeLessThan(vi.mocked(fs.writeTextFile).mock.invocationCallOrder[0])
		}
	})

	it.each(['bak', 'tmp'])('fails closed without publishing redaction if an older .%s artifact cannot be removed', async (kind) => {
		const original = seedRawRecord()
		const artifact = `${recordPath}.${kind}-123456789-3`
		disk.files.set(artifact, original)
		disk.failRemovalPath = artifact

		expect(await updateTranscriptSummary(recordPath, 'Derived result', true)).toBe(false)
		expect(disk.files.get(recordPath)).toBe(original)
		expect(disk.files.get(artifact)).toBe(original)
		expect(fs.writeTextFile).not.toHaveBeenCalled()
	})

	it('fails closed if the directory cannot be inspected for older raw artifacts', async () => {
		const original = seedRawRecord()
		disk.failReadDir = true
		expect(await updateTranscriptSummary(recordPath, 'Derived result', true)).toBe(false)
		expect(disk.files.get(recordPath)).toBe(original)
		expect(fs.writeTextFile).not.toHaveBeenCalled()
	})

	it('purges only strictly named artifacts of this record, leaving user exports and directories alone', async () => {
		const original = seedRawRecord()
		const artifact = `${recordPath}.bak-123456789-4`
		disk.files.set(artifact, original)
		const preserved = [
			`${recordPath}.bak-manual`,
			`${recordPath}.bak-123456789-x`,
			`${recordPath}.bak-123456789-4.pdf`,
			`${recordPath}.tmp-123456789`,
			`${recordPath}.tmp-123456789-4-copy`,
			'/media/projects/Meeting/other.vibe.json.bak-123456789-4',
			'/media/projects/Meeting/notes.md',
		]
		for (const path of preserved) disk.files.set(path, 'User-owned export')
		const directory = `${recordPath}.tmp-123456789-5`
		disk.directories.add(directory)
		disk.files.set(`${directory}/export.txt`, 'User-owned nested export')

		expect(await updateTranscriptSummary(recordPath, 'Derived result', true)).toBe(true)
		expect(disk.files.has(artifact)).toBe(false)
		for (const path of preserved) {
			expect(disk.files.get(path)).toBe('User-owned export')
			expect(fs.remove).not.toHaveBeenCalledWith(path)
		}
		expect(disk.directories.has(directory)).toBe(true)
		expect(disk.files.get(`${directory}/export.txt`)).toBe('User-owned nested export')
		expect(fs.remove).not.toHaveBeenCalledWith(directory)
	})
})
