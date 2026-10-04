// @vitest-environment jsdom

import { act, cleanup, renderHook } from '@testing-library/react'
import { createElement, type ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const native = vi.hoisted(() => ({
	invoke: vi.fn(),
	store: {
		entries: vi.fn(async () => []),
		onChange: vi.fn(async () => () => {}),
		get: vi.fn(async () => null),
		set: vi.fn(async () => {}),
		delete: vi.fn(async () => {}),
		save: vi.fn(async () => {}),
	},
}))

vi.mock('@tauri-apps/api/core', () => ({ invoke: native.invoke }))
vi.mock('@tauri-apps/api/event', () => ({ listen: vi.fn(async () => () => {}) }))
vi.mock('@tauri-apps/api/webviewWindow', () => ({
	getCurrentWebviewWindow: () => ({ setTheme: vi.fn(async () => {}) }),
}))
vi.mock('@tauri-apps/plugin-store', () => ({ load: vi.fn(async () => native.store) }))
vi.mock('@tauri-apps/plugin-dialog', () => ({ message: vi.fn(async () => {}) }))
vi.mock('./keep-awake', () => ({ KEEP_AWAKE: { background: 'background' }, startKeepAwake: vi.fn(), stopKeepAwake: vi.fn() }))

import { PreferenceProvider, usePreferenceProvider } from '~/providers/preference'
import { CONFIG_KEYS } from './config-keys'
import { deleteConfig, loadConfigStore, writeConfig } from './config-store'
import { createSerialExecutor, discardPrivateAudio, isPrivateAudioPath } from './privacy'

afterEach(cleanup)
beforeEach(async () => {
	deleteConfig(CONFIG_KEYS.noAudioStorage)
	deleteConfig(CONFIG_KEYS.deleteTranscriptAfterAi)
	await loadConfigStore()
	vi.clearAllMocks()
	native.invoke.mockResolvedValue(undefined)
})

function wrapper({ children }: { children: ReactNode }) {
	return createElement(PreferenceProvider, { children })
}

describe('privacy preference persistence', () => {
	it('leaves both privacy controls off when no config value exists', () => {
		const { result } = renderHook(usePreferenceProvider, { wrapper })
		expect(result.current.noAudioStorage).toBe(false)
		expect(result.current.deleteTranscriptAfterAi).toBe(false)
	})

	it('honours explicit stored opt-ins and changes each config key independently', () => {
		writeConfig(CONFIG_KEYS.noAudioStorage, true)
		writeConfig(CONFIG_KEYS.deleteTranscriptAfterAi, true)
		const { result } = renderHook(usePreferenceProvider, { wrapper })
		expect(result.current.noAudioStorage).toBe(true)
		expect(result.current.deleteTranscriptAfterAi).toBe(true)

		act(() => result.current.setNoAudioStorage(false))
		expect(result.current.noAudioStorage).toBe(false)
		expect(result.current.deleteTranscriptAfterAi).toBe(true)
		expect(native.store.set).toHaveBeenCalledWith(CONFIG_KEYS.noAudioStorage, false)
		act(() => result.current.setDeleteTranscriptAfterAi(false))
		expect(result.current.deleteTranscriptAfterAi).toBe(false)
		expect(native.store.set).toHaveBeenCalledWith(CONFIG_KEYS.deleteTranscriptAfterAi, false)
	})
})

describe('private recording handles', () => {
	it('recognises only native RAM handles, not filesystem or ordinary media URLs', () => {
		expect(isPrivateAudioPath('vibe-memory:recording-1')).toBe(true)
		for (const path of ['', '/temp/recording.wav', 'C:\\recording.wav', 'https://example.com/audio.wav', '/user/vibe-memory:notes.wav']) {
			expect(isPrivateAudioPath(path)).toBe(false)
		}
	})

	it('releases a RAM handle through the native command and ignores ordinary paths', async () => {
		await discardPrivateAudio('/user/interview.wav')
		expect(native.invoke).not.toHaveBeenCalled()
		await discardPrivateAudio('vibe-memory:recording-1')
		expect(native.invoke).toHaveBeenCalledExactlyOnceWith('discard_private_audio', { path: 'vibe-memory:recording-1' })
	})

	it('treats an already consumed or failed discard as best-effort cleanup', async () => {
		native.invoke.mockRejectedValueOnce(new Error('already consumed'))
		const log = vi.spyOn(console, 'error').mockImplementation(() => {})
		try {
			await expect(discardPrivateAudio('vibe-memory:consumed')).resolves.toBeUndefined()
			expect(log).toHaveBeenCalled()
		} finally {
			log.mockRestore()
		}
	})
})

function deferred<T>() {
	let resolve!: (value: T) => void
	const promise = new Promise<T>((done) => {
		resolve = done
	})
	return { promise, resolve }
}

describe('per-project serialization', () => {
	it('keeps operations for one job ordered without blocking a different job', async () => {
		const serial = createSerialExecutor()
		const gate = deferred<void>()
		const started: string[] = []
		const first = serial('one', async () => {
			started.push('first')
			await gate.promise
			return 1
		})
		const second = serial('one', async () => {
			started.push('second')
			return 2
		})
		const independent = serial('two', async () => {
			started.push('independent')
			return 3
		})
		expect(await independent).toBe(3)
		expect(started).toEqual(['first', 'independent'])
		gate.resolve()
		expect(await Promise.all([first, second])).toEqual([1, 2])
		expect(started).toEqual(['first', 'independent', 'second'])
	})

	it('does not poison later work when an earlier operation fails', async () => {
		const serial = createSerialExecutor()
		const failed = serial('one', async () => {
			throw new Error('write failed')
		})
		const next = serial('one', async () => 'saved')
		await expect(failed).rejects.toThrow('write failed')
		await expect(next).resolves.toBe('saved')
		await expect(serial('one', async () => 'later')).resolves.toBe('later')
	})
})
