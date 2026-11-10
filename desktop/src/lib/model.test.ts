import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
	AUDIO8_MODEL_FILENAME,
	checkModelFiles,
	getFriendlyModelName,
	getModelExtension,
	isModelFile,
	listInstalledModels,
	withoutUnsupportedOptions,
	type ModelCapabilities,
} from './model'
import { serverHandlers } from '../mock-tauri/handlers/server'
import { MODELS_FOLDER, virtualFs } from '../mock-tauri/state'

const mocks = vi.hoisted(() => ({ invoke: vi.fn(), lsFiles: vi.fn() }))
vi.mock('@tauri-apps/api/core', () => ({ invoke: mocks.invoke }))
vi.mock('./fs', () => ({ lsFiles: mocks.lsFiles }))

beforeEach(() => {
	mocks.invoke.mockReset()
	mocks.lsFiles.mockReset()
})
afterEach(() => vi.restoreAllMocks())

const whisper: ModelCapabilities = {
	engine: 'whisper',
	requires_vad: false,
	languages: ['en'],
	language_detection: true,
	streaming: true,
	translation: true,
	timestamps: true,
	text_prompts: true,
}

const options = { lang: 'en', init_prompt: 'Punctuate. Use full sentences.', translate: true, n_threads: 4 }

describe('withoutUnsupportedOptions', () => {
	it('does not send timestamp preferences that Audio8 cannot honour', () => {
		const timestampOptions = { ...options, word_timestamps: true, stable_timestamps: true }
		expect(withoutUnsupportedOptions(timestampOptions, { ...whisper, engine: 'audio8', timestamps: false })).toEqual(options)
		expect(timestampOptions.word_timestamps).toBe(true)
	})

	it('sends everything to a model that takes prompts and translation', () => {
		expect(withoutUnsupportedOptions(options, whisper)).toEqual(options)
	})

	it('drops the prompt and translation for engines that reject them, keeping the rest', () => {
		expect(withoutUnsupportedOptions(options, { ...whisper, engine: 'parakeet', text_prompts: false, translation: false })).toEqual({
			lang: 'en',
			n_threads: 4,
		})
	})

	it('leaves the options alone when no capabilities are known yet', () => {
		expect(withoutUnsupportedOptions(options, null)).toEqual(options)
		expect(withoutUnsupportedOptions(options, undefined)).toBe(options)
	})
})

describe('Audio8 model discovery', () => {
	it('recognises the compound marker extension without accepting arbitrary JSON files', () => {
		expect(getModelExtension(AUDIO8_MODEL_FILENAME)).toBe('audio8.json')
		expect(getModelExtension('AUDIO8.AUDIO8.JSON')).toBe('audio8.json')
		expect(isModelFile('notes.json')).toBe(false)
		expect(isModelFile('model.audio8Xjson')).toBe(false)
		expect(isModelFile('model.audio8.json.part')).toBe(false)
		expect(getModelExtension('model.GGUF')).toBe('gguf')
		expect(getFriendlyModelName(AUDIO8_MODEL_FILENAME)).toBe('Audio8-ASR-Infinite')
	})

	it('checks markers separately from weights and keeps the listing order', async () => {
		const marker = { name: AUDIO8_MODEL_FILENAME, path: `/models/${AUDIO8_MODEL_FILENAME}` }
		const weight = { name: 'ggml-medium.bin', path: '/models/ggml-medium.bin' }
		mocks.lsFiles.mockResolvedValue([marker, weight, { name: 'notes.json', path: '/models/notes.json' }])
		mocks.invoke.mockImplementation(async (_command, { paths }: { paths: string[] }) =>
			paths.map((path) => ({ path, valid: true, size: 100, reason: null })),
		)
		expect(await listInstalledModels('/models')).toEqual([
			{ ...marker, valid: true, reason: null },
			{ ...weight, valid: true, reason: null },
		])
		expect(mocks.invoke).toHaveBeenCalledWith('check_model_files', { paths: [weight.path] })
		expect(mocks.invoke).toHaveBeenCalledWith('check_audio8_model_files', { paths: [marker.path] })
	})

	it('fails closed when marker validation is unavailable, unlike legacy weight checks', async () => {
		vi.spyOn(console, 'error').mockImplementation(() => {})
		mocks.invoke.mockRejectedValue(new Error('Validation unavailable'))
		const results = await checkModelFiles(['/models/model.audio8.json', '/models/model.bin'])
		expect(results[0]).toMatchObject({ valid: false, reason: 'Error: Validation unavailable' })
		expect(results[1]).toMatchObject({ valid: true, reason: null })
	})

	it('does not call native validation for an empty model folder', async () => {
		expect(await checkModelFiles([])).toEqual([])
		expect(mocks.invoke).not.toHaveBeenCalled()
	})
})

describe('Nemotron 3.5 Transcribe discovery', () => {
	it('lists the decoder but not the speech encoder the server loads next to it', async () => {
		const decoder = { name: 'NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-AD-IQ4_NL.gguf', path: '/models/decoder.gguf' }
		const encoder = { name: 'NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-Encoder-F16.gguf', path: '/models/encoder.gguf' }
		const streaming = { name: 'nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf', path: '/models/streaming.gguf' }
		mocks.lsFiles.mockResolvedValue([decoder, encoder, streaming])
		mocks.invoke.mockImplementation(async (_command, { paths }: { paths: string[] }) =>
			paths.map((path) => ({ path, valid: true, size: 100, reason: null })),
		)
		expect((await listInstalledModels('/models')).map((model) => model.name)).toEqual([decoder.name, streaming.name])
	})
})

describe('Audio8 browser mock', () => {
	const markerPath = `${MODELS_FOLDER}/${AUDIO8_MODEL_FILENAME}`
	afterEach(() => virtualFs.delete(markerPath))

	it('adds only the pinned marker and reports the actual supported capabilities', async () => {
		const before = virtualFs.size
		expect(await serverHandlers.add_audio8_model({})).toBe(markerPath)
		expect(virtualFs.size).toBe(before + 1)
		expect(JSON.parse(String(virtualFs.get(markerPath)))).toEqual({
			engine: 'audio8',
			checkpoint: 'Edge0/Audio8-ASR-Infinite',
			revision: '7476824bc222e4ad509d286e8cae8b8d3f371129',
		})
		const metadata = (await serverHandlers.get_model_metadata({ modelPath: markerPath })) as { capabilities: ModelCapabilities }
		expect(metadata.capabilities).toEqual({
			engine: 'audio8',
			requires_vad: false,
			languages: ['en', 'zh'],
			language_detection: false,
			streaming: false,
			translation: false,
			timestamps: false,
			text_prompts: false,
		})
		expect(withoutUnsupportedOptions(options, metadata.capabilities)).toEqual({ lang: 'en', n_threads: 4 })
	})

	it('rejects an unpinned or remote-code-enabled marker', async () => {
		await serverHandlers.add_audio8_model({})
		const marker = JSON.parse(String(virtualFs.get(markerPath)))
		for (const invalid of [
			{ ...marker, revision: 'main' },
			{ ...marker, trust_remote_code: true },
		]) {
			virtualFs.set(markerPath, JSON.stringify(invalid))
			expect(await serverHandlers.check_audio8_model_files({ paths: [markerPath] })).toEqual([
				expect.objectContaining({ path: markerPath, valid: false }),
			])
			expect(() => serverHandlers.get_model_metadata({ modelPath: markerPath })).toThrow('Invalid or unsupported Audio8 marker')
		}
	})
})
