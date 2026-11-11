// Mock handlers for the transcription / model commands (Server backend).
// Simulates model loading, streaming transcription and downloads with real timers
// so the UI exercises the same progress/abort code paths as the desktop runtime.
import { emitMockEvent, onMockEvent } from '../event-bus'
import { DEFAULT_MODEL_FILE, DOWNLOAD_TICKS, DOWNLOAD_TICK_MS, MODELS_FOLDER, TRANSCRIBE_SEGMENT_INTERVAL_MS, sampleSegments, virtualFs } from '../state'
import type { CommandHandlerMap } from '../types'

interface Segment {
	start: number
	stop: number
	text: string
	speaker?: number
}

interface ServerError {
	code: string
	message: string
}

function serverError(code: string, message: string): ServerError {
	return { code, message }
}

function sleep(ms: number) {
	return new Promise<void>((resolve) => setTimeout(resolve, ms))
}

const MOCK_LANGUAGES = [
	'en',
	'zh',
	'de',
	'es',
	'ru',
	'ko',
	'fr',
	'ja',
	'pt',
	'tr',
	'pl',
	'ca',
	'nl',
	'ar',
	'sv',
	'it',
	'id',
	'hi',
	'fi',
	'vi',
	'he',
	'uk',
	'el',
	'ms',
	'cs',
	'ro',
	'da',
	'hu',
	'ta',
	'no',
	'th',
	'ur',
	'hr',
	'bg',
	'lt',
	'la',
	'mi',
	'ml',
	'cy',
	'sk',
	'te',
	'fa',
	'lv',
	'bn',
	'sr',
	'az',
	'sl',
	'kn',
	'et',
	'mk',
]

const AUDIO8_FILENAME = 'Audio8-ASR-Infinite.audio8.json'
const AUDIO8_MARKER = { engine: 'audio8', checkpoint: 'Edge0/Audio8-ASR-Infinite', revision: '7476824bc222e4ad509d286e8cae8b8d3f371129' }

function isAudio8Path(path: unknown) {
	return typeof path === 'string' && path.toLowerCase().endsWith('.audio8.json')
}

function validAudio8Marker(path: string) {
	try {
		const content = virtualFs.get(path)
		const marker = JSON.parse(typeof content === 'string' ? content : '') as Record<string, unknown>
		return (
			Object.keys(marker).length === 3 &&
			marker.engine === AUDIO8_MARKER.engine &&
			marker.checkpoint === AUDIO8_MARKER.checkpoint &&
			marker.revision === AUDIO8_MARKER.revision
		)
	} catch {
		return false
	}
}

// Module level state, reset only by the mock itself.
let apiBaseUrl: string | null = null

function fileExists(path: unknown): boolean {
	return typeof path === 'string' && virtualFs.has(path)
}

/** Emits `download_progress` tuples; resolves false when aborted mid-flight. */
async function runDownloadTicks(abortEvent: string): Promise<boolean> {
	const total = 100_000_000
	let aborted = false
	const unsub = onMockEvent(abortEvent, () => {
		aborted = true
	})
	try {
		for (let tick = 1; tick <= DOWNLOAD_TICKS; tick += 1) {
			await sleep(DOWNLOAD_TICK_MS)
			if (aborted) return false
			const current = Math.round((total * tick) / DOWNLOAD_TICKS)
			emitMockEvent('download_progress', [current, total])
		}
		return !aborted
	} finally {
		unsub()
	}
}

function runTranscribe(
	options: Record<string, unknown>,
): Promise<{ processing_time: { secs: number; nanos: number }; segments: Segment[]; word_segments: undefined }> {
	return new Promise((resolve, reject) => {
		if (!fileExists(options.path)) {
			reject(serverError('invalid_request', `File not found: ${String(options.path ?? '')}`))
			return
		}

		const diarize = options.diarize === true || (typeof options.diarize_model === 'string' && options.diarize_model.length > 0)
		const segments: Segment[] = sampleSegments.map((segment, index) =>
			diarize ? { ...segment, speaker: index % 2 } : { start: segment.start, stop: segment.stop, text: segment.text },
		)

		const startedAt = Date.now()
		let timer: ReturnType<typeof setInterval> | null = null
		let unsub: (() => void) | null = null
		let index = 0
		let settled = false

		function cleanup() {
			if (timer !== null) {
				clearInterval(timer)
				timer = null
			}
			if (unsub) {
				unsub()
				unsub = null
			}
		}

		function onAbort() {
			if (settled) return
			settled = true
			cleanup()
			reject(serverError('aborted', 'transcription aborted'))
		}

		function tick() {
			if (settled) return
			const segment = segments[index]
			index += 1
			emitMockEvent('transcribe_progress', Math.round((index / segments.length) * 100))
			if (segment) {
				emitMockEvent('new_segment', segment)
			}
			if (index >= segments.length) {
				settled = true
				cleanup()
				resolve({
					processing_time: { secs: Math.max(1, Math.round((Date.now() - startedAt) / 1000)), nanos: 0 },
					segments,
					word_segments: undefined,
				})
			}
		}

		unsub = onMockEvent('abort_transcribe', onAbort)
		emitMockEvent('transcribe_progress', 0)
		timer = setInterval(tick, TRANSCRIBE_SEGMENT_INTERVAL_MS)
	})
}

export const serverHandlers: CommandHandlerMap = {
	get_models_folder: () => MODELS_FOLDER,

	add_audio8_model: () => {
		const path = `${MODELS_FOLDER}/${AUDIO8_FILENAME}`
		virtualFs.set(path, JSON.stringify(AUDIO8_MARKER))
		return path
	},

	check_audio8_model_files: (args) =>
		(args.paths as string[]).map((path) => ({
			path,
			valid: validAudio8Marker(path),
			size: String(virtualFs.get(path) ?? '').length,
			reason: validAudio8Marker(path) ? null : 'Invalid or unsupported Audio8 marker',
		})),

	get_model_metadata: (args) => {
		if (isAudio8Path(args.modelPath)) {
			if (!validAudio8Marker(String(args.modelPath))) throw new Error('Invalid or unsupported Audio8 marker')
			return {
				format: 'audio8',
				capabilities: {
					engine: 'audio8',
					requires_vad: false,
					languages: ['en', 'zh'],
					language_detection: false,
					streaming: false,
					translation: false,
					timestamps: false,
					text_prompts: false,
				},
			}
		}
		return {
			format: 'gguf',
			capabilities: {
				engine: 'whisper',
				requires_vad: false,
				languages: MOCK_LANGUAGES,
				language_detection: true,
				streaming: true,
				translation: true,
				timestamps: true,
				text_prompts: true,
			},
		}
	},

	load_model: async () => {
		await sleep(300)
		return 'ok'
	},

	transcribe: (args) => {
		const options = (args.options ?? {}) as Record<string, unknown>
		return runTranscribe(options)
	},

	download_model: async (args) => {
		const path = String(args.path ?? `${MODELS_FOLDER}/${DEFAULT_MODEL_FILE}`)
		const completed = await runDownloadTicks('abort_download')
		if (!completed) {
			return { status: 'cancelled' }
		}
		virtualFs.set(path, null)
		return { status: 'completed', path }
	},

	download_file: async (args) => {
		const path = String(args.path ?? '')
		const completed = await runDownloadTicks('abort_download')
		if (completed && path) {
			virtualFs.set(path, null)
		}
		return undefined
	},

	get_gpu_devices: () => [{ index: 0, name: 'Mock GPU (Apple M-series)', description: 'mock', type: 'integrated' }],

	get_api_base_url: () => apiBaseUrl,

	start_api_server: () => {
		apiBaseUrl = 'http://127.0.0.1:33333'
		return apiBaseUrl
	},

	stop_api_server: () => {
		apiBaseUrl = null
		return true
	},

	is_avx2_enabled: () => true,

	get_commit_hash: () => 'mockhash',

	get_cargo_features: () => ['vulkan'],

	get_logs: () =>
		[
			{ fields: { message: 'mock log line' } },
			{ fields: { message: 'mock log line: model loaded' } },
			{ fields: { message: 'mock log line: ERROR simulated failure' } },
		]
			.map((line) => JSON.stringify(line))
			.join('\n'),
}
