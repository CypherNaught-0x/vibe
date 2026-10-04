import { invoke } from '@tauri-apps/api/core'
import * as pathExt from '@tauri-apps/api/path'
import * as fsExt from '@tauri-apps/plugin-fs'
import {
	diarizeModelFilename,
	embeddingModelFilename,
	legacyDiarizeModelFilenames,
	segmentModelFilename,
	vadModelFilename,
	type ModelDownload,
	type ModelIntegrity,
} from './config'
import { lsFiles } from './fs'
import { NamedPath } from './types'

export const MODEL_EXTENSIONS = ['bin', 'gguf', 'audio8.json'] as const
export type ModelExtension = (typeof MODEL_EXTENSIONS)[number]

const MODEL_EXTENSION_PATTERN = new RegExp(`\\.(${MODEL_EXTENSIONS.map((extension) => extension.replace('.', '\\.')).join('|')})$`, 'i')
export const AUDIO8_MODEL_FILENAME = 'Audio8-ASR-Infinite.audio8.json'
export const AUDIO8_MODEL_URL = 'https://huggingface.co/Edge0/Audio8-ASR-Infinite'

type DownloadModelResult = { status: 'completed'; path: string } | { status: 'cancelled' }

export function getModelExtension(filename: string): ModelExtension | null {
	const extension = filename.match(MODEL_EXTENSION_PATTERN)?.[1]?.toLowerCase()
	return MODEL_EXTENSIONS.includes(extension as ModelExtension) ? (extension as ModelExtension) : null
}

export function isGgufModel(filename: string) {
	return getModelExtension(filename) === 'gguf'
}

export function isAudio8Model(filename: string) {
	return getModelExtension(filename) === 'audio8.json'
}

export function randomString(length: number, prefix: string, suffix: string) {
	const chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
	let result = prefix
	for (let i = 0; i < length; i++) {
		result += chars.charAt(Math.floor(Math.random() * chars.length))
	}
	return result + suffix
}

export async function getFilenameFromUrl(url: string) {
	const urlObj = new URL(url)
	const fileName = urlObj.pathname.split('/').pop() || ''
	return fileName
}

export function getFriendlyModelName(filename: string) {
	if (filename.toLowerCase() === AUDIO8_MODEL_FILENAME.toLowerCase()) return 'Audio8-ASR-Infinite'
	const name = filename.replace(MODEL_EXTENSION_PATTERN, '').replace(/^ggml[-_]?/, '')
	if (!name || name === 'model') return 'Custom model'
	return name.replace(/[-_]+/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

/** A model file on disk, together with the verdict of the backend integrity check. */
export interface InstalledModel extends NamedPath {
	valid: boolean
	/** Why the file was rejected — shown to the user so "corrupt" is not a mystery. */
	reason: string | null
}

interface ModelFileCheck {
	path: string
	valid: boolean
	size: number
	reason: string | null
}

export async function getModelsFolder() {
	return invoke<string>('get_models_folder')
}

/** Weights use magic bytes; Audio8 markers must contain only the supported, pinned checkpoint. */
export async function checkModelFiles(paths: string[]): Promise<ModelFileCheck[]> {
	const weights = paths.filter((path) => !isAudio8Model(path))
	const markers = paths.filter(isAudio8Model)
	async function check(command: string, files: string[], failOpen: boolean): Promise<ModelFileCheck[]> {
		if (files.length === 0) return []
		try {
			return await invoke<ModelFileCheck[]>(command, { paths: files })
		} catch (error) {
			console.error('failed to check model files:', error)
			return files.map((path) => ({ path, valid: failOpen, size: 0, reason: failOpen ? null : String(error) }))
		}
	}
	const [weightChecks, markerChecks] = await Promise.all([check('check_model_files', weights, true), check('check_audio8_model_files', markers, false)])
	const byPath = new Map([...weightChecks, ...markerChecks].map((result) => [result.path, result]))
	return paths.map((path) => byPath.get(path) ?? { path, valid: false, size: 0, reason: 'Model validation returned no result' })
}

/** Delete leftover `*.part` files — partial downloads cannot be resumed, so they only take space. */
export async function cleanupPartialDownloads(folder?: string): Promise<string[]> {
	try {
		return await invoke<string[]>('cleanup_partial_downloads', { folder: folder ?? (await getModelsFolder()) })
	} catch (error) {
		console.error('failed to clean up partial downloads:', error)
		return []
	}
}

/**
 * Support models that share the models folder — and, for the Silero VAD, even the GGML magic — but
 * are never transcription models. The gates that need them address them by exact path, so leaving
 * them out of this listing does not make them look missing.
 */
const AUXILIARY_MODEL_FILENAMES = [vadModelFilename, diarizeModelFilename, ...legacyDiarizeModelFilenames, embeddingModelFilename, segmentModelFilename]

export function isAuxiliaryModelFile(filename: string) {
	return AUXILIARY_MODEL_FILENAMES.some((auxiliary) => auxiliary.toLowerCase() === filename.toLowerCase())
}

/** Every transcription model in the folder, with truncated or corrupt ones flagged rather than hidden. */
export async function listInstalledModels(folder?: string): Promise<InstalledModel[]> {
	const modelsFolder = folder ?? (await getModelsFolder())
	const files = (await lsFiles(modelsFolder)).filter((entry) => isModelFile(entry.name) && !isAuxiliaryModelFile(entry.name))
	const checks = await checkModelFiles(files.map((file) => file.path))
	const byPath = new Map(checks.map((check) => [check.path, check]))
	return files.map((file) => ({ ...file, valid: byPath.get(file.path)?.valid ?? true, reason: byPath.get(file.path)?.reason ?? null }))
}

/**
 * Whether a file the app depends on is usable. Weights are validated against their magic bytes —
 * which now includes the diarization model, a GGUF since it moved off ONNX Runtime. Anything else
 * (the legacy ONNX embedding/segmentation pair, yt-dlp) has no magic, so existence is all we have.
 */
export async function isModelFileUsable(path: string) {
	if (!(await fsExt.exists(path))) return false
	if (!isModelFile(path)) return true
	const [check] = await checkModelFiles([path])
	return check?.valid ?? true
}

interface DownloadModelOptions {
	/** Overwrite exactly this file — used to replace a model that failed its integrity check. */
	replacePath?: string
}

export async function downloadModel(source: string | ModelDownload, options: DownloadModelOptions = {}) {
	const { url, ...integrity }: ModelDownload = typeof source === 'string' ? { url: source } : source
	const modelPath = options.replacePath ?? (await resolveDownloadPath(url))
	const result = await invoke<DownloadModelResult>('download_model', { url, path: modelPath, integrity: toIntegrity(integrity) })
	return result.status === 'completed' ? result.path : null
}

function toIntegrity(integrity: ModelIntegrity): ModelIntegrity | undefined {
	return integrity.size === undefined && integrity.sha256 === undefined ? undefined : integrity
}

/**
 * Where a download should land. A name already taken by a healthy model gets a random suffix so the
 * two can coexist, but one taken by a corrupt file is reused: retrying a failed download must
 * replace the broken file instead of leaving it behind next to a second copy.
 */
async function resolveDownloadPath(url: string) {
	let filename = await getFilenameFromUrl(url)
	if (!isModelFile(filename) || isAudio8Model(filename)) {
		filename = 'ggml-model.bin'
	}
	const modelsFolder = await getModelsFolder()
	const modelPath = await pathExt.join(modelsFolder, filename)
	if (!(await fsExt.exists(modelPath)) || !(await isModelFileUsable(modelPath))) {
		return modelPath
	}
	return pathExt.join(modelsFolder, randomString(8, 'ggml-model_', `.${getModelExtension(filename) ?? 'bin'}`))
}

export function isModelFile(filename: string) {
	return getModelExtension(filename) !== null
}

export interface ModelCapabilities {
	engine: 'whisper' | 'nemotron' | string
	requires_vad: boolean
	languages: string[]
	language_detection: boolean
	streaming: boolean
	translation: boolean
	timestamps: boolean
	text_prompts: boolean
}

export interface ModelMetadata {
	format: string
	capabilities: ModelCapabilities
}

/** The subset of transcribe options that only some engines honour. */
interface EngineSpecificOptions {
	init_prompt?: string
	translate?: boolean
	word_timestamps?: boolean
	stable_timestamps?: boolean
}

/**
 * Drop the Whisper-only options a model cannot use instead of letting server reject the run.
 * A prompt written for Turbo stays saved in settings, so switching to Parakeet or Nemotron
 * and back needs no retyping. Unknown capabilities (no metadata yet) leave the options alone.
 */
export function withoutUnsupportedOptions<T extends EngineSpecificOptions>(options: T, capabilities: ModelCapabilities | null | undefined): T {
	if (!capabilities) return options
	const next = { ...options }
	if (!capabilities.text_prompts) delete next.init_prompt
	if (!capabilities.translation) delete next.translate
	if (!capabilities.timestamps) {
		delete next.word_timestamps
		delete next.stable_timestamps
	}
	return next
}
