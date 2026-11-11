import { describe, expect, it } from 'vitest'
import { defaultRecordingFormat, recordingFormats, toRecordingFormat } from './recording-format'
import { audioExtensions } from './config'

describe('recording formats', () => {
	it('defaults to Opus, including for unknown stored values', () => {
		expect(defaultRecordingFormat).toBe('opus')
		expect(toRecordingFormat(undefined)).toBe('opus')
		expect(toRecordingFormat('aiff')).toBe('opus')
		expect(toRecordingFormat('flac')).toBe('flac')
	})

	it('only stores files the app itself accepts as media', () => {
		for (const format of recordingFormats) expect(audioExtensions).toContain(format)
	})
})
