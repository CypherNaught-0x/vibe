/**
 * File format of stored disk recordings. The backend (`start_record`, `recording_encoder.rs`)
 * owns the encoder settings; anything it does not recognise is stored as Opus.
 */
export const recordingFormats = ['opus', 'm4a', 'mp3', 'flac', 'wav'] as const
export type RecordingFormat = (typeof recordingFormats)[number]

/** Opus: 48 kHz mono, 48 kbps VBR in Ogg — ~22 MB per hour, transparent for speech. */
export const defaultRecordingFormat: RecordingFormat = 'opus'

/** A stored value from an older or hand-edited config falls back to the default. */
export function toRecordingFormat(value: unknown): RecordingFormat {
	return recordingFormats.includes(value as RecordingFormat) ? (value as RecordingFormat) : defaultRecordingFormat
}
