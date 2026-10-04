// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { m } from '~/paraglide/messages.js'
import { PrivacySection } from './privacy'
import type { SettingsViewModel } from './shared'

afterEach(cleanup)

function renderPrivacy(noAudioStorage = false, deleteTranscriptAfterAi = false) {
	const setNoAudioStorage = vi.fn()
	const setDeleteTranscriptAfterAi = vi.fn()
	const vm = {
		preference: {
			analyticsEnabled: true,
			saveTranscripts: true,
			noAudioStorage,
			deleteTranscriptAfterAi,
			setNoAudioStorage,
			setDeleteTranscriptAfterAi,
		},
	} as unknown as SettingsViewModel
	render(<PrivacySection vm={vm} />)
	return { setNoAudioStorage, setDeleteTranscriptAfterAi }
}

describe('privacy retention settings', () => {
	it('shows default-off controls and changes each preference independently', () => {
		const setters = renderPrivacy()
		const audio = screen.getByRole('switch', { name: m.privacyNoAudioStorage() })
		const transcript = screen.getByRole('switch', { name: m.privacyDeleteTranscriptAfterAi() })
		expect(audio.getAttribute('aria-checked')).toBe('false')
		expect(transcript.getAttribute('aria-checked')).toBe('false')
		fireEvent.click(audio)
		expect(setters.setNoAudioStorage).toHaveBeenCalledExactlyOnceWith(true)
		expect(setters.setDeleteTranscriptAfterAi).not.toHaveBeenCalled()
		fireEvent.click(transcript)
		expect(setters.setDeleteTranscriptAfterAi).toHaveBeenCalledExactlyOnceWith(true)
	})

	it('shows enabled preferences and does not truncate the limitations or warnings', () => {
		renderPrivacy(true, true)
		expect(screen.getByRole('switch', { name: m.privacyNoAudioStorage() }).getAttribute('aria-checked')).toBe('true')
		expect(screen.getByRole('switch', { name: m.privacyDeleteTranscriptAfterAi() }).getAttribute('aria-checked')).toBe('true')
		expect(screen.getByText(m.privacyNoAudioStorageInfo()).className).not.toContain('line-clamp')
		expect(screen.getByText(m.privacyDeleteTranscriptAfterAiInfo()).className).not.toContain('line-clamp')
		expect(screen.getByText(m.privacyRetentionWarning())).toBeTruthy()
	})
})
