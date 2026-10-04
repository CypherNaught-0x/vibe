// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { TooltipProvider } from '~/components/ui/tooltip'
import { m } from '~/paraglide/messages.js'
import { ModelsSection } from './models'
import type { SettingsViewModel } from './shared'

afterEach(cleanup)

function renderModels(isAddingAudio8 = false) {
	const addAudio8Model = vi.fn()
	const openAudio8Source = vi.fn()
	const vm = {
		models: [],
		corruptModels: [],
		gpuDevices: [],
		isMacOS: true,
		downloadURL: '',
		preference: { modelPath: null, modelDisplayNames: {}, noGpu: false },
		isAddingAudio8,
		addAudio8Model,
		openAudio8Source,
	} as unknown as SettingsViewModel
	render(
		<TooltipProvider>
			<ModelsSection vm={vm} />
		</TooltipProvider>,
	)
	return { addAudio8Model, openAudio8Source }
}

describe('Audio8 setup controls', () => {
	it('discloses the first-load download and capability limits before adding the model', () => {
		const actions = renderModels()
		expect(screen.getByText(m.audio8ModelInfo())).toBeTruthy()
		expect(screen.getByText(m.audio8RuntimeInfo())).toBeTruthy()
		expect(screen.getByText(m.audio8Limitations())).toBeTruthy()
		fireEvent.click(screen.getByRole('button', { name: m.audio8AddModel() }))
		expect(actions.addAudio8Model).toHaveBeenCalledTimes(1)
		fireEvent.click(screen.getByRole('button', { name: 'Hugging Face · Edge0/Audio8-ASR-Infinite' }))
		expect(actions.openAudio8Source).toHaveBeenCalledTimes(1)
	})

	it('disables duplicate setup while the native marker command is pending', () => {
		const actions = renderModels(true)
		const button = screen.getByRole('button', { name: m.audio8AddModel() }) as HTMLButtonElement
		expect(button.disabled).toBe(true)
		expect(button.getAttribute('aria-busy')).toBe('true')
		fireEvent.click(button)
		expect(actions.addAudio8Model).not.toHaveBeenCalled()
	})
})
