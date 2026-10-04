// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { Session } from '../session'
import type { LiveRecordingState } from '~/pages/home/hooks/use-live-recording'
import { TooltipProvider } from '~/components/ui/tooltip'
import IdleHero from '../components/idle-hero'

const mocks = vi.hoisted(() => ({ session: {} as Session, listen: vi.fn() }))
vi.mock('../session', () => ({ useSession: () => mocks.session }))
vi.mock('@tauri-apps/api/event', () => ({ listen: mocks.listen }))
vi.mock('~/components/audio-device-input', () => ({ default: () => null }))
vi.mock('../components/quiet-row', () => ({ default: () => null }))
vi.mock('~/paraglide/messages.js', () => ({
	m: new Proxy(
		{},
		{
			get: (_target, key) => (parameters?: Record<string, string>) =>
				parameters ? `${String(key)} ${Object.values(parameters).join(' ')}` : String(key),
		},
	),
}))

function live(phase: LiveRecordingState['phase']): LiveRecordingState {
	return {
		phase,
		name: 'Meeting',
		segments: [],
		partial: '',
		partialIndex: null,
		startedAt: phase === 'ready' || phase === 'finishing' ? 1000 : null,
		stoppedAt: null,
	}
}
function show() {
	return render(
		<TooltipProvider>
			<IdleHero />
		</TooltipProvider>,
	)
}
beforeEach(() => {
	vi.resetAllMocks()
	mocks.listen.mockResolvedValue(vi.fn())
	mocks.session = {
		panel: 'record',
		setPanel: vi.fn(),
		recordElapsed: 12,
		recording: {
			isRecording: true,
			isLiveRecording: true,
			liveEnabled: true,
			live: live('loading'),
			startRecord: vi.fn(),
			stopRecord: vi.fn(),
			devices: [],
			inputDevice: { id: 'mic' },
			outputDevice: null,
		},
		preference: {
			liveTranscription: true,
			saveTranscripts: true,
			modelPath: '/models/model.audio8.json',
			modelOptions: { lang: 'en' },
			setHomeTab: vi.fn(),
		},
		link: {},
	} as unknown as Session
})
afterEach(cleanup)

describe('live recording panel', () => {
	it.each(['loading', 'calibrating'] as const)('shows %s and a cancellable request without pretending capture has begun', (phase) => {
		mocks.session.recording.live = live(phase)
		show()
		expect(screen.getByRole('status').textContent).toContain(phase === 'loading' ? 'liveLoading' : 'liveCalibrating')
		expect(screen.queryByText('00:12')).toBeNull()
		fireEvent.click(screen.getByRole('button', { name: 'cancel' }))
		expect(mocks.session.recording.stopRecord).toHaveBeenCalledOnce()
		expect(mocks.listen).not.toHaveBeenCalled()
	})

	it('shows committed/partial captions, RAM-only audio policy and calibrated buffer/device', () => {
		mocks.session.recording.live = {
			...live('ready'),
			segments: [{ start: 0, stop: 100, text: 'Committed caption' }],
			partial: 'Current partial',
			partialIndex: 1,
			calibration: { window_tokens: 128, pcm_capacity_samples: 32000, step_seconds: 0.05, realtime: true, device: 'mps' },
		}
		show()
		expect(screen.getByRole('log', { name: 'liveCaptions' }).textContent).toBe('Committed captionCurrent partial')
		expect(screen.getByText('liveBenchmarkInfo 2.0 128').textContent).toBe('liveBenchmarkInfo 2.0 128 · mps')
		expect(screen.getByText('liveAudioRamOnly')).toBeTruthy()
		expect(screen.getByText('saveTranscript: enabled')).toBeTruthy()
		expect(screen.getByText('00:12')).toBeTruthy()
		expect(mocks.listen).toHaveBeenCalledWith('record_level', expect.any(Function))
	})

	it('shows graceful finishing with Stop disabled', () => {
		mocks.session.recording.live = live('finishing')
		show()
		expect(screen.getByRole('status').textContent).toContain('liveFinishing')
		expect((screen.getByRole('button', { name: 'stopRecording' }) as HTMLButtonElement).disabled).toBe(true)
	})

	it('retains captions and the error after failure, with transcript saving explicitly disabled', () => {
		mocks.session.recording.isRecording = false
		mocks.session.recording.isLiveRecording = false
		mocks.session.recording.live = { ...live('error'), partial: 'Retained partial', partialIndex: 0, error: 'Device failed' }
		mocks.session.preference.saveTranscripts = false
		show()
		expect(screen.getByRole('alert').textContent).toBe('Device failed')
		expect(screen.getByRole('log').textContent).toBe('Retained partial')
		expect(screen.getByText('saveTranscript: disabled')).toBeTruthy()
	})

	it('prevents unsupported languages from starting live capture', () => {
		mocks.session.recording.isRecording = false
		mocks.session.recording.live = null
		mocks.session.preference.modelOptions.lang = 'auto'
		show()
		expect((screen.getByRole('button', { name: 'startRecord' }) as HTMLButtonElement).disabled).toBe(true)
		expect(screen.getByText('liveTranscriptionInfo')).toBeTruthy()
	})

	it('keeps the legacy recording panel unchanged', () => {
		mocks.session.recording.isLiveRecording = false
		mocks.session.recording.live = null
		show()
		expect(screen.getByText('00:12')).toBeTruthy()
		expect(screen.getByRole('button', { name: 'stopRecording' })).toBeTruthy()
		expect(screen.queryByRole('log')).toBeNull()
		expect(screen.queryByText('liveAudioRamOnly')).toBeNull()
	})
})
