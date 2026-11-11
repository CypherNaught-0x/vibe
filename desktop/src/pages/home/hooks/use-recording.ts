import { emit } from '@tauri-apps/api/event'
import { invoke } from '@tauri-apps/api/core'
import { type SetStateAction, useCallback, useContext, useEffect, useRef, useState } from 'react'
import { selectInputDevice, type AudioDevice } from '~/lib/audio'
import { CONFIG_KEYS } from '~/lib/config-keys'
import { usePersisted } from '~/lib/config-store'
import { KEEP_AWAKE, startKeepAwake, stopKeepAwake } from '~/lib/keep-awake'
import { ensureSystemAudioPermission } from '~/lib/permissions'
import { ErrorModalContext } from '~/providers/error-modal'
import { usePreferenceProvider } from '~/providers/preference'
import { supportsLiveRecording, useLiveRecording, type LiveRecordingCompletion } from './use-live-recording'

const STOP_ACK_TIMEOUT_MS = 10_000

export function useRecording(onBeforeStart: () => void, onLiveFinished?: LiveRecordingCompletion) {
	const preference = usePreferenceProvider()
	const { setState: setErrorModal } = useContext(ErrorModalContext)
	const [devices, setDevices] = useState<AudioDevice[]>([])
	const [savedInputDeviceId, setSavedInputDeviceId] = usePersisted<string | null>(CONFIG_KEYS.inputDeviceId, null)
	const [savedOutputDeviceId, setSavedOutputDeviceId] = usePersisted<string | null>(CONFIG_KEYS.outputDeviceId, null)
	const [inputDevice, setInputDevice] = useState<AudioDevice | null>(null)
	const [outputDevice, setOutputDevice] = useState<AudioDevice | null>(null)
	const [isRecording, setIsRecording] = useState(false)
	/** Stop requested until the recording's project is saved (or failed): no second stop or start. */
	const [isSaving, setIsSavingState] = useState(false)
	const savingRef = useRef(false)
	const [recordingName, setRecordingName] = useState('')
	const liveRecording = useLiveRecording(onBeforeStart, onLiveFinished)
	const liveEnabled = preference.liveTranscription && supportsLiveRecording(preference.modelPath)

	function setInputDeviceAndSave(value: SetStateAction<AudioDevice | null>) {
		const device = typeof value === 'function' ? value(inputDevice) : value
		setSavedInputDeviceId(device?.id ?? '')
		setInputDevice(device)
	}

	function setOutputDeviceAndSave(value: SetStateAction<AudioDevice | null>) {
		const device = typeof value === 'function' ? value(outputDevice) : value
		setSavedOutputDeviceId(device?.id ?? '')
		setOutputDevice(device)
	}

	async function loadAudioDevices() {
		const newDevices = await invoke<AudioDevice[]>('get_audio_devices')
		const outputs = newDevices.filter((device) => !device.isInput)
		setInputDevice(selectInputDevice(newDevices, savedInputDeviceId))
		setOutputDevice(
			savedOutputDeviceId === null
				? (outputs.find((device) => device.isDefault) ?? null)
				: (outputs.find((device) => device.id === savedOutputDeviceId) ?? null),
		)
		setDevices(newDevices)
	}

	useEffect(() => {
		if (preference.homeTab === 'record') loadAudioDevices()
	}, [preference.homeTab])

	const stopWatchdogRef = useRef<number | undefined>(undefined)
	/** Any outside update (the backend's record_stopping/finish/error) also settles the watchdog. */
	const setIsSaving = useCallback((saving: boolean) => {
		window.clearTimeout(stopWatchdogRef.current)
		savingRef.current = saving
		setIsSavingState(saving)
	}, [])
	useEffect(() => () => window.clearTimeout(stopWatchdogRef.current), [])

	async function startRecord() {
		if (liveRecording.isActive() || savingRef.current) return
		const selectedDevices = [inputDevice, outputDevice].filter((device): device is AudioDevice => device !== null)
		if (liveEnabled) {
			await liveRecording.start(selectedDevices, recordingName)
			return
		}
		liveRecording.clear()
		if (outputDevice && !(await ensureSystemAudioPermission())) return
		startKeepAwake(KEEP_AWAKE.record)
		onBeforeStart()
		setIsRecording(true)
		try {
			await invoke('start_record', {
				devices: selectedDevices,
				recordingName: recordingName.trim() || null,
				noAudioStorage: preference.noAudioStorage,
				recordingFormat: preference.recordingFormat,
			})
		} catch (error) {
			stopKeepAwake(KEEP_AWAKE.record)
			setIsRecording(false)
			console.error('startRecord error: ', error)
			setErrorModal?.({ log: String(error), open: true })
		}
	}

	async function stopRecord() {
		if (liveRecording.isActive()) {
			await liveRecording.stop()
			return
		}
		if (savingRef.current) return
		// Set before the event leaves: the button disables on this click, not a round trip later.
		setIsSaving(true)
		// The backend acknowledges a stop within milliseconds (record_stopping). Without one there
		// was no session to stop; release the button instead of leaving it disabled for good.
		stopWatchdogRef.current = window.setTimeout(() => setIsSaving(false), STOP_ACK_TIMEOUT_MS)
		try {
			await emit('stop_record')
		} catch (error) {
			stopKeepAwake(KEEP_AWAKE.record)
			setIsSaving(false)
			setIsRecording(false)
			console.error('stopRecord error: ', error)
			setErrorModal?.({ log: String(error), open: true })
		}
	}

	return {
		devices,
		setDevices,
		inputDevice,
		outputDevice,
		isRecording: isRecording || liveRecording.isRecording,
		isLiveRecording: liveRecording.isRecording,
		isLiveActive: liveRecording.isActive,
		liveEnabled,
		live: liveRecording.state,
		clearLive: liveRecording.clear,
		setIsRecording,
		isSaving,
		setIsSaving,
		recordingName,
		setRecordingName,
		setInputDevice: setInputDeviceAndSave,
		setOutputDevice: setOutputDeviceAndSave,
		startRecord,
		stopRecord,
	}
}
