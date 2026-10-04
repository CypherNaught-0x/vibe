import { openUrl } from '@tauri-apps/plugin-opener'
import { m } from '~/paraglide/messages.js'
import { ReactComponent as LinkIcon } from '~/icons/link.svg'
import * as config from '~/lib/config'
import { Switch } from '~/components/ui/switch'
import { ActionRow, SettingsGroup, SettingsNote, SettingsRow, type SettingsViewModel } from './shared'

export function PrivacySection({ vm }: { vm: SettingsViewModel }) {
	return (
		<div className="space-y-6">
			<SettingsGroup>
				<SettingsRow label={m.analyticsEnabled()} description={!vm.preference.analyticsEnabled ? m.analyticsDisabledWarning() : undefined}>
					<Switch checked={vm.preference.analyticsEnabled} onCheckedChange={vm.preference.setAnalyticsEnabled} />
				</SettingsRow>
				<SettingsRow label={m.saveProjects()} description={m.saveProjectsInfo()}>
					<Switch checked={vm.preference.saveTranscripts} onCheckedChange={vm.preference.setSaveTranscripts} />
				</SettingsRow>
				<SettingsRow label={m.privacyNoAudioStorage()} description={m.privacyNoAudioStorageInfo()} clampDescription={false}>
					<Switch checked={vm.preference.noAudioStorage} onCheckedChange={vm.preference.setNoAudioStorage} aria-label={m.privacyNoAudioStorage()} />
				</SettingsRow>
				<SettingsRow label={m.privacyDeleteTranscriptAfterAi()} description={m.privacyDeleteTranscriptAfterAiInfo()} clampDescription={false}>
					<Switch
						checked={vm.preference.deleteTranscriptAfterAi}
						onCheckedChange={vm.preference.setDeleteTranscriptAfterAi}
						aria-label={m.privacyDeleteTranscriptAfterAi()}
					/>
				</SettingsRow>
				<SettingsNote>{m.privacyRetentionWarning()}</SettingsNote>
				<ActionRow label={m.privacyPolicy()} icon={<LinkIcon className="h-4 w-4" />} onClick={() => openUrl(config.privacyPolicyURL)} />
			</SettingsGroup>
		</div>
	)
}
