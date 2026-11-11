import { shortcutLabel, type AppShortcutId } from '~/lib/app-shortcuts'

/** Tooltip body: the action, then its keyboard shortcut in a quiet key cap. */
export default function ShortcutHint({ label, shortcut }: { label: string; shortcut: AppShortcutId }) {
	return (
		<span className="inline-flex items-center gap-2">
			<span>{label}</span>
			<kbd className="rounded bg-primary-foreground/15 px-1.5 py-px font-sans text-[11px] font-medium tracking-wide">{shortcutLabel(shortcut)}</kbd>
		</span>
	)
}
