import { useEffect, useRef } from 'react'

/**
 * In-window keyboard shortcuts for the main page. One table, one keydown listener (mounted by the
 * main page), and a window event for the actions whose state lives deeper in the tree (search,
 * export, the transcript/summary switch) — the same pattern the sidebar toggle already uses.
 *
 * These are page shortcuts, not OS-wide ones: the global recording/dictation hotkeys are registered
 * natively and never reach the webview, and none of these collide with their defaults.
 */

/** Same convention as the shortcut recorder: ⌘ on macOS, Ctrl everywhere else. */
export const IS_MAC = typeof navigator !== 'undefined' && navigator.platform.toUpperCase().includes('MAC')

export type AppShortcutId = 'find' | 'newTranscript' | 'settings' | 'toggleSidebar' | 'export' | 'toggleSummary'

export interface AppShortcut {
	/** Lower-case `KeyboardEvent.key` for letters, the literal character otherwise. */
	key: string
	shift?: boolean
	/** Fires while typing into an input, textarea or contenteditable. */
	whileTyping?: boolean
}

export const APP_SHORTCUTS: Record<AppShortcutId, AppShortcut> = {
	// Searching is what a user reaches for mid-edit too, so it is the one that works while typing.
	find: { key: 'f', whileTyping: true },
	newTranscript: { key: 'n' },
	settings: { key: ',' },
	toggleSidebar: { key: 'b' },
	export: { key: 'e' },
	toggleSummary: { key: 's', shift: true },
}

/** Dispatched with the shortcut id as `detail` for actions owned by a component further down. */
export const APP_SHORTCUT_EVENT = 'vibe:app-shortcut'

const CODE_KEYS: Record<string, string> = { Comma: ',', Period: '.', Slash: '/' }

/**
 * The key of a press as the table writes it. Letters come from `key` so AZERTY/Dvorak users press
 * what they see; non-Latin layouts (Cyrillic, Hebrew, …) fall back to the physical key.
 */
export function pressedKey(event: Pick<KeyboardEvent, 'key' | 'code'>): string {
	const key = event.key ?? ''
	if (key.length === 1 && /[a-z0-9,./]/i.test(key)) return key.toLowerCase()
	if (/^Key[A-Z]$/.test(event.code ?? '')) return event.code.slice(3).toLowerCase()
	if (/^Digit[0-9]$/.test(event.code ?? '')) return event.code.slice(5)
	return CODE_KEYS[event.code] ?? key.toLowerCase()
}

type ShortcutEvent = Pick<KeyboardEvent, 'key' | 'code' | 'metaKey' | 'ctrlKey' | 'altKey' | 'shiftKey'>

/** Exact modifier match: ⌘⇧F or ⌃⌘F (macOS fullscreen) must not count as ⌘F. */
export function matchesShortcut(event: ShortcutEvent, shortcut: AppShortcut, isMac = IS_MAC): boolean {
	const primary = isMac ? event.metaKey : event.ctrlKey
	const other = isMac ? event.ctrlKey : event.metaKey
	if (!primary || other || event.altKey) return false
	if (Boolean(shortcut.shift) !== event.shiftKey) return false
	return pressedKey(event) === shortcut.key
}

export function findShortcut(event: ShortcutEvent, isMac = IS_MAC): AppShortcutId | null {
	for (const id of Object.keys(APP_SHORTCUTS) as AppShortcutId[]) {
		if (matchesShortcut(event, APP_SHORTCUTS[id], isMac)) return id
	}
	return null
}

/** Typing into a field owns the keyboard: Ctrl+B there is bold/cursor-back, Ctrl+E end-of-line. */
export function isTypingTarget(target: EventTarget | null): boolean {
	if (!target || typeof (target as HTMLElement).tagName !== 'string') return false
	const element = target as HTMLElement
	if (element.isContentEditable) return true
	if (element.tagName === 'TEXTAREA' || element.tagName === 'SELECT') return true
	if (element.tagName !== 'INPUT') return false
	// Checkboxes, radios and buttons take no text, so shortcuts still apply on them.
	const type = (element as HTMLInputElement).type
	return !['checkbox', 'radio', 'button', 'submit', 'reset', 'range', 'color', 'file'].includes(type)
}

/** A modal dialog (or open popover) sits over the page: its own keys win, the page's stay quiet. */
export function overlayOpen(root: ParentNode = document): boolean {
	return Boolean(root.querySelector('[role="dialog"][data-state="open"], [role="alertdialog"][data-state="open"], [aria-modal="true"]'))
}

/** `⌘⇧S` on macOS, `Ctrl+Shift+S` elsewhere — for tooltips. */
export function shortcutLabel(id: AppShortcutId, isMac = IS_MAC): string {
	const shortcut = APP_SHORTCUTS[id]
	const key = shortcut.key.toUpperCase()
	if (isMac) return `⌘${shortcut.shift ? '⇧' : ''}${key}`
	return ['Ctrl', shortcut.shift ? 'Shift' : null, key].filter(Boolean).join('+')
}

/** Resolves a keydown to the shortcut it should run, or null when the page must leave it alone. */
export function resolveShortcut(
	event: ShortcutEvent & Pick<KeyboardEvent, 'target' | 'repeat' | 'defaultPrevented' | 'isComposing'>,
	options: { isMac?: boolean; overlay?: boolean } = {},
): AppShortcutId | null {
	if (event.defaultPrevented || event.isComposing) return null
	const id = findShortcut(event, options.isMac ?? IS_MAC)
	if (!id) return null
	if (options.overlay) return null
	if (!APP_SHORTCUTS[id].whileTyping && isTypingTarget(event.target)) return null
	return id
}

export type AppShortcutHandlers = Partial<Record<AppShortcutId, () => void>>

/**
 * The page's single keydown listener. Ids without a handler here are forwarded as
 * `APP_SHORTCUT_EVENT` for the component that owns them (see `useAppShortcutEvent`).
 */
export function useAppShortcuts(handlers: AppShortcutHandlers) {
	const handlersRef = useRef(handlers)
	handlersRef.current = handlers

	useEffect(() => {
		function onKeyDown(event: KeyboardEvent) {
			const id = resolveShortcut(event, { overlay: overlayOpen() })
			if (!id) return
			// Claimed even when nothing can run right now, so the webview never opens its own find
			// bar (WebView2) or a second window behind the user's back.
			event.preventDefault()
			// A held key would flicker the toggles.
			if (event.repeat) return
			const handler = handlersRef.current[id]
			if (handler) handler()
			else window.dispatchEvent(new CustomEvent<AppShortcutId>(APP_SHORTCUT_EVENT, { detail: id }))
		}
		window.addEventListener('keydown', onKeyDown)
		return () => window.removeEventListener('keydown', onKeyDown)
	}, [])
}

/** Runs `handler` when the page forwards shortcut `id` to the component that owns its state. */
export function useAppShortcutEvent(id: AppShortcutId, handler: () => void) {
	const handlerRef = useRef(handler)
	handlerRef.current = handler

	useEffect(() => {
		function onShortcut(event: Event) {
			if ((event as CustomEvent<AppShortcutId>).detail === id) handlerRef.current()
		}
		window.addEventListener(APP_SHORTCUT_EVENT, onShortcut)
		return () => window.removeEventListener(APP_SHORTCUT_EVENT, onShortcut)
	}, [id])
}
