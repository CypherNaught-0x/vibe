// @vitest-environment jsdom

import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import {
	APP_SHORTCUT_EVENT,
	findShortcut,
	isTypingTarget,
	matchesShortcut,
	overlayOpen,
	pressedKey,
	resolveShortcut,
	shortcutLabel,
	useAppShortcutEvent,
	useAppShortcuts,
	type AppShortcutId,
} from './app-shortcuts'

afterEach(() => {
	cleanup()
	document.body.innerHTML = ''
})

const press = (key: string, init: Partial<KeyboardEventInit> & { code?: string } = {}) => ({
	key,
	code: init.code ?? `Key${key.toUpperCase()}`,
	metaKey: Boolean(init.metaKey),
	ctrlKey: Boolean(init.ctrlKey),
	altKey: Boolean(init.altKey),
	shiftKey: Boolean(init.shiftKey),
})

describe('key matching', () => {
	it('uses ⌘ on macOS and Ctrl elsewhere', () => {
		expect(findShortcut(press('f', { metaKey: true }), true)).toBe('find')
		expect(findShortcut(press('f', { ctrlKey: true }), true)).toBeNull()
		expect(findShortcut(press('f', { ctrlKey: true }), false)).toBe('find')
		expect(findShortcut(press('f', { metaKey: true }), false)).toBeNull()
	})

	it('requires the exact modifiers', () => {
		// ⌃⌘F is macOS fullscreen, ⌘⇧F and ⌘⌥F are something else.
		expect(findShortcut(press('f', { metaKey: true, ctrlKey: true }), true)).toBeNull()
		expect(findShortcut(press('F', { metaKey: true, shiftKey: true }), true)).toBeNull()
		expect(findShortcut(press('f', { metaKey: true, altKey: true }), true)).toBeNull()
		expect(findShortcut(press('f'), true)).toBeNull()
	})

	it('matches shifted shortcuts only with Shift', () => {
		expect(findShortcut(press('S', { ctrlKey: true, shiftKey: true }), false)).toBe('toggleSummary')
		expect(findShortcut(press('s', { ctrlKey: true }), false)).toBeNull()
	})

	it('maps every shortcut', () => {
		const cases: [string, Partial<KeyboardEventInit> & { code?: string }, AppShortcutId][] = [
			['n', {}, 'newTranscript'],
			[',', { code: 'Comma' }, 'settings'],
			['b', {}, 'toggleSidebar'],
			['e', {}, 'export'],
		]
		for (const [key, init, id] of cases) expect(findShortcut(press(key, { ...init, metaKey: true }), true)).toBe(id)
	})

	it('falls back to the physical key on non-Latin layouts', () => {
		expect(pressedKey({ key: 'а', code: 'KeyF' })).toBe('f')
		expect(matchesShortcut({ ...press('б', { ctrlKey: true }), code: 'Comma' }, { key: ',' }, false)).toBe(true)
		// Latin layouts keep what is printed on the key (AZERTY: the F key is still F).
		expect(pressedKey({ key: 'q', code: 'KeyA' })).toBe('q')
	})

	it('formats labels per platform', () => {
		expect(shortcutLabel('find', true)).toBe('⌘F')
		expect(shortcutLabel('find', false)).toBe('Ctrl+F')
		expect(shortcutLabel('toggleSummary', true)).toBe('⌘⇧S')
		expect(shortcutLabel('toggleSummary', false)).toBe('Ctrl+Shift+S')
		expect(shortcutLabel('settings', false)).toBe('Ctrl+,')
	})
})

describe('input suppression', () => {
	const event = (target: EventTarget | null, key: string, extra: Partial<KeyboardEventInit> = {}) => ({
		...press(key, { ctrlKey: true, ...extra }),
		target,
		repeat: false,
		defaultPrevented: false,
		isComposing: false,
	})

	it('recognises text fields', () => {
		const editable = document.createElement('div')
		editable.contentEditable = 'true'
		// jsdom does not implement isContentEditable.
		Object.defineProperty(editable, 'isContentEditable', { value: true })
		const checkbox = Object.assign(document.createElement('input'), { type: 'checkbox' })
		expect(isTypingTarget(document.createElement('input'))).toBe(true)
		expect(isTypingTarget(document.createElement('textarea'))).toBe(true)
		expect(isTypingTarget(editable)).toBe(true)
		expect(isTypingTarget(checkbox)).toBe(false)
		expect(isTypingTarget(document.createElement('button'))).toBe(false)
		expect(isTypingTarget(null)).toBe(false)
	})

	it('leaves typing alone except for find', () => {
		const textarea = document.createElement('textarea')
		expect(resolveShortcut(event(textarea, 'n'), { isMac: false })).toBeNull()
		expect(resolveShortcut(event(textarea, 'b'), { isMac: false })).toBeNull()
		expect(resolveShortcut(event(textarea, 'f'), { isMac: false })).toBe('find')
		expect(resolveShortcut(event(document.body, 'n'), { isMac: false })).toBe('newTranscript')
	})

	it('stays quiet under an overlay, an IME composition or a handled event', () => {
		expect(resolveShortcut(event(document.body, 'f'), { isMac: false, overlay: true })).toBeNull()
		expect(resolveShortcut({ ...event(document.body, 'f'), isComposing: true }, { isMac: false })).toBeNull()
		expect(resolveShortcut({ ...event(document.body, 'f'), defaultPrevented: true }, { isMac: false })).toBeNull()
	})

	it('detects open dialogs and popovers', () => {
		expect(overlayOpen()).toBe(false)
		document.body.innerHTML = '<div role="dialog" data-state="closed"></div>'
		expect(overlayOpen()).toBe(false)
		document.body.innerHTML = '<div role="dialog" data-state="open"></div>'
		expect(overlayOpen()).toBe(true)
		document.body.innerHTML = '<div role="dialog" aria-modal="true"></div>'
		expect(overlayOpen()).toBe(true)
	})
})

describe('hooks', () => {
	// jsdom reports a non-Mac platform, so Ctrl is the primary modifier here.
	const keydown = (key: string, init: KeyboardEventInit = {}, target: EventTarget = window) => {
		const event = new KeyboardEvent('keydown', { key, code: `Key${key.toUpperCase()}`, ctrlKey: true, bubbles: true, cancelable: true, ...init })
		act(() => {
			target.dispatchEvent(event)
		})
		return event
	}

	it('runs page handlers, forwards the rest, and claims the key', () => {
		const newTranscript = vi.fn()
		const find = vi.fn()
		renderHook(() => {
			useAppShortcuts({ newTranscript })
			useAppShortcutEvent('find', find)
		})
		const forwarded = vi.fn()
		window.addEventListener(APP_SHORTCUT_EVENT, forwarded)

		expect(keydown('n').defaultPrevented).toBe(true)
		expect(newTranscript).toHaveBeenCalledTimes(1)
		expect(forwarded).not.toHaveBeenCalled()

		expect(keydown('f').defaultPrevented).toBe(true)
		expect(find).toHaveBeenCalledTimes(1)
		window.removeEventListener(APP_SHORTCUT_EVENT, forwarded)
	})

	it('ignores repeats and typing, but find still works from a field', () => {
		const newTranscript = vi.fn()
		const find = vi.fn()
		renderHook(() => {
			useAppShortcuts({ newTranscript })
			useAppShortcutEvent('find', find)
		})
		keydown('n', { repeat: true })
		expect(newTranscript).not.toHaveBeenCalled()

		const input = document.createElement('input')
		document.body.appendChild(input)
		expect(keydown('n', {}, input).defaultPrevented).toBe(false)
		expect(newTranscript).not.toHaveBeenCalled()
		keydown('f', {}, input)
		expect(find).toHaveBeenCalledTimes(1)
	})

	it('does nothing without the modifier', () => {
		const newTranscript = vi.fn()
		renderHook(() => useAppShortcuts({ newTranscript }))
		expect(keydown('n', { ctrlKey: false }).defaultPrevented).toBe(false)
		expect(newTranscript).not.toHaveBeenCalled()
	})
})
