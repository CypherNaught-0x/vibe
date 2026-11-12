import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'

/** Extra room the full layout needs before it comes back, so a width right at the edge does not flip-flop. */
const HYSTERESIS = 24

/**
 * Whether a toolbar row should drop its labels. The row is measured, not the window: the sidebar
 * beside it can be resized or collapsed. The full layout collapses as soon as its content overflows
 * the row (its width is remembered then), and returns once the row is that wide again plus a margin.
 *
 * Returns a callback ref for the row (a single-line flex container) and the current mode.
 */
export function useCompactToolbar<T extends HTMLElement>() {
	const [element, setElement] = useState<T | null>(null)
	const [compact, setCompact] = useState(false)
	// What the full-label layout needed the last time it was on screen.
	const fullWidth = useRef(0)

	const check = useCallback(() => {
		if (!element) return
		const available = element.clientWidth
		if (available === 0) return // Not laid out (hidden, or mid-mount).
		if (!compact) {
			if (element.scrollWidth > available + 1) {
				fullWidth.current = element.scrollWidth
				setCompact(true)
			}
		} else if (available >= fullWidth.current + HYSTERESIS) {
			setCompact(false)
		}
	}, [element, compact])

	// Labels change with state (a summary starting, a batch arriving): re-check after every render.
	// Only reads layout, and only sets state on a real change, so it settles in one extra pass.
	useLayoutEffect(() => {
		check()
	})

	useEffect(() => {
		if (!element || typeof ResizeObserver === 'undefined') return
		const observer = new ResizeObserver(() => check())
		observer.observe(element)
		for (const child of Array.from(element.children)) observer.observe(child)
		return () => observer.disconnect()
	}, [element, check])

	return { ref: setElement, compact }
}
