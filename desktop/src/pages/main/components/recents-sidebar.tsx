import { useVirtualizer } from '@tanstack/react-virtual'
import { invoke } from '@tauri-apps/api/core'
import * as pathApi from '@tauri-apps/api/path'
import * as dialog from '@tauri-apps/plugin-dialog'
import * as fs from '@tauri-apps/plugin-fs'
import { Download, MoreHorizontal, Plus, Search, Settings, Smartphone } from 'lucide-react'
import { useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { m } from '~/paraglide/messages.js'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from '~/components/ui/dropdown-menu'
import { Tooltip, TooltipContent, TooltipTrigger } from '~/components/ui/tooltip'
import ShortcutHint from '~/components/shortcut-hint'
import { cn } from '~/lib/style'
import {
	deleteTranscript,
	listTranscriptsAndFolders,
	readTranscript,
	renameTranscript,
	resolveProjectAudio,
	TRANSCRIPTS_CHANGED_EVENT,
	type TranscriptEntry,
} from '~/lib/transcripts-store'

import { openSettingsSection } from '~/lib/app'
import { TOGGLE_SIDEBAR_EVENT } from '~/components/layout'
import { getTextDirection } from '~/paraglide/runtime.js'
import { UpdaterContext } from '~/providers/updater'
import { useSession } from '../session'
import RetranscribeDialog from './retranscribe-dialog'
import { isTranscribing, RowProgress } from './transcribe-progress'
import type { Job } from '../hooks/use-transcribe-queue'
import { useRecentsFolders } from '../hooks/use-recents-folders'
import { FolderNameInput, FolderRow, MoveToSubmenu, NewFolderButton, TargetFolderChip } from './recents-folders'

/** Resize bounds: never narrower than the rows need, never much wider than the default. */
const DEFAULT_WIDTH = 288
const MIN_WIDTH = 248
const MAX_WIDTH = 380
/**
 * Closing is a separate gesture from resizing: the panel stops following the pointer at MIN_WIDTH,
 * and only collapses once the pointer is dragged this close to the window edge — far enough past
 * the minimum that it can't happen by accident.
 */
const CLOSE_EDGE = 56
/** A recent row is two lines of text plus its padding; the virtualizer measures the real height after mount. */
const ROW_HEIGHT = 52
/** A folder heading is one line of text. */
const FOLDER_ROW_HEIGHT = 36
const WIDTH_STORAGE_KEY = 'vibe_sidebar_width'

function clampWidth(width: number) {
	return Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, width))
}

/** Pointer-drag resize on the sidebar's trailing edge; the width survives a reload. */
function useResizableWidth() {
	const [width, setWidth] = useState(() => {
		try {
			const stored = Number(window.localStorage.getItem(WIDTH_STORAGE_KEY))
			return Number.isFinite(stored) && stored > 0 ? clampWidth(stored) : DEFAULT_WIDTH
		} catch {
			return DEFAULT_WIDTH
		}
	})
	const [dragging, setDragging] = useState(false)
	const start = useRef({ x: 0, width: DEFAULT_WIDTH })
	// Tracks the un-clamped width so we can tell "dragged well past the minimum" from "at the minimum".
	const raw = useRef(width)
	// The collapse fires mid-drag; this keeps a stray move event from toggling it straight back open.
	const collapsed = useRef(false)

	function persist(value: number) {
		try {
			window.localStorage.setItem(WIDTH_STORAGE_KEY, String(value))
		} catch {
			/* private mode — the width still holds for this session */
		}
	}

	function onPointerDown(event: React.PointerEvent<HTMLDivElement>) {
		if (event.button !== 0) return
		event.preventDefault()
		event.currentTarget.setPointerCapture(event.pointerId)
		start.current = { x: event.clientX, width }
		raw.current = width
		collapsed.current = false
		setDragging(true)
	}

	function onPointerMove(event: React.PointerEvent<HTMLDivElement>) {
		if (!dragging || collapsed.current) return
		// The panel is always docked to the window's left edge, so dragging right always widens it.
		raw.current = start.current.width + (event.clientX - start.current.x)
		// The panel stops moving at MIN_WIDTH; only a pointer dragged to the very window edge closes it.
		if (event.clientX < CLOSE_EDGE) {
			collapsed.current = true
			setDragging(false)
			if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
			// The width we reopen at stays untouched.
			window.dispatchEvent(new CustomEvent(TOGGLE_SIDEBAR_EVENT))
			return
		}
		setWidth(clampWidth(raw.current))
	}

	function endDrag(event: React.PointerEvent<HTMLDivElement>) {
		if (!dragging) return
		if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
		setDragging(false)
		persist(clampWidth(raw.current))
	}

	function onDoubleClick() {
		setWidth(DEFAULT_WIDTH)
		raw.current = DEFAULT_WIDTH
		persist(DEFAULT_WIDTH)
	}

	// While dragging, the whole window should show the resize cursor and stop selecting text.
	useEffect(() => {
		if (!dragging) return
		const previousCursor = document.body.style.cursor
		const previousSelect = document.body.style.userSelect
		document.body.style.cursor = 'col-resize'
		document.body.style.userSelect = 'none'
		return () => {
			document.body.style.cursor = previousCursor
			document.body.style.userSelect = previousSelect
		}
	}, [dragging])

	return { width, dragging, handleProps: { onPointerDown, onPointerMove, onPointerUp: endDrag, onPointerCancel: endDrag, onDoubleClick } }
}

/** "just now" / "14m ago" / "3h ago" / "2d ago" / "Aug 19" / "Aug 19, 2024" */
function relativeDate(date: Date) {
	const time = date.getTime()
	if (!time) return ''
	const minutes = Math.floor((Date.now() - time) / 60_000)
	if (minutes < 1) return m.justNow()
	if (minutes < 60) return m.minutesAgo({ minutes: String(minutes) })
	const hours = Math.floor(minutes / 60)
	if (hours < 24) return m.hoursAgo({ hours: String(hours) })
	const days = Math.floor(hours / 24)
	if (days < 7) return m.daysAgo({ days: String(days) })
	const sameYear = date.getFullYear() === new Date().getFullYear()
	return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', ...(sameYear ? {} : { year: 'numeric' }) })
}

interface RowMenuState {
	/** media to re-transcribe: the original file, else the project folder's copy; null when neither */
	sourcePath: string | null
	sourceExists: boolean
}

function RecentRow({
	entry,
	active,
	job,
	folders,
	onOpen,
	onDeleted,
	onRenamed,
	onMove,
	onNewFolder,
}: {
	entry: TranscriptEntry
	active: boolean
	/** The session job transcribing this project right now, if any. */
	job: Job | undefined
	folders: string[]
	onOpen: () => void
	onDeleted: () => void
	onRenamed: (renamed: TranscriptEntry) => void
	onMove: (folder: string | null) => void
	onNewFolder: () => void
}) {
	const { queue } = useSession()
	const [menu, setMenu] = useState<RowMenuState>({ sourcePath: null, sourceExists: false })
	const [renaming, setRenaming] = useState(false)
	const [retranscribing, setRetranscribing] = useState(false)
	const [draftName, setDraftName] = useState(entry.name)
	// The new-folder editor takes focus while the menu closes; handing it back to "…" would blur it.
	const keepFocusAway = useRef(false)

	// The list is built from names only, so the media is located lazily — the first time the row's
	// menu opens — to decide whether "Re-transcribe" can do anything. The original file wins; the
	// project folder's copy keeps re-transcribing possible once the original is gone.
	const loadSource = useCallback(async () => {
		const record = await readTranscript(entry.path)
		if (!record) {
			setMenu({ sourcePath: null, sourceExists: false })
			return
		}
		let sourceExists = false
		try {
			sourceExists = !!record.sourcePath && (await fs.exists(record.sourcePath))
		} catch (error) {
			console.warn('failed to check source file:', error)
		}
		if (sourceExists) {
			setMenu({ sourcePath: record.sourcePath, sourceExists: true })
			return
		}
		const copy = await resolveProjectAudio(entry.path, record)
		setMenu({ sourcePath: copy ?? (record.sourcePath || null), sourceExists: !!copy })
	}, [entry.path])

	async function reveal() {
		try {
			await invoke('open_path', { path: await pathApi.dirname(entry.path) })
		} catch (error) {
			console.warn('failed to reveal transcript:', error)
		}
	}

	async function remove() {
		const confirmed = await dialog.ask(m.deleteTranscriptBody({ name: entry.name }), {
			title: m.deleteTranscript(),
			kind: 'warning',
		})
		if (!confirmed) return
		await deleteTranscript(entry.path)
		onDeleted()
	}

	function retranscribe() {
		if (!menu.sourcePath || !menu.sourceExists) return
		queue.enqueue([{ name: entry.name, path: menu.sourcePath, projectName: entry.name, folder: entry.folder ?? null }])
	}

	async function commitRename() {
		const next = draftName.trim()
		setRenaming(false)
		if (!next || next === entry.name) return
		const renamed = await renameTranscript(entry.path, next)
		if (renamed) onRenamed(renamed)
	}

	if (renaming) {
		return (
			<div className="flex items-center rounded-xl bg-muted px-2 py-1.5">
				<input
					autoFocus
					value={draftName}
					onChange={(event) => setDraftName(event.target.value)}
					onKeyDown={(event) => {
						if (event.key === 'Enter') void commitRename()
						if (event.key === 'Escape') setRenaming(false)
					}}
					onBlur={() => void commitRename()}
					aria-label={m.transcriptName()}
					className="h-7 w-full min-w-0 rounded-lg border border-ring/40 bg-background px-2 text-[13px] text-foreground outline-none"
				/>
			</div>
		)
	}

	return (
		<div className={cn('group relative flex items-center rounded-xl transition-colors duration-150', active ? 'bg-muted' : 'hover:bg-muted/60')}>
			<RetranscribeDialog open={retranscribing} onOpenChange={setRetranscribing} name={entry.name} onConfirm={retranscribe} />
			<button
				type="button"
				onClick={onOpen}
				title={entry.name}
				className="min-w-0 flex-1 cursor-pointer px-3 py-2 text-start">
				<p className="truncate text-[13px] font-medium text-foreground">{entry.name}</p>
				{job ? <RowProgress job={job} /> : <p className="mt-0.5 truncate text-[11px] text-muted-foreground">{relativeDate(entry.createdAt)}</p>}
			</button>

			<DropdownMenu
				onOpenChange={(open) => {
					if (open) void loadSource()
				}}>
				<DropdownMenuTrigger asChild>
					<button
						type="button"
						aria-label={m.transcriptActions()}
						className={cn(
							'me-1.5 flex h-7 w-7 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition-opacity duration-150',
							'opacity-0 group-hover:opacity-100 focus-visible:opacity-100 data-[state=open]:opacity-100 hover:text-foreground',
						)}>
						<MoreHorizontal className="h-4 w-4" />
					</button>
				</DropdownMenuTrigger>
				<DropdownMenuContent
					align="end"
					className="w-48"
					onCloseAutoFocus={(event) => {
						if (!keepFocusAway.current) return
						keepFocusAway.current = false
						event.preventDefault()
					}}>
					<DropdownMenuItem disabled={!!job || !menu.sourceExists} onSelect={() => setRetranscribing(true)}>
						{m.reTranscribe()}
					</DropdownMenuItem>
					<DropdownMenuItem
						onSelect={() => {
							setDraftName(entry.name)
							setRenaming(true)
						}}>
						{m.rename()}
					</DropdownMenuItem>
					{/* Moving a project mid-transcription would pull it out from under the write of its result. */}
					<MoveToSubmenu current={entry.folder} folders={folders} disabled={!!job} onMove={onMove}
						onNewFolder={() => {
							keepFocusAway.current = true
							onNewFolder()
						}}
					/>
					<DropdownMenuItem onSelect={() => void reveal()}>{m.showInFolder()}</DropdownMenuItem>
					<DropdownMenuSeparator />
					<DropdownMenuItem className="text-destructive focus:text-destructive" onSelect={() => void remove()}>
						{m.delete()}
					</DropdownMenuItem>
				</DropdownMenuContent>
			</DropdownMenu>
		</div>
	)
}

/** One job from the running session — it shows up the moment it is queued, not when it is saved. */
function SessionRow({ job, active, onOpen }: { job: Job; active: boolean; onOpen: () => void }) {
	const status = job.status === 'error' ? m.error() : job.status === 'cancelled' ? m.cancel() : ''

	return (
		<div className={cn('group relative flex items-center rounded-xl transition-colors duration-150', active ? 'bg-muted' : 'hover:bg-muted/60')}>
			<button type="button" onClick={onOpen} title={job.name} className="min-w-0 flex-1 cursor-pointer px-3 py-2 text-start">
				<p className="truncate text-[13px] font-medium text-foreground">{job.name}</p>
				{isTranscribing(job) ? <RowProgress job={job} /> : status && <p className="mt-0.5 truncate text-[11px] text-muted-foreground">{status}</p>}
			</button>
		</div>
	)
}

/** One line of the windowed list: a folder heading, a transcript (top level or inside an open folder), or the name editor of a folder being created. */
export type RecentsItem =
	| { kind: 'folder'; folder: string; count: number; open: boolean }
	| { kind: 'entry'; entry: TranscriptEntry; nested: boolean }
	| { kind: 'new-folder' }

function itemKey(item: RecentsItem | undefined, index: number) {
	if (!item) return index
	if (item.kind === 'folder') return `folder:${item.folder}`
	if (item.kind === 'entry') return item.entry.path
	return 'new-folder'
}

/**
 * The recents list, windowed: only the rows near the viewport are mounted, so a folder holding
 * hundreds of projects costs the same to render as one holding ten.
 */
function RecentsList({
	items,
	folders,
	targetFolder,
	projectsPath,
	activePath,
	activeName,
	transcribing,
	onOpen,
	onChanged,
	onRenamed,
	onMove,
	onCreateFolder,
	onCancelNewFolder,
	onStartNewFolder,
	onToggleFolder,
	onNewInFolder,
	onRenameFolder,
	onDeleteFolder,
}: {
	items: RecentsItem[]
	folders: string[]
	targetFolder: string | null
	projectsPath: string | null
	activePath: string | null
	activeName: string | null
	/** Session jobs transcribing a saved project, by the project's record path. */
	transcribing: Map<string, Job>
	onOpen: (entry: TranscriptEntry) => void
	onChanged: () => void
	onRenamed: (entry: TranscriptEntry, renamed: TranscriptEntry) => void
	onMove: (entry: TranscriptEntry, folder: string | null) => void
	onCreateFolder: (name: string) => void
	onCancelNewFolder: () => void
	onStartNewFolder: (moving: TranscriptEntry) => void
	onToggleFolder: (folder: string) => void
	onNewInFolder: (folder: string) => void
	onRenameFolder: (folder: string, name: string) => void
	onDeleteFolder: (folder: string) => void
}) {
	const scrollRef = useRef<HTMLDivElement>(null)
	const virtualizer = useVirtualizer({
		count: items.length,
		getScrollElement: () => scrollRef.current,
		estimateSize: (index) => (items[index]?.kind === 'entry' ? ROW_HEIGHT : FOLDER_ROW_HEIGHT),
		getItemKey: (index) => itemKey(items[index], index),
		// A row grows when its name wraps, and shrinks while it is being renamed.
		measureElement: (element) => element.getBoundingClientRect().height,
		overscan: 8,
	})

	function renderItem(item: RecentsItem) {
		if (item.kind === 'new-folder') return <FolderNameInput onCommit={onCreateFolder} onCancel={onCancelNewFolder} />
		if (item.kind === 'folder') {
			return (
				<FolderRow
					folder={item.folder}
					count={item.count}
					open={item.open}
					target={item.folder === targetFolder}
					projectsPath={projectsPath}
					onToggle={() => onToggleFolder(item.folder)}
					onNewHere={() => onNewInFolder(item.folder)}
					onRename={(name) => onRenameFolder(item.folder, name)}
					onDelete={() => onDeleteFolder(item.folder)}
				/>
			)
		}
		const { entry, nested } = item
		const active = entry.path === activePath
		return (
			<div className={cn(nested && 'ps-5')}>
				<RecentRow
					entry={active && activeName ? { ...entry, name: activeName } : entry}
					active={active}
					job={transcribing.get(entry.path)}
					folders={folders}
					onOpen={() => onOpen(entry)}
					onDeleted={onChanged}
					onRenamed={(renamed) => onRenamed(entry, renamed)}
					onMove={(folder) => onMove(entry, folder)}
					onNewFolder={() => onStartNewFolder(entry)}
				/>
			</div>
		)
	}

	return (
		<div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
			<div className="relative w-full" style={{ height: virtualizer.getTotalSize() }}>
				{virtualizer.getVirtualItems().map((virtual) => {
					const item = items[virtual.index]
					if (!item) return null
					return (
						<div
							key={virtual.key}
							ref={virtualizer.measureElement}
							data-index={virtual.index}
							className="absolute inset-x-0 top-0"
							style={{ transform: `translateY(${virtual.start}px)` }}>
							{renderItem(item)}
						</div>
					)
				})}
			</div>
		</div>
	)
}

export default function RecentsSidebar() {
	const { queue, startNew, preference } = useSession()
	const { updateApp, availableUpdate } = useContext(UpdaterContext)
	const { width, dragging, handleProps } = useResizableWidth()
	const [entries, setEntries] = useState<TranscriptEntry[]>([])
	const [folders, setFolders] = useState<string[]>([])
	const [query, setQuery] = useState('')

	const refresh = useCallback(() => {
		void listTranscriptsAndFolders(preference.projectsPath).then((listing) => {
			setEntries(listing.entries)
			setFolders(listing.folders)
		})
	}, [preference.projectsPath])
	const folderActions = useRecentsFolders(entries, folders, query, refresh)

	useEffect(() => {
		refresh()
		// Saves happen inside the queue; deletes happen in other rows. Both announce themselves.
		window.addEventListener(TRANSCRIPTS_CHANGED_EVENT, refresh)
		return () => window.removeEventListener(TRANSCRIPTS_CHANGED_EVENT, refresh)
	}, [refresh])

	// Jobs of the open session that the store does not know about yet (still queued, running, or
	// finished with saving switched off) — without them a new transcription looks like it went nowhere.
	const liveJobs = useMemo(() => queue.jobs.filter((job) => !job.savedPath && !job.hydrated), [queue.jobs])

	const activePath = queue.selectedJob?.savedPath ?? null
	// Saved projects being transcribed (a recording, say) show their progress in their own row.
	const transcribing = useMemo(
		() => new Map(queue.jobs.filter((job) => job.savedPath && isTranscribing(job)).map((job) => [job.savedPath!, job])),
		[queue.jobs],
	)

	function renamed(entry: TranscriptEntry, next: TranscriptEntry) {
		const job = queue.jobs.find((candidate) => candidate.savedPath === entry.path)
		queue.relocateSaved(entry.path, next.path, next.mediaPath)
		if (job) queue.previewJobName(job.id, next.name)
		refresh()
	}

	async function open(entry: TranscriptEntry) {
		const record = await readTranscript(entry.path)
		if (!record) return
		// The folder entry is the canonical project title. Older records may still contain the
		// original media extension even though the sidebar and project folder do not.
		const projectRecord = record.name === entry.name ? record : { ...record, name: entry.name }
		// Resolved here rather than in the queue so `hydrate` stays synchronous.
		queue.hydrate(projectRecord, entry.path, await resolveProjectAudio(entry.path, record))
	}

	return (
		<aside className="relative flex h-full shrink-0 flex-col border-r border-border" style={{ width, direction: getTextDirection() }}>
			{/* Titlebar strip: hosts the toggle beside the macOS traffic lights (ChatGPT style). */}
			<div data-tauri-drag-region className="h-14 shrink-0" />
			{/* Codex-style: the wordmark gets its own row under the titlebar strip. */}
			<div className="px-4 pt-0.5 pb-2">
				<span className="select-none text-[17px] font-semibold tracking-[-0.03em] text-foreground">{m.appTitle()}</span>
			</div>
			<div className="px-2">
				<Tooltip>
					<TooltipTrigger asChild>
						<button
							type="button"
							onClick={startNew}
							className="flex w-full cursor-pointer items-center gap-2 rounded-xl px-3 py-2 text-[13px] font-medium text-foreground transition-colors duration-150 hover:bg-muted/60">
							<Plus className="h-4 w-4 text-muted-foreground" strokeWidth={1.75} />
							{m.newTranscription()}
						</button>
					</TooltipTrigger>
					<TooltipContent side="right">
						<ShortcutHint label={m.newTranscription()} shortcut="newTranscript" />
					</TooltipContent>
				</Tooltip>
			</div>
			{folderActions.targetFolder && <TargetFolderChip folder={folderActions.targetFolder} onClear={folderActions.clearTarget} />}

			{entries.length > 8 && (
				<div className="relative mt-2 px-3">
					<Search className="pointer-events-none absolute start-6 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
					<input
						autoComplete="off"
						autoCorrect="off"
						autoCapitalize="none"
						spellCheck={false}
						value={query}
						onChange={(event) => setQuery(event.target.value)}
						placeholder={m.searchRecents()}
						className="h-8 w-full rounded-full border border-border bg-transparent ps-7 pe-3 text-[13px] text-foreground outline-none placeholder:text-muted-foreground focus:border-ring/40"
					/>
				</div>
			)}

			{liveJobs.length > 0 && (
				<>
					<p className="px-4 pt-4 pb-2 text-[11px] font-medium tracking-[0.08em] text-muted-foreground uppercase">{m.inProgress()}</p>
					<div className="px-2">
						{liveJobs.map((job) => (
							<SessionRow key={job.id} job={job} active={job.id === queue.selectedId} onOpen={() => queue.selectJob(job.id)} />
						))}
					</div>
				</>
			)}

			<div className="flex items-center justify-between ps-4 pe-3 pt-4 pb-2">
				<p className="text-[11px] font-medium tracking-[0.08em] text-muted-foreground uppercase">{m.recents()}</p>
				<NewFolderButton onClick={() => folderActions.startCreating()} />
			</div>

			{folderActions.items.length === 0 ? (
				<div className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
					<p className="px-3 py-2 text-[13px] text-muted-foreground">
						{entries.length === 0 && folders.length === 0 ? m.noRecentTranscripts() : m.noMatches()}
					</p>
				</div>
			) : (
				<RecentsList
					items={folderActions.items}
					folders={folders}
					targetFolder={folderActions.targetFolder}
					projectsPath={preference.projectsPath}
					activePath={activePath}
					activeName={queue.selectedJob?.name ?? null}
					transcribing={transcribing}
					onOpen={(entry) => void open(entry)}
					onChanged={refresh}
					onRenamed={renamed}
					onMove={folderActions.move}
					onCreateFolder={folderActions.create}
					onCancelNewFolder={folderActions.cancelCreating}
					onStartNewFolder={folderActions.startCreating}
					onToggleFolder={folderActions.toggle}
					onNewInFolder={folderActions.newIn}
					onRenameFolder={folderActions.rename}
					onDeleteFolder={folderActions.remove}
				/>
			)}

			{/* Footer, ChatGPT-style: Settings row with a small update badge when one is waiting. */}
			<div className="mt-auto border-t border-border/60 p-2">
				<div className="flex items-center gap-1">
					<button
						type="button"
						onClick={() => openSettingsSection('')}
						className="flex min-w-0 flex-1 cursor-pointer items-center gap-2.5 rounded-xl px-3 py-2 text-[13px] font-medium text-foreground transition-colors duration-150 hover:bg-muted/60">
						<Settings className="h-4 w-4 text-muted-foreground" strokeWidth={1.75} />
						{m.settings()}
					</button>
					<Tooltip>
						<TooltipTrigger asChild>
							<button
								type="button"
								aria-label={m.phone()}
								onClick={() => openSettingsSection('phone')}
								className="flex h-8 w-8 shrink-0 cursor-pointer items-center justify-center rounded-xl transition-colors duration-150 hover:bg-muted/60">
								<Smartphone className="h-4 w-4 text-muted-foreground" strokeWidth={1.75} />
							</button>
						</TooltipTrigger>
						<TooltipContent side="top">{m.phone()}</TooltipContent>
					</Tooltip>
					{availableUpdate && (
						<Tooltip>
							<TooltipTrigger asChild>
								<button
									type="button"
									aria-label={m.updateVersion()}
									onClick={updateApp}
									className="me-1 flex h-7 w-7 shrink-0 cursor-pointer items-center justify-center rounded-full bg-primary text-primary-foreground transition-opacity duration-150 hover:opacity-90">
									<Download className="h-3.5 w-3.5" />
								</button>
							</TooltipTrigger>
							<TooltipContent side="top">{m.updateVersion()}</TooltipContent>
						</Tooltip>
					)}
				</div>
			</div>

			{/* Drag edge: grab anywhere along the border to resize, double-click to reset. */}
			<div
				role="separator"
				aria-orientation="vertical"
				aria-label={m.resizeSidebar()}
				{...handleProps}
				className={cn(
					'absolute inset-y-0 right-0 z-20 w-1.5 translate-x-1/2 cursor-col-resize touch-none',
					'after:absolute after:inset-y-0 after:left-1/2 after:w-px after:-translate-x-1/2 after:bg-primary/60 after:opacity-0 after:transition-opacity after:duration-150 hover:after:opacity-100',
					dragging && 'after:opacity-100',
				)}
			/>
		</aside>
	)
}
