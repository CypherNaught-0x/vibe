import { invoke } from '@tauri-apps/api/core'
import * as pathApi from '@tauri-apps/api/path'
import { ChevronRight, Folder, FolderOpen, FolderPlus, MoreHorizontal, Plus, X } from 'lucide-react'
import { useRef, useState } from 'react'
import { m } from '~/paraglide/messages.js'
import {
	DropdownMenuItem,
	DropdownMenuSeparator,
	DropdownMenuSub,
	DropdownMenuSubContent,
	DropdownMenuSubTrigger,
	DropdownMenu,
	DropdownMenuContent,
	DropdownMenuTrigger,
} from '~/components/ui/dropdown-menu'
import { Tooltip, TooltipContent, TooltipTrigger } from '~/components/ui/tooltip'
import { cn } from '~/lib/style'
import { transcriptsFolder } from '~/lib/transcripts-store'

const OPEN_FOLDERS_STORAGE_KEY = 'vibe_sidebar_open_folders'

/** Which folders are expanded in the sidebar; survives a reload. */
export function useOpenFolders() {
	const [open, setOpen] = useState<Set<string>>(() => {
		try {
			const stored: unknown = JSON.parse(window.localStorage.getItem(OPEN_FOLDERS_STORAGE_KEY) ?? '[]')
			return new Set(Array.isArray(stored) ? stored.filter((item): item is string => typeof item === 'string') : [])
		} catch {
			return new Set()
		}
	})

	function update(next: Set<string>) {
		setOpen(next)
		try {
			window.localStorage.setItem(OPEN_FOLDERS_STORAGE_KEY, JSON.stringify([...next]))
		} catch {
			/* private mode — the state still holds for this session */
		}
	}

	return {
		open,
		isOpen: (folder: string) => open.has(folder),
		toggle: (folder: string) => {
			const next = new Set(open)
			if (next.has(folder)) next.delete(folder)
			else next.add(folder)
			update(next)
		},
		expand: (folder: string) => {
			if (!open.has(folder)) update(new Set(open).add(folder))
		},
		rename: (from: string, to: string) => {
			if (!open.has(from)) return
			const next = new Set(open)
			next.delete(from)
			next.add(to)
			update(next)
		},
	}
}

/** A one-line name editor used for creating and renaming folders. */
export function FolderNameInput({ initial = '', onCommit, onCancel }: { initial?: string; onCommit: (name: string) => void; onCancel: () => void }) {
	const [draft, setDraft] = useState(initial)
	// Enter commits, then the unmount blurs the input; only the first of the two counts.
	const done = useRef(false)
	function finish(commit: boolean) {
		if (done.current) return
		done.current = true
		const name = draft.trim()
		if (commit && name && name !== initial) onCommit(name)
		else onCancel()
	}
	return (
		<div className="flex items-center gap-2 rounded-xl bg-muted px-3 py-1.5">
			<Folder className="h-4 w-4 shrink-0 text-muted-foreground" strokeWidth={1.75} />
			<input
				autoFocus
				value={draft}
				placeholder={m.folderName()}
				onChange={(event) => setDraft(event.target.value)}
				onKeyDown={(event) => {
					if (event.key === 'Enter') finish(true)
					if (event.key === 'Escape') finish(false)
				}}
				onBlur={() => finish(true)}
				aria-label={m.folderName()}
				className="h-7 w-full min-w-0 rounded-lg border border-ring/40 bg-background px-2 text-[13px] text-foreground outline-none"
			/>
		</div>
	)
}

/** Header row of one folder: expands/collapses it and carries the folder's own actions. */
export function FolderRow({
	folder,
	count,
	open,
	target,
	projectsPath,
	onToggle,
	onNewHere,
	onRename,
	onDelete,
}: {
	folder: string
	count: number
	open: boolean
	/** New transcriptions are currently saved into this folder. */
	target: boolean
	projectsPath: string | null
	onToggle: () => void
	onNewHere: () => void
	onRename: (name: string) => void
	onDelete: () => void
}) {
	const [renaming, setRenaming] = useState(false)

	async function reveal() {
		try {
			await invoke('open_path', { path: await pathApi.join(await transcriptsFolder(projectsPath), folder) })
		} catch (error) {
			console.warn('failed to reveal folder:', error)
		}
	}

	if (renaming) {
		return (
			<FolderNameInput
				initial={folder}
				onCommit={(name) => {
					setRenaming(false)
					onRename(name)
				}}
				onCancel={() => setRenaming(false)}
			/>
		)
	}

	const Icon = open ? FolderOpen : Folder
	return (
		<div className="group relative flex items-center rounded-xl transition-colors duration-150 hover:bg-muted/60">
			<button
				type="button"
				onClick={onToggle}
				aria-expanded={open}
				title={folder}
				className="flex min-w-0 flex-1 cursor-pointer items-center gap-1.5 px-2 py-2 text-start">
				<ChevronRight className={cn('h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform duration-150 rtl:rotate-180', open && 'rotate-90 rtl:rotate-90')} />
				<Icon className={cn('h-4 w-4 shrink-0', target ? 'text-primary' : 'text-muted-foreground')} strokeWidth={1.75} />
				<span className="truncate text-[13px] font-medium text-foreground">{folder}</span>
				<span className="shrink-0 text-[11px] text-muted-foreground tabular-nums">{count}</span>
			</button>

			<Tooltip>
				<TooltipTrigger asChild>
					<button
						type="button"
						onClick={onNewHere}
						aria-label={m.newTranscriptionHere()}
						className="flex h-7 w-7 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground opacity-0 transition-opacity duration-150 group-hover:opacity-100 hover:text-foreground focus-visible:opacity-100">
						<Plus className="h-4 w-4" />
					</button>
				</TooltipTrigger>
				<TooltipContent side="right">{m.newTranscriptionHere()}</TooltipContent>
			</Tooltip>

			<DropdownMenu>
				<DropdownMenuTrigger asChild>
					<button
						type="button"
						aria-label={m.folderActions()}
						className="me-1.5 flex h-7 w-7 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground opacity-0 transition-opacity duration-150 group-hover:opacity-100 hover:text-foreground focus-visible:opacity-100 data-[state=open]:opacity-100">
						<MoreHorizontal className="h-4 w-4" />
					</button>
				</DropdownMenuTrigger>
				<DropdownMenuContent align="end" className="w-52">
					<DropdownMenuItem onSelect={onNewHere}>{m.newTranscriptionHere()}</DropdownMenuItem>
					<DropdownMenuItem onSelect={() => setRenaming(true)}>{m.rename()}</DropdownMenuItem>
					<DropdownMenuItem onSelect={() => void reveal()}>{m.showInFolder()}</DropdownMenuItem>
					<DropdownMenuSeparator />
					<DropdownMenuItem className="text-destructive focus:text-destructive" onSelect={onDelete}>
						{m.deleteFolder()}
					</DropdownMenuItem>
				</DropdownMenuContent>
			</DropdownMenu>
		</div>
	)
}

/** "Move to" submenu of a transcript row: the top level, every other folder, or a new one. */
export function MoveToSubmenu({
	current,
	folders,
	disabled,
	onMove,
	onNewFolder,
}: {
	current: string | undefined
	folders: string[]
	disabled: boolean
	onMove: (folder: string | null) => void
	onNewFolder: () => void
}) {
	return (
		<DropdownMenuSub>
			<DropdownMenuSubTrigger disabled={disabled}>{m.moveTo()}</DropdownMenuSubTrigger>
			<DropdownMenuSubContent className="max-h-72 w-48 overflow-y-auto">
				{current !== undefined && <DropdownMenuItem onSelect={() => onMove(null)}>{m.noFolder()}</DropdownMenuItem>}
				{folders
					.filter((folder) => folder !== current)
					.map((folder) => (
						<DropdownMenuItem key={folder} onSelect={() => onMove(folder)}>
							<Folder className="h-3.5 w-3.5 shrink-0 text-muted-foreground" strokeWidth={1.75} />
							<span className="truncate">{folder}</span>
						</DropdownMenuItem>
					))}
				{(current !== undefined || folders.some((folder) => folder !== current)) && <DropdownMenuSeparator />}
				<DropdownMenuItem onSelect={onNewFolder}>
					<FolderPlus className="h-3.5 w-3.5 shrink-0 text-muted-foreground" strokeWidth={1.75} />
					{m.newFolderEllipsis()}
				</DropdownMenuItem>
			</DropdownMenuSubContent>
		</DropdownMenuSub>
	)
}

/** Shows which folder new transcriptions go to, with a way back to the top level. */
export function TargetFolderChip({ folder, onClear }: { folder: string; onClear: () => void }) {
	return (
		<div className="mx-2 mt-1 flex items-center gap-2 rounded-xl bg-primary/10 py-1 ps-3 pe-1 text-[12px] text-foreground">
			<Folder className="h-3.5 w-3.5 shrink-0 text-primary" strokeWidth={1.75} />
			<span className="min-w-0 flex-1 truncate" title={folder}>
				{m.savingToFolder({ name: folder })}
			</span>
			<Tooltip>
				<TooltipTrigger asChild>
					<button
						type="button"
						onClick={onClear}
						aria-label={m.stopSavingToFolder()}
						className="flex h-6 w-6 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground hover:bg-background/60 hover:text-foreground">
						<X className="h-3.5 w-3.5" />
					</button>
				</TooltipTrigger>
				<TooltipContent side="right">{m.stopSavingToFolder()}</TooltipContent>
			</Tooltip>
		</div>
	)
}

/** The small "new folder" button beside the Recents heading. */
export function NewFolderButton({ onClick }: { onClick: () => void }) {
	return (
		<Tooltip>
			<TooltipTrigger asChild>
				<button
					type="button"
					onClick={onClick}
					aria-label={m.newFolder()}
					className="flex h-6 w-6 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition-colors duration-150 hover:bg-muted/60 hover:text-foreground">
					<FolderPlus className="h-3.5 w-3.5" strokeWidth={1.75} />
				</button>
			</TooltipTrigger>
			<TooltipContent side="right">{m.newFolder()}</TooltipContent>
		</Tooltip>
	)
}
