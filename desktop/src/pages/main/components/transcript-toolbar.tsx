import { AnimatePresence, motion } from 'framer-motion'
import { ChevronDown, Download, FileText, NotebookPen, Pencil, PilcrowLeft, PilcrowRight, Plus, Search, Settings2, Sparkles, X } from 'lucide-react'
import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { m } from '~/paraglide/messages.js'
import { usePreferenceProvider } from '~/providers/preference'
import { Button } from '~/components/ui/button'
import ResummarizeDialog from '~/components/resummarize-dialog'
import { Popover, PopoverContent, PopoverTrigger } from '~/components/ui/popover'
import { Spinner } from '~/components/ui/spinner'
import { Switch } from '~/components/ui/switch'
import { Tooltip, TooltipContent, TooltipTrigger } from '~/components/ui/tooltip'
import { cn } from '~/lib/style'
import { useAppShortcutEvent } from '~/lib/app-shortcuts'
import ShortcutHint from '~/components/shortcut-hint'
import type { TextFormat } from '~/components/format-select'
import type { Job } from '../hooks/use-transcribe-queue'
import type { TranscriptTab, TranscriptTextSize, TranscriptViewOptions } from '../hooks/use-transcript-view'
import { useTranscriptExport } from '../hooks/use-transcript-export'
import TranscriptExportDialog from './transcript-export-dialog'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from '~/components/ui/dropdown-menu'
import { useSession } from '../session'
import { useCompactToolbar } from '../hooks/use-compact-toolbar'

const TEXT_SIZES: { value: TranscriptTextSize; label: () => string }[] = [
	{ value: 'sm', label: () => m.textSizeSmall() },
	{ value: 'md', label: () => m.textSizeMedium() },
	{ value: 'lg', label: () => m.textSizeLarge() },
]

/** Toolbar buttons never wrap or shrink; compact ones (icon only) trade padding for room. */
function buttonClass(compact: boolean, extra?: string) {
	return cn('shrink-0 rounded-full text-[13px] font-medium whitespace-nowrap', compact ? 'px-2.5' : 'px-3', extra)
}

/** Reading controls for the transcript: type scale, timestamps, speaker labels. */
function ViewOptions({ options }: { options: TranscriptViewOptions }) {
	// Direction belongs to the reading view too, even though it is stored with the other preferences.
	const preference = usePreferenceProvider()

	return (
		<Popover>
			<Tooltip>
				<TooltipTrigger asChild>
					<PopoverTrigger asChild>
						<Button variant="ghost" size="sm" className={buttonClass(true)} aria-label={m.viewOptions()}>
							<Settings2 className="h-3.5 w-3.5" />
						</Button>
					</PopoverTrigger>
				</TooltipTrigger>
				<TooltipContent>{m.viewOptions()}</TooltipContent>
			</Tooltip>
			<PopoverContent align="end" className="w-64 rounded-2xl p-4">
				<p className="mb-2 text-[11px] font-medium tracking-[0.08em] text-muted-foreground uppercase">{m.viewOptions()}</p>
				<div className="space-y-3">
					<div className="space-y-1.5">
						<span className="text-sm text-foreground">{m.textSize()}</span>
						<div className="inline-flex w-full items-center gap-1 rounded-full border border-border bg-muted/50 p-1">
							{TEXT_SIZES.map((size) => (
								<button
									key={size.value}
									type="button"
									onClick={() => options.setTextSize(size.value)}
									className={cn(
										'flex-1 cursor-pointer rounded-full py-1 text-[12px] font-medium transition-colors duration-150',
										options.textSize === size.value
											? 'bg-background text-foreground shadow-sm'
											: 'text-muted-foreground hover:text-foreground',
									)}>
									{size.label()}
								</button>
							))}
						</div>
					</div>
					<div className="flex items-center justify-between gap-4 border-t border-border pt-3">
						<span className="text-sm text-foreground">{m.textDirection()}</span>
						<div className="inline-flex items-center gap-1 rounded-full border border-border bg-muted/50 p-1">
							{(['ltr', 'rtl'] as const).map((value) => (
								<button
									key={value}
									type="button"
									aria-label={value.toUpperCase()}
									aria-pressed={preference.textAreaDirection === value}
									onClick={() => preference.setTextAreaDirection(value)}
									className={cn(
										'flex h-6 w-8 cursor-pointer items-center justify-center rounded-full transition-colors duration-150',
										preference.textAreaDirection === value
											? 'bg-background text-foreground shadow-sm'
											: 'text-muted-foreground hover:text-foreground',
									)}>
									{value === 'ltr' ? <PilcrowRight className="h-3.5 w-3.5" /> : <PilcrowLeft className="h-3.5 w-3.5" />}
								</button>
							))}
						</div>
					</div>

					<div className="flex items-center justify-between gap-4">
						<span className="text-sm text-foreground">{m.showTimestamps()}</span>
						<Switch checked={options.showTimestamps} onCheckedChange={options.setShowTimestamps} />
					</div>
					<div className="flex items-center justify-between gap-4">
						<span className="text-sm text-foreground">{m.showSpeakers()}</span>
						<Switch checked={options.showSpeakers} onCheckedChange={options.setShowSpeakers} />
					</div>
				</div>
			</PopoverContent>
		</Popover>
	)
}

const VIEWS = [
	{ value: 'transcript', label: () => m.segmentsTab(), Icon: FileText },
	{ value: 'summary', label: () => m.summaryTab(), Icon: Sparkles },
] as const

/**
 * Transcript ⇄ Summary, always in view while there is a summary to read (or one arriving). A
 * tablist: arrow keys move between the two, and ⌘⇧S / Ctrl+Shift+S flips it from anywhere.
 */
function ViewSwitch({ tab, setTab, compact }: { tab: TranscriptTab; setTab: (tab: TranscriptTab) => void; compact: boolean }) {
	const refs = useRef<Partial<Record<TranscriptTab, HTMLButtonElement | null>>>({})

	function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
		// Two tabs: every arrow lands on the other one, whichever way the text runs.
		if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
		event.preventDefault()
		const next: TranscriptTab =
			event.key === 'Home' ? 'transcript' : event.key === 'End' ? 'summary' : tab === 'summary' ? 'transcript' : 'summary'
		setTab(next)
		refs.current[next]?.focus()
	}

	return (
		<div
			role="tablist"
			aria-label={m.viewSwitch()}
			onKeyDown={onKeyDown}
			className="me-1 inline-flex shrink-0 items-center gap-0.5 rounded-full border border-border bg-muted/50 p-0.5">
			{VIEWS.map(({ value, label, Icon }) => {
				const selected = tab === value
				return (
					<Tooltip key={value}>
						<TooltipTrigger asChild>
							<button
								ref={(element) => {
									refs.current[value] = element
								}}
								type="button"
								role="tab"
								aria-selected={selected}
								aria-label={compact && !selected ? label() : undefined}
								tabIndex={selected ? 0 : -1}
								onClick={() => setTab(value)}
								className={cn(
									'flex h-7 shrink-0 cursor-pointer items-center gap-1.5 rounded-full text-[13px] font-medium whitespace-nowrap transition-colors duration-150 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none',
									selected ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground',
									// Compact: only the open reading keeps its word.
									compact && !selected ? 'px-2' : 'px-3',
								)}>
								<Icon className="h-3.5 w-3.5" />
								{(!compact || selected) && label()}
							</button>
						</TooltipTrigger>
						<TooltipContent>
							<ShortcutHint label={label()} shortcut="toggleSummary" />
						</TooltipContent>
					</Tooltip>
				)
			})}
		</div>
	)
}

/**
 * What the AI step can do for the open transcript — summarize, improve, customize — behind one
 * button. Switching between transcript and summary is not in here: that is the toolbar's ViewSwitch.
 */
function SummaryMenu({ job, compact }: { job: Job | null; compact: boolean }) {
	const { summaries } = useSession()
	const [open, setOpen] = useState(false)
	const [promptOpen, setPromptOpen] = useState(false)

	const summarizing = job ? Boolean(summaries.pending[job.id]) : false
	const summaryError = job ? summaries.errors[job.id] : undefined
	const hasSummary = Boolean(job?.summary)
	const hasText = (job?.segments.length ?? 0) > 0
	const canSummarize = summaries.enabled && hasText && job?.status !== 'running'
	const actionLabel = summarizing
		? m.summarizeLoading()
		: summaryError
			? m.retrySummary()
			: hasSummary
				? m.regenerateSummary()
				: m.summarizeTranscript()

	// Nothing to offer: summaries are off in settings, or there is no text to summarize yet.
	if (!canSummarize && !summarizing) return null

	function run(prompt?: string) {
		setOpen(false)
		if (job) void summaries.summarize(job, prompt)
	}

	return (
		<>
			<Popover open={open} onOpenChange={setOpen}>
				<Tooltip>
					<TooltipTrigger asChild>
						<PopoverTrigger asChild>
							<Button variant="ghost" size="sm" className={buttonClass(compact)} aria-label={actionLabel}>
								{summarizing ? <Spinner className="h-3.5 w-3.5" /> : <NotebookPen className="h-3.5 w-3.5" />}
								{!compact && actionLabel}
							</Button>
						</PopoverTrigger>
					</TooltipTrigger>
					<TooltipContent>{actionLabel}</TooltipContent>
				</Tooltip>
				<PopoverContent align="end" className="w-64 rounded-2xl p-4">
					<p className="mb-2 text-[11px] font-medium tracking-[0.08em] text-muted-foreground uppercase">{m.aiSection()}</p>

					<div className="space-y-3">
						<div className="space-y-0.5">
							{canSummarize && (
								<button
									type="button"
									disabled={summarizing}
									onClick={() => run()}
									className="flex w-full cursor-pointer items-center gap-2.5 rounded-lg px-2 py-2 text-start text-sm text-foreground transition-colors duration-150 hover:bg-muted/70 disabled:pointer-events-none disabled:opacity-50">
									<Sparkles className="h-3.5 w-3.5 text-muted-foreground" />
									{summaryError ? m.retrySummary() : hasSummary ? m.regenerateSummary() : m.summarizeTranscript()}
								</button>
							)}
							{canSummarize && (
								<button
									type="button"
									disabled={summarizing}
									onClick={() => {
										setOpen(false)
										setPromptOpen(true)
									}}
									className="flex w-full cursor-pointer items-center gap-2.5 rounded-lg px-2 py-2 text-start text-sm text-foreground transition-colors duration-150 hover:bg-muted/70 disabled:pointer-events-none disabled:opacity-50">
									<Pencil className="h-3.5 w-3.5 text-muted-foreground" />
									{m.customizeSummary()}
								</button>
							)}
							{summaryError && <p className="px-2 pt-1 text-xs text-destructive">{m.summaryFailed()}</p>}
						</div>
					</div>
				</PopoverContent>
			</Popover>

			<ResummarizeDialog open={promptOpen} onOpenChange={setPromptOpen} loading={summarizing} onSubmit={(prompt) => run(prompt)} />
		</>
	)
}

export default function TranscriptToolbar({
	job,
	query,
	setQuery,
	options,
	tab,
	setTab,
}: {
	job: Job | null
	query: string
	setQuery: (value: string) => void
	options: TranscriptViewOptions
	tab: TranscriptTab
	setTab: (tab: TranscriptTab) => void
}) {
	const { queue, startNew, summaries } = useSession()
	const preference = usePreferenceProvider()
	const [exportOpen, setExportOpen] = useState(false)
	const autoExport = preference.autoExport
	const exportOptions = preference.exportOptions
	// Each control changes one field of the stored object.
	const updateExport = (patch: Partial<typeof exportOptions>) => preference.setExportOptions({ ...exportOptions, ...patch })
	// A one-off override: the dialog opens on the transcript's own direction every time.
	const [exportDirection, setExportDirection] = useState(preference.textAreaDirection)
	const [searching, setSearching] = useState(false)
	// Narrow row (small window, wide sidebar): buttons drop their words and keep their icons.
	const { ref: actionsRef, compact } = useCompactToolbar<HTMLDivElement>()
	const searchRef = useRef<HTMLInputElement>(null)
	const exporter = useTranscriptExport({
		enabled: exportOpen,
		direction: exportDirection,
		segments: job?.segments ?? [],
		summary: job?.summary,
		file: job ? { name: job.name, path: job.path } : null,
		format: exportOptions.format,
		content: exportOptions.content,
		showTimestamps: exportOptions.showTimestamps,
		showSpeakers: exportOptions.showSpeakers,
		speakerNames: job?.speakerNames,
	})

	useEffect(() => {
		if (exportOpen) setExportDirection(preference.textAreaDirection)
	}, [exportOpen, preference.textAreaDirection])

	useEffect(() => {
		if (!searching) return
		// The row animates in, so claim focus again once that frame has painted.
		const frame = requestAnimationFrame(() => searchRef.current?.focus())
		searchRef.current?.focus()
		return () => cancelAnimationFrame(frame)
	}, [searching])

	function openExport() {
		// The dialog reopens on the choices it was left with; only a saved choice that this
		// transcript cannot honour — a summary it does not have — is corrected.
		if (!job?.summary && exportOptions.content !== 'transcript') updateExport({ content: 'transcript' })
		setExportOpen(true)
	}

	function closeSearch() {
		setSearching(false)
		setQuery('')
	}

	async function saveExport() {
		if (await exporter.save()) setExportOpen(false)
	}

	const hasText = (job?.segments.length ?? 0) > 0
	const hasExportContent = hasText || Boolean(job?.summary)
	// Two readings only when both exist: a summary (or one streaming in) next to a transcript that
	// was kept. A transcript without a summary offers "Summarize" instead, not an empty tab.
	const summarizing = job ? Boolean(summaries.pending[job.id]) : false
	const showViewSwitch = hasText && (Boolean(job?.summary) || summarizing)
	const cancelLabel = queue.isAborting ? m.aborting() : m.cancel()

	useAppShortcutEvent('find', () => {
		if (!hasText) return
		if (!searching) return setSearching(true)
		searchRef.current?.focus()
		searchRef.current?.select()
	})
	useAppShortcutEvent('export', () => {
		if (hasExportContent && !exportOpen) openExport()
	})
	useAppShortcutEvent('toggleSummary', () => {
		if (showViewSwitch) setTab(tab === 'summary' ? 'transcript' : 'summary')
	})

	return (
		<>
			{/* Its empty space moves the window too, like the titlebar strip above (buttons stay clickable). */}
			<div data-tauri-drag-region className="sticky top-0 z-10 flex h-14 shrink-0 items-center gap-1 overflow-hidden border-b border-border bg-background/85 px-4 backdrop-blur-md">
				<AnimatePresence initial={false} mode="wait">
					{searching ? (
						<motion.div
							key="search"
							initial={{ opacity: 0, y: -4 }}
							animate={{ opacity: 1, y: 0 }}
							exit={{ opacity: 0, y: -4 }}
							transition={{ duration: 0.15, ease: 'easeOut' }}
							className="flex w-full items-center gap-2">
							<Search className="h-4 w-4 shrink-0 text-muted-foreground" />
							<input
								ref={searchRef}
								autoFocus
								autoComplete="off"
								autoCorrect="off"
								autoCapitalize="none"
								spellCheck={false}
								value={query}
								onChange={(event) => setQuery(event.target.value)}
								onKeyDown={(event) => event.key === 'Escape' && closeSearch()}
								placeholder={m.searchTranscript()}
								className="h-9 w-full bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
							/>
							<Button variant="ghost" size="iconSm" onClick={closeSearch} className="rounded-full">
								<X className="h-4 w-4" />
							</Button>
						</motion.div>
					) : (
						<motion.div
							key="actions"
							ref={actionsRef}
							data-tauri-drag-region
							initial={{ opacity: 0, y: -4 }}
							animate={{ opacity: 1, y: 0 }}
							exit={{ opacity: 0, y: -4 }}
							transition={{ duration: 0.15, ease: 'easeOut' }}
							className="flex w-full min-w-0 items-center gap-1">
							{showViewSwitch && <ViewSwitch tab={tab} setTab={setTab} compact={compact} />}

							{queue.jobs.length > 1 ? (
								/* A batch: the same word, with everything that writes files behind it. */
								<DropdownMenu>
									<Tooltip>
										<TooltipTrigger asChild>
											<DropdownMenuTrigger asChild>
												<Button variant="ghost" size="sm" className={buttonClass(compact)} aria-label={compact ? m.exportDialogTitle() : undefined}>
													<Download className="h-3.5 w-3.5" />
													{!compact && m.exportDialogTitle()}
													<ChevronDown className="h-3 w-3 opacity-70" />
												</Button>
											</DropdownMenuTrigger>
										</TooltipTrigger>
										<TooltipContent>{m.exportDialogTitle()}</TooltipContent>
									</Tooltip>
									<DropdownMenuContent align="start" className="w-60">
										<DropdownMenuItem onClick={openExport} disabled={!hasExportContent}>
											{m.exportThisTranscript()}
										</DropdownMenuItem>
										<DropdownMenuItem onClick={() => void queue.exportFinished()} disabled={queue.unexportedCount === 0}>
											{m.exportAllFinished({ count: String(queue.unexportedCount) })}
										</DropdownMenuItem>
										<DropdownMenuSeparator />
										<DropdownMenuItem
											onClick={() => window.dispatchEvent(new CustomEvent('vibe:open-settings', { detail: { scrollTo: 'auto-export' } }))}
											className="flex items-center justify-between gap-3">
											<span className="shrink-0 whitespace-nowrap">{m.autoExport()}</span>
											<span className="truncate text-[11px] text-muted-foreground">
												{autoExport.enabled
													? `${m.autoExportOn()} · ${autoExport.formats.map((format) => (format === 'normal' ? 'TXT' : format.toUpperCase())).join(', ')}`
													: m.autoExportOff()}
											</span>
										</DropdownMenuItem>
									</DropdownMenuContent>
								</DropdownMenu>
							) : (
								<Tooltip>
									<TooltipTrigger asChild>
										<Button
											variant="ghost"
											size="sm"
											onClick={openExport}
											disabled={!hasExportContent}
											aria-label={compact ? m.exportDialogTitle() : undefined}
											className={buttonClass(compact)}>
											<Download className="h-3.5 w-3.5" />
											{!compact && m.exportDialogTitle()}
										</Button>
									</TooltipTrigger>
									<TooltipContent>
										<ShortcutHint label={m.exportDialogTitle()} shortcut="export" />
									</TooltipContent>
								</Tooltip>
							)}

							<Tooltip>
								<TooltipTrigger asChild>
									<Button
										variant="ghost"
										size="sm"
										onClick={() => setSearching(true)}
										disabled={!hasText}
										aria-label={m.searchTranscript()}
										className={buttonClass(true)}>
										<Search className="h-3.5 w-3.5" />
									</Button>
								</TooltipTrigger>
								<TooltipContent>
									<ShortcutHint label={m.searchTranscript()} shortcut="find" />
								</TooltipContent>
							</Tooltip>

							<SummaryMenu job={job} compact={compact} />

							<ViewOptions options={options} />

							<div className="ms-auto flex shrink-0 items-center gap-1">
								{queue.running && (
									<Tooltip>
										<TooltipTrigger asChild>
											<Button
												variant="ghost"
												size="sm"
												onClick={queue.cancelAll}
												disabled={queue.isAborting}
												aria-label={compact ? cancelLabel : undefined}
												className={buttonClass(compact, 'text-muted-foreground hover:text-destructive')}>
												{/* The word is the button at full width; compact needs a glyph. */}
												{compact && (queue.isAborting ? <Spinner className="h-3.5 w-3.5" /> : <X className="h-3.5 w-3.5" />)}
												{!compact && cancelLabel}
											</Button>
										</TooltipTrigger>
										{compact && <TooltipContent>{cancelLabel}</TooltipContent>}
									</Tooltip>
								)}
								<Tooltip>
									<TooltipTrigger asChild>
										<Button
											variant="ghost"
											size="sm"
											onClick={startNew}
											aria-label={compact ? m.newSession() : undefined}
											className={buttonClass(compact)}>
											<Plus className="h-3.5 w-3.5" />
											{!compact && m.newSession()}
										</Button>
									</TooltipTrigger>
									<TooltipContent>
										<ShortcutHint label={m.newTranscription()} shortcut="newTranscript" />
									</TooltipContent>
								</Tooltip>
							</div>
						</motion.div>
					)}
				</AnimatePresence>
			</div>
			<TranscriptExportDialog
				open={exportOpen}
				onOpenChange={setExportOpen}
				format={exportOptions.format}
				onFormatChange={(format: TextFormat) => updateExport({ format })}
				content={exportOptions.content}
				onContentChange={(content) => updateExport({ content })}
				hasSummary={Boolean(job?.summary)}
				showTimestamps={exportOptions.showTimestamps}
				onShowTimestampsChange={(showTimestamps) => updateExport({ showTimestamps })}
				showSpeakers={exportOptions.showSpeakers}
				onShowSpeakersChange={(showSpeakers) => updateExport({ showSpeakers })}
				preview={exporter.preview}
				renderedPreview={exporter.renderedPreview}
				direction={exportDirection}
				onDirectionChange={setExportDirection}
				theme={exportOptions.theme}
				onThemeChange={(theme) => updateExport({ theme })}
				onCopy={exporter.copy}
				onSave={saveExport}
			/>
		</>
	)
}
