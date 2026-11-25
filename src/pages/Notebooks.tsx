import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import { BookOpen, Play, Square, RotateCcw, Save, Download, Upload, Plus, Trash2, Code, FileText, Loader2, Search } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { api } from '@/api/client'
import { toast } from 'sonner'
import { cellSource, plainText, stripAnsi, type Lesson, type Notebook, type NotebookCell, type NotebookOutput, type NotebookSession } from '@/api/notebook-types'

function Output({ output }: { output: NotebookOutput }) {
  if (output.output_type === 'error') return <pre className="nb-output nb-error">{stripAnsi((output.traceback ?? [`${output.ename}: ${output.evalue}`]).join('\n'))}</pre>
  if (output.output_type === 'stream') return <pre className={`nb-output ${output.name === 'stderr' ? 'text-amber-600' : ''}`}>{plainText(output.text)}</pre>
  const data = output.data ?? {}
  const mime = data['image/png'] ? 'image/png' : data['image/jpeg'] ? 'image/jpeg' : null
  if (mime) return <img alt="Notebook output" className="max-w-full rounded-md bg-white" src={`data:${mime};base64,${plainText(data[mime]).replace(/\s/g, '')}`} />
  return <pre className="nb-output">{plainText(data['text/plain'] ?? data['application/json'] ?? '[Unsupported rich output]')}</pre>
}

export function Notebooks() {
  const { i18n } = useTranslation()
  const ru = i18n.language.startsWith('ru')
  const l = (a: string, b: string) => ru ? a : b
  const [lessons, setLessons] = useState<Lesson[]>([])
  const [runtime, setRuntime] = useState<{ available: boolean; python: string; executable: string; install: string } | null>(null)
  const [query, setQuery] = useState('')
  const [topic, setTopic] = useState('all')
  const [track, setTrack] = useState('all')
  const [path, setPath] = useState('')
  const [notebook, setNotebook] = useState<Notebook | null>(null)
  const [revision, setRevision] = useState<string | null>(null)
  const [dirty, setDirty] = useState(false)
  const [session, setSession] = useState<NotebookSession | null>(null)
  const [busy, setBusy] = useState(false)
  const [saving, setSaving] = useState(false)
  const [copyName, setCopyName] = useState('')
  const [showSaveAs, setShowSaveAs] = useState(false)
  const [error, setError] = useState('')
  const sessionRef = useRef<NotebookSession | null>(null)
  const stopRef = useRef(false)
  const mounted = useRef(true)
  const importRef = useRef<HTMLInputElement>(null)
  const updateSession = (value: NotebookSession | null) => { sessionRef.current = value; if (mounted.current) setSession(value) }
  const fail = (e: unknown) => { const message = e instanceof Error ? e.message : String(e); setError(message); toast.error(message) }
  const load = () => api.notebookLibrary(i18n.language).then(data => { setLessons(data.notebooks); setRuntime(data.runtime) }).catch(fail)
  useEffect(() => { void load() }, [i18n.language])
  useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false; stopRef.current = true; if (sessionRef.current) void api.closeNotebook(sessionRef.current.id).catch(() => {}) }
  }, [])
  useEffect(() => {
    const listener = (event: BeforeUnloadEvent) => { if (dirty) { event.preventDefault(); event.returnValue = '' } }
    window.addEventListener('beforeunload', listener)
    return () => window.removeEventListener('beforeunload', listener)
  }, [dirty])
  useEffect(() => {
    const listener = (event: MouseEvent) => {
      const anchor = (event.target as Element).closest?.('a[href]') as HTMLAnchorElement | null
      if (dirty && anchor && anchor.target !== '_blank' && anchor.href !== window.location.href && !window.confirm(ru ? 'Есть несохранённые изменения. Перейти без сохранения?' : 'Leave without saving notebook changes?')) {
        event.preventDefault(); event.stopPropagation()
      }
    }
    document.addEventListener('click', listener, true)
    return () => document.removeEventListener('click', listener, true)
  }, [dirty, ru])
  // Confirm before replacing the current in-memory document.
  const canLeave = () => !dirty || window.confirm(l('Есть несохранённые изменения. Закрыть их?', 'Discard unsaved changes?'))
  const normalize = (doc: Notebook): Notebook => {
    const seen = new Set<string>()
    return { ...doc, metadata: doc.metadata ?? {}, cells: doc.cells.map(c => { const id = c.id && !seen.has(c.id) ? c.id : crypto.randomUUID(); seen.add(id); return { ...c, id } }) }
  }
  const open = async (entry: Lesson) => {
    if (busy || !canLeave()) return
    setBusy(true); setError('')
    try {
      const data = await api.notebookDocument(entry.path)
      if (sessionRef.current) await api.closeNotebook(sessionRef.current.id)
      updateSession(null); setNotebook(normalize(data.notebook)); setPath(entry.path); setRevision(data.revision); setDirty(false)
    } catch (e) { fail(e) } finally { setBusy(false) }
  }
  const changeCell = (id: string, updates: Partial<NotebookCell>) => {
    setNotebook(doc => doc ? { ...doc, cells: doc.cells.map(c => c.id === id ? { ...c, ...updates } : c) } : doc)
    setDirty(true)
  }
  const saveTo = async (target: string) => {
    if (!notebook || !target.trim()) return
    target = target.trim()
    if (!target.endsWith('.ipynb')) target += '.ipynb'
    setSaving(true)
    try {
      const result = await api.saveNotebook(target, notebook, target === path ? revision : null)
      setPath(result.path); setRevision(result.revision); setDirty(false); setShowSaveAs(false); void load()
      toast.success(l('Ноутбук сохранён', 'Notebook saved'))
    } catch (e) { fail(e) } finally { setSaving(false) }
  }
  const save = async (copy = false) => {
    if (copy || !path) { setCopyName(`my-${path.split('/').pop() || 'notebook.ipynb'}`); setShowSaveAs(true); return }
    await saveTo(path)
  }
  const ensureKernel = async () => {
    if (!path || !revision) throw new Error(l('Сначала сохраните ноутбук', 'Save the notebook first'))
    if (sessionRef.current?.state === 'idle') return sessionRef.current
    if (sessionRef.current) await api.closeNotebook(sessionRef.current.id)
    const created = await api.startNotebook(path)
    if (!mounted.current) { await api.closeNotebook(created.id); throw new Error('Notebook closed') }
    updateSession(created)
    return created
  }
  const execute = async (cell: NotebookCell, kernel: NotebookSession) => {
    let current = await api.executeNotebook(kernel.id, cellSource(cell), cell.id!)
    updateSession(current)
    while (current.execution?.status === 'running') {
      await new Promise(resolve => setTimeout(resolve, 250))
      if (!mounted.current) return false
      current = await api.notebookSession(kernel.id)
      updateSession(current)
      if (current.execution) changeCell(cell.id!, { outputs: current.execution.outputs, execution_count: current.execution.execution_count })
      if (current.state === 'error' || current.state === 'closed') throw new Error(current.error || 'Kernel stopped')
    }
    if (current.execution) changeCell(cell.id!, { outputs: current.execution.outputs, execution_count: current.execution.execution_count })
    return current.execution?.status === 'ok'
  }
  const run = async (cells: NotebookCell[]) => {
    if (busy) return
    setBusy(true); stopRef.current = false; setError('')
    try {
      const kernel = await ensureKernel()
      for (const cell of cells) {
        if (stopRef.current || !mounted.current) break
        if (cell.cell_type === 'code' && !(await execute(cell, kernel))) break
      }
    } catch (e) { if (mounted.current) fail(e) } finally { if (mounted.current) setBusy(false) }
  }
  const interrupt = async () => { stopRef.current = true; if (sessionRef.current) { try { updateSession(await api.interruptNotebook(sessionRef.current.id)) } catch (e) { fail(e) } } }
  const reset = async () => {
    if (busy) return
    setBusy(true)
    try { if (sessionRef.current) await api.closeNotebook(sessionRef.current.id); updateSession(null) } catch (e) { fail(e) } finally { setBusy(false) }
  }
  const exportFile = () => {
    if (!notebook) return
    const url = URL.createObjectURL(new Blob([JSON.stringify(notebook, null, 2)], { type: 'application/x-ipynb+json' }))
    const link = document.createElement('a'); link.href = url; link.download = path.split('/').pop() || 'notebook.ipynb'; link.click(); URL.revokeObjectURL(url)
  }
  const importFile = async (file: File) => {
    if (busy || !canLeave()) return
    try {
      if (file.size > 10_000_000) throw new Error('10 MB notebook limit')
      const doc = JSON.parse(await file.text()) as Notebook
      if (doc.nbformat !== 4 || !Array.isArray(doc.cells) || doc.cells.length > 500 || doc.cells.some(c => !['code', 'markdown', 'raw'].includes(c.cell_type) || !(typeof c.source === 'string' || Array.isArray(c.source) && c.source.every(s => typeof s === 'string')))) throw new Error(l('Неверный формат .ipynb', 'Invalid .ipynb format'))
      await api.validateNotebook(doc)
      if (sessionRef.current) await api.closeNotebook(sessionRef.current.id)
      updateSession(null); setNotebook(normalize(doc)); setPath(''); setRevision(null); setDirty(true); setError('')
    } catch (e) { fail(e) }
  }
  const addCell = (type: 'code' | 'markdown') => {
    setNotebook(doc => doc ? { ...doc, cells: [...doc.cells, { id: crypto.randomUUID(), cell_type: type, source: '', metadata: {}, ...(type === 'code' ? { outputs: [], execution_count: null } : {}) }] } : doc)
    setDirty(true)
  }
  const filtered = lessons.filter(item => (topic === 'all' || item.topic === topic) && (track === 'all' || (item.track ?? (item.lesson ? 'foundations' : 'custom')) === track) && `${item.title} ${item.description} ${item.paper?.title ?? ''} ${item.paper?.authors ?? ''}`.toLowerCase().includes(query.toLowerCase()))
  const title = notebook?.metadata.studio?.title || path.split('/').pop() || l('Свой ноутбук', 'My notebook')
  return <div className="notebook-studio">
    {showSaveAs && <div className="fixed inset-0 z-50 bg-black/40 flex items-center justify-center p-6"><form role="dialog" aria-modal="true" aria-label={l('Сохранить копию', 'Save notebook copy')} className="bg-card border border-border rounded-xl p-6 w-full max-w-md shadow-xl" onKeyDown={e => { if (e.key === 'Escape' && !saving) { e.preventDefault(); setShowSaveAs(false) } if (e.key === 'Tab') { const elements = Array.from(e.currentTarget.querySelectorAll<HTMLInputElement | HTMLButtonElement>('input,button:not(:disabled)')); const first = elements[0], last = elements[elements.length - 1]; if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last?.focus() } else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first?.focus() } } }} onSubmit={e => { e.preventDefault(); void saveTo(copyName) }}><h2 className="font-semibold mb-3">{l('Сохранить копию', 'Save notebook copy')}</h2><label className="text-sm">{l('Имя файла', 'Filename')}<input autoFocus className="mt-2 w-full bg-background border border-border rounded-md p-2" value={copyName} onChange={e => setCopyName(e.target.value)} /></label><p className="text-xs text-muted-foreground mt-2">notebooks/{copyName}</p><div className="flex gap-2 mt-5"><Button type="submit" disabled={saving || !copyName.trim()}>{l('Сохранить', 'Save')}</Button><Button type="button" variant="outline" disabled={saving} onClick={() => setShowSaveAs(false)}>{l('Отмена', 'Cancel')}</Button></div></form></div>}
    <aside className="nb-library">
      <div className="mb-5"><div className="flex items-center gap-2 text-primary"><BookOpen size={18} /><span className="text-xs uppercase tracking-widest font-semibold">Learning Lab</span></div><h2 className="text-xl font-semibold mt-2">{l('Учебные ноутбуки', 'Learning notebooks')}</h2><p className="text-sm text-muted-foreground mt-2">{l('Читайте, меняйте код и проверяйте идеи прямо в студии.', 'Read, edit code, and test ideas inside Studio.')}</p></div>
      <label className="flex items-center gap-2 rounded-md border border-border px-3 py-2"><Search size={15} /><input aria-label={l('Поиск уроков', 'Search lessons')} className="bg-transparent min-w-0 w-full text-sm outline-none" value={query} onChange={e => setQuery(e.target.value)} placeholder={l('Найти тему', 'Find a topic')} /></label>
      <select aria-label={l('Подборка', 'Collection')} className="mt-2 w-full rounded-md border border-border bg-background p-2 text-sm" value={track} onChange={e => setTrack(e.target.value)}><option value="all">{l('Все подборки', 'All collections')}</option><option value="foundations">{l('Основы · короткие уроки', 'Foundations · short lessons')}</option><option value="papers">{l('Разборы статей · с экспериментами', 'Paper studies · with experiments')}</option><option value="custom">{l('Другие ноутбуки', 'Other notebooks')}</option></select>
      <select aria-label={l('Тема', 'Topic')} className="mt-2 mb-4 w-full rounded-md border border-border bg-background p-2 text-sm" value={topic} onChange={e => setTopic(e.target.value)}><option value="all">{l('Все темы', 'All topics')}</option>{[...new Set(lessons.map(e => e.topic))].map(t => <option key={t} value={t}>{t.toUpperCase()}</option>)}</select>
      <div className="space-y-2">{filtered.map(item => <button disabled={busy} key={item.path} onClick={() => void open(item)} className={`nb-lesson ${path === item.path ? 'nb-lesson-selected' : ''}`}><div className="flex justify-between gap-2 text-[11px] uppercase tracking-wide text-muted-foreground"><span>{item.topic}</span>{item.minutes && <span>{item.minutes} {l('мин', 'min')}</span>}</div><div className="font-medium text-sm mt-1">{item.title}</div>{item.paper && <div className="text-[11px] text-primary mt-1">{item.paper.authors} · {item.paper.year}</div>}<div className="text-xs text-muted-foreground mt-1 leading-relaxed">{item.description}</div></button>)}</div>
      {!filtered.length && <p className="text-sm py-4 text-muted-foreground">{l('Ноутбуки не найдены.', 'No notebooks found.')}</p>}
      <Button variant="outline" className="w-full mt-4" disabled={busy} onClick={() => importRef.current?.click()}><Upload size={14} />{l('Импорт .ipynb', 'Import .ipynb')}</Button><input ref={importRef} type="file" accept=".ipynb" className="hidden" onChange={e => { if (e.target.files?.[0]) void importFile(e.target.files[0]); e.target.value = '' }} />
      <div className="mt-5 text-xs text-muted-foreground leading-relaxed">Python {runtime?.python ?? '…'} · {l('локальное ядро', 'local kernel')}<p className="mt-2">{l('Код выполняется с доступом к вашим файлам и ресурсам. Запускайте только понятные вам ячейки.', 'Code can access your files and hardware. Run cells you understand.')}</p></div>
    </aside>
    <section className="nb-editor min-w-0">
      <header className="nb-toolbar"><div className="flex-1 min-w-0"><h1 className="font-semibold truncate">{notebook ? title : l('Учитесь на работающих примерах', 'Learn with working examples')}{dirty && <span className="text-primary ml-2" aria-label="Unsaved">●</span>}</h1><p className="text-xs text-muted-foreground mt-1 truncate">{path || l('Выберите урок слева или импортируйте свой .ipynb', 'Choose a lesson or import an .ipynb file')}</p></div>{notebook && <div className="flex flex-wrap gap-2"><Button size="sm" disabled={busy || !runtime?.available || !revision} onClick={() => void run(notebook.cells)}><Play size={14} />{busy ? l('Выполняется…', 'Running…') : l('Запустить всё', 'Run all')}</Button><Button size="sm" variant="outline" disabled={!busy || !session} onClick={() => void interrupt()}><Square size={14} />{l('Стоп', 'Stop')}</Button><Button size="sm" variant="outline" disabled={busy || saving} onClick={() => void save()}><Save size={14} />{l('Сохранить', 'Save')}</Button><Button size="sm" variant="ghost" disabled={busy || saving} onClick={() => void save(true)}>{l('Копия', 'Save copy')}</Button><Button title={l('Экспорт .ipynb', 'Export .ipynb')} aria-label={l('Экспорт .ipynb', 'Export .ipynb')} size="icon" variant="ghost" disabled={busy} onClick={exportFile}><Download size={16} /></Button></div>}</header>
      {runtime && !runtime.available && <div className="m-6 rounded-lg border border-amber-500/40 p-4 text-sm">{l('Для запуска установите зависимости в Python-окружении бэкенда:', 'Install dependencies in the backend Python environment:')}<pre className="mt-2 whitespace-pre-wrap">{runtime.install}</pre><p className="text-xs mt-2">{runtime.executable}</p><Button className="mt-2" size="sm" variant="outline" onClick={() => void load()}>{l('Проверить снова', 'Check again')}</Button></div>}
      {error && <div role="alert" className="m-6 rounded-md border border-destructive/40 bg-destructive/5 p-3 text-sm break-words">{error}</div>}
      {notebook ? <div className="nb-document"><div className="flex flex-wrap items-center justify-between gap-3 mb-6 text-xs text-muted-foreground"><span className="flex items-center gap-2"><span className={`h-2 w-2 rounded-full ${session?.state === 'busy' ? 'bg-primary animate-pulse' : session?.state === 'idle' ? 'bg-green-600' : 'bg-muted-foreground/50'}`} />{session ? `Python · ${session.state}` : l('Ядро запустится при выполнении', 'Kernel starts on first run')}</span><div className="flex gap-3">{notebook.metadata.studio?.section && <Link className="text-primary underline" to={notebook.metadata.studio.section}>{l('Открыть раздел студии', 'Open Studio workspace')}</Link>}<button disabled={busy || !session} onClick={() => void reset()} className="flex items-center gap-1 disabled:opacity-40"><RotateCcw size={12} />{l('Сбросить ядро', 'Reset kernel')}</button></div></div>
      {notebook.cells.map((cell, index) => <article key={cell.id} className={`nb-cell ${cell.cell_type === 'code' ? 'nb-code-cell' : ''}`}><div className="nb-cell-gutter">{cell.cell_type === 'code' ? <><button aria-label={`${l('Выполнить ячейку', 'Run cell')} ${index + 1}`} disabled={busy || !runtime?.available || !revision} onClick={() => void run([cell])} className="text-primary disabled:opacity-30">{busy && session?.execution?.cell_id === cell.id ? <Loader2 size={17} className="animate-spin" /> : <Play size={17} />}</button><span className="font-mono text-[11px]">[{cell.execution_count ?? ' '}]</span></> : <FileText size={15} />}</div><div className="min-w-0 flex-1">{cell.cell_type === 'code' ? <textarea aria-label={`${l('Код ячейки', 'Cell code')} ${index + 1}`} spellCheck={false} disabled={busy} className="nb-code" rows={Math.min(25, Math.max(3, cellSource(cell).split('\n').length))} value={cellSource(cell)} onChange={e => changeCell(cell.id!, { source: e.target.value })} onKeyDown={e => { if (e.key === 'Enter' && e.shiftKey) { e.preventDefault(); void run([cell]) } }} /> : <><div className="nb-markdown"><ReactMarkdown components={{ a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer" onClick={event => { if (href && /^https?:/.test(href) && window.electronAPI) { event.preventDefault(); void window.electronAPI.shell.openExternal(href) } }}>{children}</a> }}>{cellSource(cell)}</ReactMarkdown></div><details className="mt-2 text-xs text-muted-foreground"><summary className="cursor-pointer">{l('Редактировать текст', 'Edit text')}</summary><textarea aria-label={`${l('Текст ячейки', 'Cell text')} ${index + 1}`} disabled={busy} className="nb-code mt-2" rows={6} value={cellSource(cell)} onChange={e => changeCell(cell.id!, { source: e.target.value })} /></details></>}{cell.outputs && cell.outputs.length > 0 && <div className="nb-outputs">{cell.outputs.map((output, idx) => <Output key={idx} output={output} />)}</div>}</div><button disabled={busy} aria-label={`${l('Удалить ячейку', 'Delete cell')} ${index + 1}`} className="nb-delete" onClick={() => { setNotebook(doc => doc ? { ...doc, cells: doc.cells.filter(c => c.id !== cell.id) } : doc); setDirty(true) }}><Trash2 size={13} /></button></article>)}
      <div className="flex gap-2 mt-6"><Button variant="outline" size="sm" disabled={busy} onClick={() => addCell('code')}><Plus size={13} /><Code size={13} />Python</Button><Button variant="outline" size="sm" disabled={busy} onClick={() => addCell('markdown')}><Plus size={13} />Markdown</Button><Button variant="ghost" size="sm" disabled={busy} onClick={() => { setNotebook(doc => doc ? { ...doc, cells: doc.cells.map(c => c.cell_type === 'code' ? { ...c, outputs: [], execution_count: null } : c) } : doc); setDirty(true) }}>{l('Очистить вывод', 'Clear outputs')}</Button></div><p className="text-xs text-muted-foreground mt-5">Shift + Enter · {l('выполнить ячейку. «Запустить всё» останавливается при первой ошибке. Сброс ядра удаляет переменные, но сохраняет код.', 'run a cell. Run all stops at the first error. Resetting the kernel clears variables and preserves code.')}</p></div> : <div className="nb-empty"><BookOpen size={40} className="text-primary mb-5" /><h2 className="text-2xl font-semibold">{l('От объяснения к эксперименту', 'From explanation to experiment')}</h2><p className="text-muted-foreground max-w-lg mt-3 leading-relaxed">{l('Каждый урок объясняет задачу, запускает небольшой пример и предлагает изменить его. Начните с основ обучения, затем переходите к архитектурам, мультимодальности и моделям действий.', 'Each lesson explains a task, runs a small example, and suggests an experiment. Start with training fundamentals, then explore architectures, multimodality, and action models.')}</p><div className="grid sm:grid-cols-3 gap-4 mt-8 max-w-2xl text-sm">{[l('Небольшие CPU-примеры', 'Small CPU examples'), l('Код и вывод рядом', 'Code beside its output'), l('Обычные файлы .ipynb', 'Standard .ipynb files')].map(t => <div className="rounded-lg border border-border p-4" key={t}>{t}</div>)}</div></div>}
    </section>
  </div>
}
