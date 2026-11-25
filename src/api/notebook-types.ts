export interface NotebookOutput {
  output_type: string
  name?: string
  text?: string | string[]
  data?: Record<string, unknown>
  ename?: string
  evalue?: string
  traceback?: string[]
}
export interface NotebookCell {
  id?: string
  cell_type: 'code' | 'markdown' | 'raw'
  source: string | string[]
  metadata: Record<string, unknown>
  outputs?: NotebookOutput[]
  execution_count?: number | null
}
export interface Notebook {
  nbformat: number
  nbformat_minor: number
  metadata: Record<string, unknown> & { studio?: { title?: string; section?: string } }
  cells: NotebookCell[]
}
export interface Lesson {
  path: string; title: string; description: string; topic: string
  minutes: number | null; level: string; lesson: boolean; cells: number
  track?: string; paper?: { title: string; year: number; authors: string; url: string } | null
}
export interface NotebookSession {
  id: string; path: string; state: 'starting' | 'idle' | 'busy' | 'error' | 'closed'; error: string | null
  execution: { id: string; cell_id: string; status: 'running' | 'ok' | 'error'; outputs: NotebookOutput[]; execution_count: number | null } | null
}
export const cellSource = (cell: NotebookCell) => Array.isArray(cell.source) ? cell.source.join('') : cell.source
export const plainText = (value: unknown): string => Array.isArray(value) ? value.join('') : typeof value === 'string' ? value : JSON.stringify(value, null, 2) ?? ''
export const stripAnsi = (value: string) => value.replace(/\u001b\[[0-?]*[ -/]*[@-~]/g, '')
