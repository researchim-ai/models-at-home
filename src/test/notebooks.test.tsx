import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, expect, it, vi } from 'vitest'
import '@/i18n'
import i18n from '@/i18n'
import { Notebooks } from '@/pages/Notebooks'
import { api } from '@/api/client'

vi.mock('@/api/client', () => ({ api: {
  notebookLibrary: vi.fn(), notebookDocument: vi.fn(), startNotebook: vi.fn(), executeNotebook: vi.fn(),
  notebookSession: vi.fn(), closeNotebook: vi.fn(), saveNotebook: vi.fn(), interruptNotebook: vi.fn(),
} }))
const lesson = { path: 'test.ipynb', title: 'Training basics', description: 'A small experiment', topic: 'basics', minutes: 15, level: 'intro', cells: 3, lesson: true }
const doc = { nbformat: 4, nbformat_minor: 5, metadata: { studio: { title: 'Training basics' } }, cells: [
  { id: 'intro', cell_type: 'markdown', source: '# Learn by running', metadata: {} },
  { id: 'one', cell_type: 'code', source: 'print(42)', metadata: {}, outputs: [], execution_count: null },
  { id: 'two', cell_type: 'code', source: 'print(43)', metadata: {}, outputs: [], execution_count: null },
] }
const session = { id: 'session1', path: 'test.ipynb', state: 'idle', error: null, execution: null }

beforeEach(async () => {
  vi.resetAllMocks()
  await i18n.changeLanguage('en')
  vi.mocked(api.notebookLibrary).mockResolvedValue({ notebooks: [lesson], runtime: { available: true, python: '3.12', executable: '/python', install: 'pip install ipykernel' } })
  vi.mocked(api.notebookDocument).mockResolvedValue({ notebook: structuredClone(doc) as never, revision: 'revision1' })
  vi.mocked(api.startNotebook).mockResolvedValue(session as never)
  vi.mocked(api.closeNotebook).mockResolvedValue({ success: true })
})

async function open() {
  render(<MemoryRouter><Notebooks /></MemoryRouter>)
  fireEvent.click(await screen.findByText('Training basics'))
  await screen.findByText('Learn by running')
}

it('opens, edits and saves notebook cells inside the app', async () => {
  await open()
  expect(document.querySelector('iframe')).toBeNull()
  fireEvent.change(screen.getByLabelText('Cell code 2'), { target: { value: 'print(123)' } })
  vi.mocked(api.saveNotebook).mockResolvedValue({ path: 'test.ipynb', revision: 'revision2' })
  fireEvent.click(screen.getByRole('button', { name: /^Save$/ }))
  await waitFor(() => expect(api.saveNotebook).toHaveBeenCalledWith('test.ipynb', expect.objectContaining({ cells: expect.arrayContaining([expect.objectContaining({ source: 'print(123)' })]) }), 'revision1'))
})

it('runs cells through a persistent local kernel and stops run-all at the first error', async () => {
  await open()
  vi.mocked(api.executeNotebook).mockResolvedValue({ ...session, state: 'busy', execution: { id: 'job', cell_id: 'one', status: 'running', outputs: [], execution_count: 1 } } as never)
  vi.mocked(api.notebookSession).mockResolvedValue({ ...session, execution: { id: 'job', cell_id: 'one', status: 'error', outputs: [{ output_type: 'error', ename: 'ValueError', evalue: 'bad', traceback: ['ValueError: bad'] }], execution_count: 1 } } as never)
  fireEvent.click(screen.getByRole('button', { name: 'Run all' }))
  await screen.findByText('ValueError: bad')
  expect(api.executeNotebook).toHaveBeenCalledTimes(1)
  expect(api.executeNotebook).toHaveBeenCalledWith('session1', 'print(42)', 'one')
})

it('shows missing dependency instructions while keeping lessons readable', async () => {
  vi.mocked(api.notebookLibrary).mockResolvedValue({ notebooks: [lesson], runtime: { available: false, python: '3.12', executable: '/python', install: 'pip install ipykernel' } })
  await open()
  expect(screen.getByText('pip install ipykernel')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Run all' })).toBeDisabled()
  expect(screen.getByLabelText('Cell code 2')).toHaveValue('print(42)')
})


it('saves a copy using the built-in dialog rather than an Electron-unsupported prompt', async () => {
  await open()
  vi.mocked(api.saveNotebook).mockResolvedValue({ path: 'my-copy.ipynb', revision: 'copy-revision' })
  fireEvent.click(screen.getByRole('button', { name: 'Save copy' }))
  fireEvent.change(screen.getByLabelText('Filename'), { target: { value: 'my-copy.ipynb' } })
  fireEvent.submit(screen.getByRole('dialog'))
  await waitFor(() => expect(api.saveNotebook).toHaveBeenCalledWith('my-copy.ipynb', expect.any(Object), null))
  await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
})

it('filters paper studies by collection and searches the original paper title', async () => {
  const study = { ...lesson, path: 'papers/en/attention.ipynb', title: 'Attention mechanism', topic: 'llm', track: 'papers', paper: { title: 'Attention Is All You Need', year: 2017, authors: 'Vaswani et al.', url: 'https://arxiv.org/abs/1706.03762' } }
  vi.mocked(api.notebookLibrary).mockResolvedValue({ notebooks: [lesson, study], runtime: { available: true, python: '3.12', executable: '/python', install: '' } })
  render(<MemoryRouter><Notebooks /></MemoryRouter>)
  await screen.findByText('Attention mechanism')
  fireEvent.change(screen.getByLabelText('Collection'), { target: { value: 'papers' } })
  expect(screen.queryByText('Training basics')).not.toBeInTheDocument()
  expect(screen.getByText('Vaswani et al. · 2017')).toBeInTheDocument()
  fireEvent.change(screen.getByLabelText('Search lessons'), { target: { value: 'All You Need' } })
  expect(screen.getByText('Attention mechanism')).toBeInTheDocument()
  fireEvent.change(screen.getByLabelText('Topic'), { target: { value: 'basics' } })
  expect(screen.queryByText('Attention mechanism')).not.toBeInTheDocument()
})
