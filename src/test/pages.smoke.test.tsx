import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import React from 'react'
import { render, waitFor, screen, fireEvent, act } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import App from '@/App'
import '@/i18n'
import { useTrainingStore } from '@/stores/trainingStore'

// ---------------------------------------------------------------------------
// Fake backend. Responses intentionally mimic the *worst realistic* shapes the
// real API can return — including the legacy Streamlit session shape (which
// uses `session_id`, has no `name`, and can carry non-string message content).
// These are precisely the shapes that previously crashed the Agent page to a
// black screen, so the harness now guards against that class of regression.
// ---------------------------------------------------------------------------

const legacySession = {
  session_id: '20260706_073432',
  id: '20260706_073432',
  updated_at: '2026-07-06T07:37:32.873274',
  model_path: '/app/models/Qwen3.5-9B.gguf',
  messages: [
    { role: 'user', content: 'Привет' },
    { role: 'assistant', content: 'Привет!', trace: [{ step: 1, prompt: '...' }] },
    // Deliberately malformed: non-string content must not crash rendering.
    { role: 'assistant', content: { tool: 'list_datasets' } },
    { role: 'assistant', content: null },
  ],
}

const modernSession = {
  id: 'de52c5166a4e',
  name: 'Session de52c5',
  messages: [],
  created_at: '2026-07-07T11:43:50.353035',
}

const agentTools = {
  groups: [
    {
      category: 'discovery',
      title: 'Разведка',
      tools: [
        { name: 'list_datasets', category: 'discovery', description: 'Datasets', arguments: {} },
      ],
    },
  ],
  tools: [{ name: 'list_datasets', category: 'discovery', description: 'Datasets', arguments: {} }],
}

const agentCapabilities = {
  capabilities: {
    text_training: {
      stages: ['pretrain', 'sft'],
      tool: 'start_text_training',
      supports: ['LoRA', 'multi-GPU'],
      key_config_params: ['stage', 'learning_rate'],
    },
  },
}

const agentRuns = {
  status_filter: 'all',
  count: 1,
  runs: [
    {
      run_id: 'run_abc',
      run_dir: '/runs/run_abc',
      pid: 1234,
      is_running: true,
      metrics: { status: 'running', step: 10, total_steps: 100, loss: 0.42 },
      config_summary: { stage: 'sft', training_backend: 'models-at-home' },
    },
  ],
}

function payloadFor(url: string): unknown {
  if (url.includes('/world-models/resources/')) return { status: 'likely_fit', device: 'cpu', hardware: {ram_available_gib: 32, ram_total_gib:64, cpu_threads:4, disk_free_gib:100, cache_disk_free_gib:100}, memory_gib:[1,2], ram_gib:[1,2], reasons:[], warnings:[], hints:[], training_supported:true, benchmark_supported:true }
  if (url.includes('/agent/sessions')) return { sessions: [legacySession, modernSession] }
  if (url.includes('/agent/tools')) return agentTools
  if (url.includes('/agent/capabilities')) return agentCapabilities
  if (url.includes('/agent/runs')) return agentRuns
  if (url.includes('/agent/server-status')) return { running: false }
  // Generous superset for everything else. Pages use `data?.x ?? []`, so this
  // satisfies the remaining endpoints without a per-route map.
  return {
    status: 'ok',
    running: false,
    count: 0,
    runs: [],
    sessions: [],
    groups: [],
    tools: [],
    gpus: [],
    backends: [],
    accelerate: [],
    deepspeed: [],
    notebooks: [],
    files: [],
    models: [],
    datasets: [],
    capabilities: {},
    metrics: {},
  }
}

function makeResponse(body: unknown): Response {
  return {
    ok: true,
    status: 200,
    json: async () => body,
    text: async () => JSON.stringify(body),
    headers: new Headers(),
  } as unknown as Response
}

function installFetchMock() {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      return makeResponse(payloadFor(url))
    }),
  )
}

// React logs render errors to console.error (before any error boundary). We
// record them so a crashing page fails the test with the real stack trace.
function installErrorCapture(sink: Error[]) {
  return vi.spyOn(console, 'error').mockImplementation((...args: unknown[]) => {
    const first = args[0]
    const error = args.find(arg => arg instanceof Error)
    if (error instanceof Error) sink.push(error)
    else if (
      typeof first === 'string' &&
      /Error|Uncaught|not a function|undefined is not|Cannot read/.test(first)
    ) {
      sink.push(new Error(args.map(String).join(' ')))
    }
  })
}

function renderAt(path: string) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  const utils = render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[path]}>
        <App />
      </MemoryRouter>
    </QueryClientProvider>,
  )
  return { queryClient, ...utils }
}

const routes = [
  { path: '/', name: 'TrainingStudio', content: /Readiness/ },
  { path: '/vlm', name: 'VLMStudio', content: /VLM Stage/ },
  { path: '/vla', name: 'VLAStudio', content: /VLA Configuration/ },
  { path: '/builder', name: 'ModelBuilder', content: /Saved Blueprints/ },
  { path: '/world-models', name: 'WorldModelsStudio', content: /World Models Studio/ },
  { path: '/study', name: 'StudyCenter', content: /Select a document from the sidebar/ },
  { path: '/notebooks', name: 'Notebooks', content: /JupyterLab Connection/ },
  { path: '/agent', name: 'AgentStudio', content: /llama-server/ },
  { path: '/settings', name: 'Settings', content: /Configure your studio preferences/ },
]

describe('page smoke tests', () => {
  const capturedErrors: Error[] = []
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    capturedErrors.length = 0
    installFetchMock()
    consoleErrorSpy = installErrorCapture(capturedErrors)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    consoleErrorSpy.mockRestore()
  })

  for (const route of routes) {
    it(`renders ${route.name} (${route.path}) without crashing`, async () => {
      let renderError: unknown = null
      let queryClient: QueryClient | null = null
      try {
        const r = renderAt(route.path)
        queryClient = r.queryClient
        await waitFor(() => expect(screen.getByText('Models at Home')).toBeInTheDocument(), {
          timeout: 5000,
        })
        await screen.findByText(route.content, {}, { timeout: 5000 })
      } catch (err) {
        renderError = err
      }

      expect(renderError, `${route.name} threw during render`).toBeNull()
      expect(
        capturedErrors,
        `${route.name} produced React errors:\n${capturedErrors
          .map((e) => e.stack ?? e.message)
          .join('\n---\n')}`,
      ).toHaveLength(0)

      queryClient?.clear()
    })
  }

  it('hands a saved Builder model to Training Studio without starting a job', async () => {
    const requests: string[] = []
    vi.mocked(fetch).mockImplementation(async (input) => {
      const url = String(input)
      requests.push(url)
      if (url.endsWith('/blueprints/blocks')) return makeResponse({ blocks: [{ type: 'token_embedding', description: '' }] })
      if (url.endsWith('/blueprints/save')) return makeResponse({ success: true, path: '/workspace/blueprints/my_model.json' })
      return makeResponse(payloadFor(url))
    })
    const { queryClient } = renderAt('/builder')
    try {
      fireEvent.click(await screen.findByRole('button', { name: 'token_embedding' }))
      fireEvent.click(screen.getByRole('button', { name: 'Train this model' }))
      await screen.findByText(/Readiness/)
      const config = useTrainingStore.getState().config
      expect(config.model_type).toBe('blueprint')
      expect(config.model_blueprint).toBe('/workspace/blueprints/my_model.json')
      expect((config.model_blueprint_spec?.blocks as unknown[])).toHaveLength(1)
      expect(requests.some(url => url.includes('/training/start'))).toBe(false)
      expect(capturedErrors).toHaveLength(0)
    } finally {
      useTrainingStore.getState().resetConfig()
      queryClient.clear()
    }
  })

  it('keeps invalid JSON edits out of the graph and validates valid edits before applying', async () => {
    const blueprint = { hidden_size: 32, vocab_size: 100, max_position_embeddings: 64, auto_project: true,
      blocks: [{ id: 'embed', type: 'token_embedding', params: {}, inputs: [] }] }
    let validations = 0
    vi.mocked(fetch).mockImplementation(async input => {
      if (String(input).endsWith('/blueprints/validate')) {
        validations++
        return makeResponse({ blueprint })
      }
      return makeResponse(payloadFor(String(input)))
    })
    const { queryClient } = renderAt('/builder')
    try {
      fireEvent.click(await screen.findByRole('button', { name: 'Edit JSON' }))
      const editor = screen.getByLabelText('Blueprint JSON')
      fireEvent.change(editor, { target: { value: '{invalid' } })
      fireEvent.click(screen.getByRole('button', { name: 'Apply JSON' }))
      await screen.findByRole('alert')
      expect(validations).toBe(0)
      expect(screen.getByRole('button', { name: 'Save Blueprint' })).toBeDisabled()
      fireEvent.change(editor, { target: { value: JSON.stringify(blueprint) } })
      fireEvent.click(screen.getByRole('button', { name: 'Apply JSON' }))
      await waitFor(() => expect(screen.queryByLabelText('Blueprint JSON')).not.toBeInTheDocument())
      expect(validations).toBe(1)
      expect(screen.getByText('token_embedding')).toBeInTheDocument()
      expect(capturedErrors).toHaveLength(0)
    } finally { queryClient.clear() }
  })

  it('shows VLM loss and errors and exposes real GRPO settings', async () => {
    vi.mocked(fetch).mockImplementation(async input => {
      const url = String(input)
      if (url.includes('/vlm/runs')) return makeResponse({ runs: [{ run_id: 'vision-run', stage: 'vlm_sft',
        status: 'training', process_running: true, config: {}, current_step: 2, total_steps: 10 }] })
      if (url.includes('/vision-run/metrics')) return makeResponse({ metrics: {
        step: 2, loss: 0.42, learning_rate: 0.01, timestamp: 2, status: 'error', error: 'Image missing: example.png' }, history: [] })
      if (url.includes('/vision-run/logs')) return makeResponse({ stderr: 'synthetic VLM worker log', stdout: '' })
      return makeResponse(payloadFor(url))
    })
    const { queryClient } = renderAt('/vlm')
    try {
      await screen.findByText('Loss 0.42000')
      expect(screen.getByRole('alert')).toHaveTextContent('Image missing')
      const grpo = screen.getByRole('option', { name: /^VLM GRPO$/i })
      expect(grpo).toBeEnabled()
      fireEvent.change(grpo.closest('select')!, { target: { value: 'vlm_grpo' } })
      expect(screen.getByLabelText('VLM GRPO algorithm')).toHaveValue('grpo')
      fireEvent.change(screen.getByLabelText('VLM GRPO algorithm'), { target: { value: 'sdpo' } })
      expect(screen.getByLabelText('VLM GRPO algorithm')).toHaveValue('sdpo')
      fireEvent.change(screen.getByLabelText('VLM backend'), { target: { value: 'unsloth' } })
      expect(screen.getByLabelText('VLM backend')).toHaveValue('unsloth')
      fireEvent.change(screen.getByLabelText('VLM tuning'), { target: { value: 'qlora' } })
      expect(screen.getByLabelText('VLM tuning')).toHaveValue('qlora')
      expect(capturedErrors).toHaveLength(0)
    } finally { queryClient.clear() }
  })

  it('keeps a discovered VLA experiment visible after completion and gates its export', async () => {
    let done = false
    vi.mocked(fetch).mockImplementation(async input => {
      const url = String(input)
      if (url.endsWith('/vla/runs')) return makeResponse({ runs: [{ run_id: 'vla_existing', stage: 'vla_sft',
        status: done ? 'completed' : 'training', process_running: !done, config: { state_dim: 0 }, current_step: done ? 2 : 0,
        total_steps: 2, final_model_path: '/policy' }] })
      if (url.includes('/vla_existing/metrics')) return makeResponse({ metrics: { status: done ? 'completed' : 'training',
        step: done ? 2 : 0, loss: .2, timestamp: 1, final_model_dir: '/policy' }, history: [] })
      return makeResponse(payloadFor(url))
    })
    const { queryClient } = renderAt('/vla')
    try {
      await screen.findByText('Эксперимент vla_existing')
      expect(screen.queryByText('Policy export: /policy')).not.toBeInTheDocument()
      done = true
      await act(async () => { await queryClient.invalidateQueries({ queryKey: ['vla'] }) })
      await screen.findByText('Policy export: /policy')
      expect(screen.getByText('Эксперимент vla_existing')).toBeInTheDocument()
      fireEvent.click(screen.getByRole('tab', { name: 'Action preview' }))
      expect(screen.getByLabelText('State JSON')).toHaveValue('[]')
    } finally { queryClient.clear() }
  })

  it('loads and controls an embedded local world and releases it on unmount', async () => {
    const requests: { url: string; body: Record<string, unknown> }[] = []
    vi.mocked(fetch).mockImplementation(async (input, init) => {
      const url=String(input);const body=JSON.parse(String(init?.body??'{}'));requests.push({url,body})
      if(url.endsWith('/foundation/models')) return makeResponse({models:[{id:'diamond_csgo',name:'DIAMOND CS:GO',provider:'diamond',tasks:['local_play'],local_weights:true}],domains:{}})
      if(url.endsWith('/play/start')) return makeResponse({session_id:'c'.repeat(32),frame:'data:image/png;base64,x',step:0,action_names:['w']})
      if(url.endsWith('/play/step')) return makeResponse({frame:'data:image/png;base64,y',step:1,seconds:.1})
      if(url.endsWith('/play/reset')) return makeResponse({frame:'data:image/png;base64,x',step:0})
      return makeResponse(payloadFor(url))
    })
    const {queryClient,unmount}=renderAt('/world-models')
    try {
      const load=await screen.findByRole('button',{name:'Загрузить модель'})
      expect(screen.queryByRole('option',{name:/Genie/})).toBeNull()
      fireEvent.click(load)
      await waitFor(()=>expect(screen.getByRole('button',{name:'Играть'})).toBeEnabled())
      fireEvent.pointerDown(screen.getByRole('button',{name:'W'}))
      fireEvent.click(screen.getByRole('button',{name:'Один шаг'}))
      await screen.findByText(/Шаг 1/)
      expect(requests.find(r=>r.url.endsWith('/play/step'))?.body).toMatchObject({keys:['w'],session_id:'c'.repeat(32)})
      fireEvent.click(screen.getByRole('button',{name:'Сбросить сцену'}))
      await screen.findByText(/Шаг 0/)
      unmount()
      await waitFor(()=>expect(requests.some(r=>r.url.endsWith('/play/stop'))).toBe(true))
      expect(capturedErrors).toHaveLength(0)
    } finally {queryClient.clear()}
  })

  it('gates world model training on the current preflight configuration', async () => {
    let resolveCheck: (value: Response) => void = () => {}
    const starts: Record<string, unknown>[] = []
    let checks = 0
    vi.mocked(fetch).mockImplementation(async (input, init) => {
      const url = String(input)
      if (url.endsWith('/world-models/preflight')) { checks++; if (checks === 1) return new Promise<Response>(resolve => { resolveCheck = resolve }); return makeResponse({ ready: true, checks: [] }) }
      if (url.endsWith('/world-models/start')) { starts.push(JSON.parse(String(init?.body))); return makeResponse({ run_id: 'world_test' }) }
      return makeResponse(payloadFor(url))
    })
    const { queryClient } = renderAt('/world-models')
    try {
      fireEvent.click(await screen.findByRole('tab', { name: 'Обучение' }))
      const start = await screen.findByRole('button', { name: 'Обучать' })
      expect(start).toBeDisabled()
      fireEvent.click(screen.getByRole('button', { name: 'Проверить данные' }))
      await waitFor(() => expect(checks).toBe(1))
      fireEvent.change(screen.getByLabelText('Переходов в окне'), { target: { value: '6' } })
      await act(async () => resolveCheck(makeResponse({ ready: true, checks: [] })))
      expect(start).toBeDisabled()
      fireEvent.click(screen.getByRole('button', { name: 'Проверить данные' }))
      await act(async () => resolveCheck(makeResponse({ ready: true, checks: [] })))
      await waitFor(() => expect(start).not.toBeDisabled())
      fireEvent.click(start)
      await waitFor(() => expect(starts).toHaveLength(1))
      expect(starts[0].sequence_length).toBe(6)
      expect(capturedErrors).toHaveLength(0)
    } finally { queryClient.clear() }
  })

  it('supports VLA domain presets and rejects stale action preflight results', async () => {
    let resolveFirst: (value: Response) => void = () => {}
    let checks = 0
    let started: Record<string, unknown> | null = null
    const report = { ready: true, checks: [{ name: 'Action contract', status: 'pass', detail: 'validated' }], sample: null }
    vi.mocked(fetch).mockImplementation(async (input, options) => {
      const url = String(input)
      if (url.endsWith('/vla/preflight')) {
        checks++
        if (checks === 1) return await new Promise<Response>(resolve => { resolveFirst = resolve })
        return makeResponse(report)
      }
      if (url.endsWith('/vla/start')) {
        started = JSON.parse(String(options?.body))
        return makeResponse({ run_id: 'vla_test' })
      }
      return makeResponse(payloadFor(url))
    })
    const { queryClient } = renderAt('/vla')
    try {
      const start = await screen.findByRole('button', { name: 'Начать VLA обучение' })
      expect(start).toBeDisabled()
      fireEvent.change(screen.getByLabelText('VLA domain'), { target: { value: 'desktop' } })
      expect(screen.getByLabelText('VLA action type')).toHaveValue('hybrid')
      expect(screen.getByLabelText('Классов в каждой категории (через запятую)')).toHaveValue('4,16')
      fireEvent.change(screen.getByLabelText('Демонстрации JSONL'), { target: { value: '/actions.jsonl' } })
      fireEvent.click(screen.getByRole('button', { name: 'Проверить VLA' }))
      await waitFor(() => expect(checks).toBe(1))
      fireEvent.change(screen.getByLabelText('VLA domain'), { target: { value: 'simulation' } })
      await act(async () => { resolveFirst(makeResponse(report)) })
      expect(start).toBeDisabled()
      fireEvent.click(screen.getByRole('button', { name: 'Проверить VLA' }))
      await screen.findByText('VLA проверка пройдена')
      expect(start).toBeEnabled()
      fireEvent.click(start)
      await waitFor(() => expect(started).toMatchObject({ stage: 'vla_sft', domain: 'simulation', action_type: 'hybrid', discrete_classes: [6], state_dim: 0 }))
      expect(capturedErrors).toHaveLength(0)
    } finally { queryClient.clear() }
  })

  it('requires a current successful VLM preflight and ignores stale responses', async () => {
    const report = { ready: true, checks: [{ name: 'Processor', status: 'pass', detail: 'test processor' }], datasets: [],
      sample: { line: 1, images: [], expanded_tokens: 2, supervised_tokens: 1, omitted_tokens: 0,
        tokens: [{ index: 0, id: 1, text: 'prompt', trainable: false }, { index: 1, id: 2, text: 'answer', trainable: true }] } }
    let resolveFirst: (response: Response) => void = () => {}
    let checks = 0
    let starts = 0
    vi.mocked(fetch).mockImplementation(async input => {
      const url = String(input)
      if (url.endsWith('/vlm/preflight')) {
        checks++
        if (checks === 1) return await new Promise<Response>(resolve => { resolveFirst = resolve })
        return makeResponse(report)
      }
      if (url.endsWith('/vlm/start')) starts++
      return makeResponse(payloadFor(url))
    })
    const { queryClient } = renderAt('/vlm')
    try {
      const start = await screen.findByRole('button', { name: /Start Training/i })
      expect(start).toBeDisabled()
      fireEvent.change(screen.getByPlaceholderText('e.g. Qwen/Qwen3.5-0.8B'), { target: { value: '/model' } })
      fireEvent.change(screen.getByPlaceholderText('datasets/my_vlm_dataset.jsonl'), { target: { value: '/data.jsonl' } })
      fireEvent.click(screen.getByRole('button', { name: 'Проверить эксперимент' }))
      await waitFor(() => expect(checks).toBe(1))
      fireEvent.change(screen.getByLabelText('VLM precision'), { target: { value: 'no' } })
      await act(async () => { resolveFirst(makeResponse(report)) })
      expect(start).toBeDisabled()
      expect(screen.queryByText('Проверка пройдена')).not.toBeInTheDocument()
      fireEvent.click(screen.getByRole('button', { name: 'Проверить эксперимент' }))
      await screen.findByText('Проверка пройдена')
      expect(start).toBeEnabled()
      expect(screen.getByText('answer')).toBeInTheDocument()
      fireEvent.change(screen.getByLabelText('Максимум токенов после processor'), { target: { value: '4096' } })
      await waitFor(() => expect(start).toBeDisabled())
      expect(starts).toBe(0)
      expect(capturedErrors).toHaveLength(0)
    } finally { queryClient.clear() }
  })

  it('AgentStudio survives all tabs and an active legacy session', async () => {
    let queryClient: QueryClient | null = null
    let renderError: unknown = null
    try {
      const r = renderAt('/agent')
      queryClient = r.queryClient
      await waitFor(() => expect(screen.getByText('Models at Home')).toBeInTheDocument(), {
        timeout: 5000,
      })

      // Wait for the tab bar to mount, then visit every tab (by index, so this
      // is language-agnostic) so Runs/Tools/Capabilities panels render.
      await waitFor(() => expect(screen.getAllByRole('tab').length).toBeGreaterThan(1))
      const tabCount = screen.getAllByRole('tab').length
      for (let i = 0; i < tabCount; i++) {
        fireEvent.click(screen.getAllByRole('tab')[i])
        await new Promise((res) => setTimeout(res, 60))
      }
      // Return to the first tab (chat).
      fireEvent.click(screen.getAllByRole('tab')[0])

      // Select the legacy session tab (this loads its messages, exercising the
      // non-string content guard in the chat panel).
      await new Promise((res) => setTimeout(res, 100))
      const sessionButtons = screen.queryAllByRole('button')
      const legacyTab = sessionButtons.find((b) => /20260706/.test(b.title || b.textContent || ''))
      expect(legacyTab).toBeDefined()
      fireEvent.click(legacyTab!)
      await screen.findByText('Привет')
      await new Promise((res) => setTimeout(res, 150))
    } catch (err) {
      renderError = err
    }

    expect(renderError, 'AgentStudio deep test threw').toBeNull()
    expect(
      capturedErrors,
      `AgentStudio produced React errors:\n${capturedErrors
        .map((e) => e.stack ?? e.message)
        .join('\n---\n')}`,
    ).toHaveLength(0)

    queryClient?.clear()
  })
})
