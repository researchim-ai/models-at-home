import { consumeSSE } from './sse'

async function resolvePort(): Promise<number> {
  return window.electronAPI ? window.electronAPI.backend.port() : 8000
}

async function getBaseUrl(): Promise<string> {
  if (window.electronAPI) {
    const port = await resolvePort()
    return `http://127.0.0.1:${port}/api`
  }
  return '/api'
}


async function authHeaders(): Promise<Record<string, string>> {
  const token = await window.electronAPI?.backend.token?.()
  return token ? { 'X-Studio-Token': token } : {}
}

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const BASE_URL = await getBaseUrl()
  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...await authHeaders(), ...options?.headers },
  })

  if (!res.ok) {
    let message = await res.text().catch(() => 'Unknown error')
    try {
      const detail = JSON.parse(message).detail
      message = typeof detail === 'string' ? detail : JSON.stringify(detail ?? message)
    } catch { /* plain text error */ }
    throw new ApiError(res.status, message)
  }

  return res.json()
}

export const api = {
  // Vision-language-action policies
  worldResources: (payload: Record<string, unknown>) => request<import('./types').WorldResourceReport>('/world-models/resources/estimate', { method: 'POST', body: JSON.stringify(payload) }),
  benchmarkWorldResources: (payload: Record<string, unknown>) => request<import('./types').WorldResourceReport>('/world-models/resources/benchmark', { method: 'POST', body: JSON.stringify(payload) }),
  startWorldPlay: (payload: Record<string, unknown>) => request<import('./types').LocalWorldFrame & { session_id: string }>('/world-models/play/start', { method: 'POST', body: JSON.stringify(payload) }),
  stepWorldPlay: (payload: Record<string, unknown>) => request<import('./types').LocalWorldFrame>('/world-models/play/step', { method: 'POST', body: JSON.stringify(payload) }),
  resetWorldPlay: (payload: Record<string, unknown>) => request<import('./types').LocalWorldFrame>('/world-models/play/reset', { method: 'POST', body: JSON.stringify(payload) }),
  stopWorldPlay: (id: string) => request<{ success: boolean }>('/world-models/play/stop', { method: 'POST', body: JSON.stringify({ session_id: id }) }),
  worldFoundationModels: () => request<{ models: import('./types').WorldFoundationModel[]; domains: Record<string, number>; checked_at: string }>('/world-models/foundation/models'),
  worldFoundationJobs: () => request<{ runs: import('./types').TrainingRun[] }>('/world-models/foundation/jobs'),
  preflightFoundation: (config: Record<string, unknown>) => request<{ ready: boolean; checks: { name: string; status: string; detail: string }[] }>('/world-models/foundation/preflight', { method: 'POST', body: JSON.stringify(config) }),
  generateFoundation: (config: Record<string, unknown>) => request<{ run_id: string }>('/world-models/foundation/generate', { method: 'POST', body: JSON.stringify(config) }),
  downloadFoundation: (config: Record<string, unknown>) => request<{ run_id: string }>('/world-models/foundation/download', { method: 'POST', body: JSON.stringify(config) }),
  stopFoundation: (id: string) => request<{ success: boolean }>(`/world-models/foundation/stop/${encodeURIComponent(id)}`, { method: 'POST' }),
  foundationResult: (id: string) => request<import('./types').WorldFoundationResult>(`/world-models/foundation/jobs/${encodeURIComponent(id)}/result`),
  foundationVideo: async (id: string) => {
    const response = await fetch(`${await getBaseUrl()}/world-models/foundation/jobs/${encodeURIComponent(id)}/video`, { headers: await authHeaders() });
    if (!response.ok) throw new Error('Не удалось загрузить видео');
    return response.blob();
  },
  worldExamples: () => request<{ dataset_path: string; val_data_path: string }>('/world-models/examples'),
  worldRuns: () => request<{ runs: import('./types').TrainingRun[] }>('/world-models/runs'),
  preflightWorld: (config: Record<string, unknown>) => request<{ ready: boolean; checks: { name: string; status: string; detail: string }[] }>('/world-models/preflight', { method: 'POST', body: JSON.stringify(config) }),
  startWorld: (config: Record<string, unknown>) => request<{ run_id: string }>('/world-models/start', { method: 'POST', body: JSON.stringify(config) }),
  stopWorld: (id: string) => request<{ success: boolean }>(`/world-models/stop/${encodeURIComponent(id)}`, { method: 'POST' }),
  rolloutWorld: (payload: Record<string, unknown>) => request<{ frames: string[]; states: number[][] | null; rewards: number[] | null; done_probabilities: number[] | null; warning: string }>('/world-models/rollout', { method: 'POST', body: JSON.stringify(payload) }),
  vlaExamples: () => request<{ examples: { domain: string; dataset_path: string; val_data_path: string; synthetic: boolean }[] }>('/vla/examples'),
  vlaRuns: () => request<{ runs: import('./types').TrainingRun[] }>('/vla/runs'),
  preflightVLA: (config: Record<string, unknown>) => request<import('./types').VLAPreflightReport>('/vla/preflight', { method: 'POST', body: JSON.stringify(config) }),
  startVLA: (config: Record<string, unknown>) => request<{ run_id: string }>('/vla/start', { method: 'POST', body: JSON.stringify(config) }),
  stopVLA: (id: string) => request<{ success: boolean }>(`/vla/stop/${encodeURIComponent(id)}`, { method: 'POST' }),
  predictVLA: (payload: Record<string, unknown>) => request<import('./types').VLAPrediction>('/vla/predict', { method: 'POST', body: JSON.stringify(payload) }),

  // System
  health: () => request<{ status: string }>('/system/health'),
  gpuInfo: () => request<{ gpus: import('./types').GpuStats[] }>('/system/gpu'),
  configs: () => request<{ accelerate: string[]; deepspeed: string[] }>('/system/configs'),

  // Training
  startTraining: (config: Record<string, unknown>) =>
    request<{ run_id: string }>('/training/start', {
      method: 'POST',
      body: JSON.stringify(config),
    }),

  stopTraining: (runId: string) =>
    request<{ success: boolean }>(`/training/stop/${runId}`, { method: 'POST' }),

  listRuns: () => request<{ runs: import('./types').TrainingRun[] }>('/training/runs'),

  runMetrics: (runId: string) => request<{ metrics: import('./types').MetricsSnapshot | null; history: import('./types').MetricsSnapshot[] }>(`/training/runs/${encodeURIComponent(runId)}/metrics`),

  getRun: (runId: string) => request<import('./types').TrainingRun>(`/training/runs/${runId}`),

  deleteRun: (runId: string) =>
    request<{ success: boolean }>(`/training/runs/${runId}`, { method: 'DELETE' }),

  continueTraining: (runId: string, config?: Record<string, unknown>) =>
    request<{ run_id: string }>(`/training/runs/${runId}/continue`, {
      method: 'POST',
      body: JSON.stringify(config ?? {}),
    }),

  // Models
  listModels: () => request<{ models: import('./types').ModelInfo[] }>('/models/local'),
  listTrainedModels: () => request<{ models: import('./types').ModelInfo[] }>('/models/trained'),
  downloadModel: (data: import('./types').HFDownloadRequest) =>
    request<{ success: boolean; path: string }>('/models/download', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  deleteLocalModel: (name: string) =>
    request<{ success: boolean }>(`/models/local/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  estimateMemory: (config: Record<string, unknown>) =>
    request<{ vram_gb: number; params: number }>('/models/estimate-memory', {
      method: 'POST',
      body: JSON.stringify(config),
    }),

  // Datasets
  listDatasets: () => request<{ datasets: import('./types').DatasetInfo[] }>('/datasets/local'),
  downloadDataset: (data: import('./types').HFDatasetDownloadRequest) =>
    request<{ success: boolean }>('/datasets/download', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  previewDataset: (name: string) =>
    request<{ rows: Record<string, unknown>[] }>(`/datasets/preview/${encodeURIComponent(name)}`),
  deleteDataset: (name: string) =>
    request<{ success: boolean }>(`/datasets/${encodeURIComponent(name)}`, { method: 'DELETE' }),

  // Chat
  loadChatModel: (modelPath: string, backend: string) =>
    request<{ success: boolean }>('/chat/load', {
      method: 'POST',
      body: JSON.stringify({ model_path: modelPath, backend }),
    }),
  unloadChatModel: () => request<{ success: boolean }>('/chat/unload', { method: 'POST' }),
  chatBackends: () => request<{ backends: Array<'transformers' | 'vllm' | 'llama.cpp'> }>('/chat/backends'),

  // Agent
  startAgentServer: (config?: Record<string, unknown>) =>
    request<{ success: boolean }>('/agent/start-server', {
      method: 'POST',
      body: JSON.stringify(config ?? {}),
    }),
  stopAgentServer: () =>
    request<{ success: boolean }>('/agent/stop-server', { method: 'POST' }),
  agentServerStatus: () =>
    request<{ running: boolean; model?: string }>('/agent/server-status'),
  listAgentSessions: () =>
    request<{ sessions: import('./types').AgentSession[] }>('/agent/sessions'),
  createAgentSession: () =>
    request<import('./types').AgentSession>('/agent/sessions', { method: 'POST' }),
  deleteAgentSession: (id: string) =>
    request<{ success: boolean }>(`/agent/sessions/${id}`, { method: 'DELETE' }),
  agentTools: () =>
    request<{ groups: import('./types').AgentToolGroup[]; tools: import('./types').AgentTool[] }>('/agent/tools'),
  agentCapabilities: () =>
    request<{ capabilities: Record<string, unknown> }>('/agent/capabilities'),
  agentRuns: (status: string = 'all') =>
    request<{ status_filter: string; count: number; runs: import('./types').AgentRun[] }>(`/agent/runs?status=${encodeURIComponent(status)}`),
  stopAgentRun: (runId: string) =>
    request<{ run_id: string; stopped: boolean; reason?: string }>(`/agent/runs/${encodeURIComponent(runId)}/stop`, { method: 'POST' }),

  // VLM
  preflightVLM: (config: Record<string, unknown>) => request<import('./types').VLMPreflightReport>('/vlm/preflight', {
    method: 'POST', body: JSON.stringify(config),
  }),
  startVLMTraining: (config: Record<string, unknown>) =>
    request<{ run_id: string }>('/vlm/start', {
      method: 'POST',
      body: JSON.stringify(config),
    }),
  stopVLMTraining: (runId: string) =>
    request<{ success: boolean }>(`/vlm/stop/${runId}`, { method: 'POST' }),
  listVLMRuns: () => request<{ runs: import('./types').TrainingRun[] }>('/vlm/runs'),
  vlmStatus: () => request<{ loaded: boolean; model_path: string | null }>('/vlm/status'),
  loadVLM: (modelPath: string) =>
    request<{ success: boolean }>('/vlm/load', {
      method: 'POST',
      body: JSON.stringify({ model_path: modelPath }),
    }),
  unloadVLM: () => request<{ success: boolean }>('/vlm/unload', { method: 'POST' }),
  generateVLM: (data: { prompt: string; image_base64: string; max_tokens?: number; temperature?: number }) =>
    request<{ text: string }>('/vlm/generate', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  // Study
  studyIndex: (lang: string) =>
    request<{ documents: Array<{ name: string; path: string; source: 'local' | 'github' }> }>(`/study/index?lang=${lang}`),
  studyDocument: (p: string) =>
    request<{ content: string }>(`/study/document?path=${encodeURIComponent(p)}`),

  // Blueprints
  validateBlueprint: (blueprint: Record<string, unknown>) =>
    request<{ blueprint: Record<string, unknown> }>('/blueprints/validate', {
      method: 'POST', body: JSON.stringify(blueprint),
    }),
  listBlueprints: () =>
    request<{ blueprints: Array<{ name: string; path: string; hidden_size: number; vocab_size: number; num_blocks: number; modified_at: string }> }>('/blueprints/list'),
  getBlueprint: (name: string) =>
    request<{ blueprint: Record<string, unknown> }>(`/blueprints/get/${encodeURIComponent(name)}`),
  saveBlueprint: (name: string, blueprint: Record<string, unknown>) =>
    request<{ success: boolean; path: string }>('/blueprints/save', {
      method: 'POST',
      body: JSON.stringify({ name, blueprint }),
    }),
  deleteBlueprint: (name: string) =>
    request<{ success: boolean }>(`/blueprints/delete/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  listBlocks: () =>
    request<{ blocks: Array<{ type: string; description: string }> }>('/blueprints/blocks'),

  // Notebooks
  listNotebooks: () =>
    request<{ notebooks: Array<{ name: string; path: string; size_bytes: number; modified_at: string }> }>('/notebooks/list'),
  notebookTemplates: () =>
    request<{ templates: Array<{ name: string; description: string; file: string }> }>('/notebooks/templates'),
  jupyterStatus: (url: string) =>
    request<{ running: boolean; url: string }>(`/notebooks/jupyter-status?url=${encodeURIComponent(url)}`),
  startJupyter: () =>
    request<{ running: boolean; url: string; token: string; root_dir: string }>('/notebooks/jupyter-start', { method: 'POST' }),
  stopJupyter: () =>
    request<{ success: boolean }>('/notebooks/jupyter-stop', { method: 'POST' }),

  // Training extras
  grpoSamples: (runId: string) =>
    request<{ samples: Array<Record<string, unknown>> }>(`/training/runs/${runId}/grpo-samples`),
  runLogs: (runId: string, lines = 500) =>
    request<{ run_id: string; stdout: string; stderr: string }>(`/training/runs/${runId}/logs?lines=${lines}`),
  estimateMemoryDetailed: (config: Record<string, unknown>, batchSize: number, distributedMode: string, numGpus: number) =>
    request<Record<string, unknown>>('/models/estimate-memory-detailed', {
      method: 'POST',
      body: JSON.stringify({ config, batch_size: batchSize, distributed_mode: distributedMode, num_gpus: numGpus }),
    }),
  getChatTemplate: (modelPath: string) =>
    request<{ has_template: boolean; template: string | null }>(`/models/chat-template?model_path=${encodeURIComponent(modelPath)}`),
}

export function createMetricsWebSocket(
  runId: string,
  onMessage: (data: import('./types').MetricsSnapshot) => void,
  onError?: (err: Event) => void,
): Promise<WebSocket> {
  return resolvePort().then(async (port) => {
    const wsUrl = window.electronAPI
      ? `ws://127.0.0.1:${port}/ws/metrics/${runId}`
      : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/metrics/${runId}`
    const token = await window.electronAPI?.backend.token?.()
    const ws = token ? new WebSocket(wsUrl, [`mah.${token}`]) : new WebSocket(wsUrl)

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        onMessage(data)
      } catch { /* ignore malformed */ }
    }

    ws.onerror = (err) => onError?.(err)
    return ws
  })
}

function startStream(
  path: string,
  body: unknown,
  onEvent: (event: Record<string, unknown>) => void,
  onDone: () => void,
  onError?: (err: Error) => void,
): AbortController {
  const controller = new AbortController()
  void (async () => {
    try {
      const base = await getBaseUrl()
      const response = await fetch(`${base}${path}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json', ...await authHeaders() },
        body: JSON.stringify(body), signal: controller.signal,
      })
      if (!response.ok) throw new Error(await response.text())
      await consumeSSE(response, onEvent, controller.signal)
    } catch (err) {
      if (!controller.signal.aborted) onError?.(err instanceof Error ? err : new Error(String(err)))
    } finally {
      onDone()
    }
  })()
  return controller
}

export function createChatStream(
  request: import('./types').ChatRequest,
  onChunk: (text: string) => void,
  onDone: () => void,
  onError?: (err: Error) => void,
): AbortController {
  return startStream('/chat/generate', request, (event) => {
    if (typeof event.error === 'string') throw new Error(event.error)
    if (typeof event.text === 'string') onChunk(event.text)
  }, onDone, onError)
}

export function createAgentChatStream(
  sessionId: string,
  message: string,
  onChunk: (text: string) => void,
  onToolCall: (tool: string, args: Record<string, unknown>) => void,
  onDone: () => void,
  onError?: (err: Error) => void,
): AbortController {
  return startStream('/agent/chat', {session_id: sessionId, message}, (event) => {
    if (event.type === 'error') throw new Error(String(event.content ?? 'Agent error'))
    if (event.type === 'text') onChunk(String(event.content ?? ''))
    if (event.type === 'tool_call') onToolCall(String(event.tool ?? ''), (event.args ?? {}) as Record<string, unknown>)
  }, onDone, onError)
}
