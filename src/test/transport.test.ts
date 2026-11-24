import { afterEach, describe, expect, it, vi } from 'vitest'
import { consumeSSE } from '@/api/sse'
import { api, createChatStream, createAgentChatStream } from '@/api/client'
import type { ChatRequest } from '@/api/types'

function response(chunks: Uint8Array[]) {
  return new Response(new ReadableStream({
    start(controller) { chunks.forEach(chunk => controller.enqueue(chunk)); controller.close() },
  }))
}

afterEach(() => {
  vi.unstubAllGlobals()
  delete window.electronAPI
})

describe('SSE transport regressions', () => {
  it('preserves UTF-8 and JSON at every possible packet boundary', async () => {
    const bytes = new TextEncoder().encode('data: {"text":"Привет 世界"}\r\n\r\ndata: [DONE]\r\n\r\n')
    for (let split = 1; split < bytes.length; split++) {
      const events: unknown[] = []
      await consumeSSE(response([bytes.slice(0, split), bytes.slice(split)]), event => events.push(event))
      expect(events).toEqual([{ text: 'Привет 世界' }])
    }
  })

  it('supports multiline data, multiple events and a final event at EOF', async () => {
    const bytes = new TextEncoder().encode(': comment\ndata: {"text":\ndata: "one"}\n\ndata:{"text":"two"}')
    const events: unknown[] = []
    await consumeSSE(response([bytes]), event => events.push(event))
    expect(events).toEqual([{ text: 'one' }, { text: 'two' }])
  })

  it('surfaces server chat errors and completes exactly once', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => response([new TextEncoder().encode('data: {"error":"OOM"}\n\ndata: [DONE]\n\n')])))
    const error = vi.fn()
    const done = vi.fn()
    await new Promise<void>(resolve => createChatStream({} as ChatRequest, vi.fn(), () => { done(); resolve() }, error))
    expect(error).toHaveBeenCalledWith(expect.objectContaining({ message: 'OOM' }))
    expect(done).toHaveBeenCalledTimes(1)
  })

  it('surfaces agent errors instead of swallowing them as malformed JSON', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => response([new TextEncoder().encode('data: {"type":"error","content":"failed"}\n\n')])))
    const error = vi.fn()
    await new Promise<void>(resolve => createAgentChatStream('id', 'hello', vi.fn(), vi.fn(), resolve, error))
    expect(error).toHaveBeenCalledWith(expect.objectContaining({ message: 'failed' }))
  })

  it('cancels a pending reader when the user aborts', async () => {
    const cancel = vi.fn()
    const stream = new ReadableStream({ cancel })
    const controller = new AbortController()
    const promise = consumeSSE(new Response(stream), vi.fn(), controller.signal)
    controller.abort()
    await promise
    expect(cancel).toHaveBeenCalledOnce()
  })

  it('queries the current Electron port for every request', async () => {
    let port = 8000
    window.electronAPI = { backend: { port: vi.fn(async () => port) } } as unknown as NonNullable<Window['electronAPI']>
    const fetch = vi.fn(async (_url: RequestInfo | URL) => new Response('{"status":"ok"}'))
    vi.stubGlobal('fetch', fetch)
    await api.health()
    port = 8001
    await api.health()
    expect(fetch.mock.calls.map(call => String(call[0]))).toEqual([
      'http://127.0.0.1:8000/api/system/health', 'http://127.0.0.1:8001/api/system/health',
    ])
  })

  it('includes the session credential on desktop HTTP requests', async () => {
    window.electronAPI = { backend: { port: async () => 8000, token: async () => 'test-secret' } } as unknown as NonNullable<Window['electronAPI']>
    const fetch = vi.fn(async (_url: RequestInfo | URL, _options?: RequestInit) => new Response('{"status":"ok"}'))
    vi.stubGlobal('fetch', fetch)
    await api.health()
    expect(fetch.mock.calls[0][1]?.headers).toMatchObject({ 'X-Studio-Token': 'test-secret' })
  })
})
