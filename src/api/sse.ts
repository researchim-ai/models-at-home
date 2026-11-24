/** Decode complete SSE events independently of HTTP chunk and UTF-8 boundaries. */
export async function consumeSSE(
  response: Response,
  onEvent: (event: Record<string, unknown>) => void,
  signal?: AbortSignal,
): Promise<void> {
  const reader = response.body?.getReader()
  if (!reader) throw new Error('Response has no stream')
  const decoder = new TextDecoder()
  let buffer = ''
  let data: string[] = []
  let finished = false
  const abort = () => { void reader.cancel().catch(() => {}) }
  signal?.addEventListener('abort', abort, { once: true })
  const dispatch = () => {
    if (!data.length) return
    const payload = data.join('\n')
    data = []
    if (payload === '[DONE]') { finished = true; return }
    onEvent(JSON.parse(payload))
  }
  const line = (value: string) => {
    if (!value) dispatch()
    else if (value.startsWith('data:')) data.push(value.slice(5).replace(/^ /, ''))
  }
  const drain = (eof: boolean) => {
    let index: number
    while ((index = buffer.search(/[\r\n]/)) >= 0 && !finished) {
      // Wait if CR could be the first half of CRLF in the next packet.
      if (!eof && buffer[index] === '\r' && index === buffer.length - 1) break
      const separator = buffer[index] === '\r' && buffer[index + 1] === '\n' ? 2 : 1
      const value = buffer.slice(0, index)
      buffer = buffer.slice(index + separator)
      line(value)
    }
    if (eof && !finished) {
      if (buffer) line(buffer)
      buffer = ''
      dispatch()
    }
  }
  try {
    if (signal?.aborted) return
    while (!finished && !signal?.aborted) {
      const { done, value } = await reader.read()
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true })
      drain(done)
      if (done) break
    }
  } finally {
    signal?.removeEventListener('abort', abort)
    try { await reader.cancel() } finally { reader.releaseLock() }
  }
}
