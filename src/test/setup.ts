import '@testing-library/jest-dom/vitest'
import { afterEach, vi } from 'vitest'
import { cleanup } from '@testing-library/react'

// jsdom is missing several browser APIs that our components (recharts,
// react-flow, ScrollArea, etc.) rely on. Provide lightweight stubs so that a
// render crash is caused by real bugs, not by missing environment APIs.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
;(globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = ResizeObserverStub

if (!window.matchMedia) {
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }))
}

// jsdom does not implement scrollTo on elements.
Element.prototype.scrollTo = Element.prototype.scrollTo || (() => {})

// jsdom has no WebSocket/EventSource. Provide inert stubs so components that
// open real-time connections on mount don't crash the render.
class WebSocketStub {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3
  readyState = 0
  onopen: ((ev: unknown) => void) | null = null
  onclose: ((ev: unknown) => void) | null = null
  onerror: ((ev: unknown) => void) | null = null
  onmessage: ((ev: unknown) => void) | null = null
  constructor(public url: string) {}
  send() {}
  close() {}
  addEventListener() {}
  removeEventListener() {}
}
;(globalThis as unknown as { WebSocket: unknown }).WebSocket = WebSocketStub

class EventSourceStub {
  onopen: ((ev: unknown) => void) | null = null
  onerror: ((ev: unknown) => void) | null = null
  onmessage: ((ev: unknown) => void) | null = null
  constructor(public url: string) {}
  close() {}
  addEventListener() {}
  removeEventListener() {}
}
;(globalThis as unknown as { EventSource: unknown }).EventSource = EventSourceStub

afterEach(() => {
  cleanup()
})
