import { Component, type ReactNode } from 'react'
import { AlertTriangle, RefreshCw } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface Props {
  children: ReactNode
  /** Changing this value resets the boundary (e.g. on route change). */
  resetKey?: string
}

interface State {
  error: Error | null
}

/**
 * Catches render/runtime errors in the page tree and shows the actual error
 * instead of letting the whole window go blank. This is the last line of
 * defence — bugs should be caught earlier by the smoke tests in src/test.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidUpdate(prev: Props) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null })
    }
  }

  componentDidCatch(error: Error, info: { componentStack?: string }) {
    // Surface to console (and Electron logs) for diagnostics.
    console.error('[ErrorBoundary] page crashed:', error, info?.componentStack)
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children

    // A failed dynamic import (stale chunk after HMR / redeploy) is not a real
    // page bug — the fix is a full reload, not a re-render.
    const isChunkError = /dynamically imported module|Failed to fetch|Loading chunk|import\(/i.test(
      `${error.message} ${error.stack ?? ''}`,
    )

    return (
      <div className="flex h-full w-full items-center justify-center p-8 overflow-auto">
        <div className="max-w-xl w-full space-y-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-destructive/10 text-destructive">
              <AlertTriangle className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-base font-semibold">
                {isChunkError ? 'Не удалось загрузить страницу' : 'Страница упала с ошибкой'}
              </h2>
              <p className="text-xs text-muted-foreground">
                {isChunkError
                  ? 'Модуль устарел (обычно после обновления). Достаточно перезагрузить страницу.'
                  : 'Это не должно происходить — ошибка перехвачена, чтобы не гас весь экран.'}
              </p>
            </div>
          </div>

          <pre className="rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-xs font-mono text-destructive whitespace-pre-wrap overflow-auto max-h-64">
            {error.message}
            {error.stack ? `\n\n${error.stack}` : ''}
          </pre>

          <div className="flex gap-2">
            <Button size="sm" onClick={() => window.location.reload()}>
              <RefreshCw className="h-4 w-4" />
              Перезагрузить страницу
            </Button>
            {!isChunkError && (
              <Button size="sm" variant="outline" onClick={() => this.setState({ error: null })}>
                Повторить рендер
              </Button>
            )}
          </div>
        </div>
      </div>
    )
  }
}
