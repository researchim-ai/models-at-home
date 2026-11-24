import { useEffect, useMemo, useState } from 'react'
import { useDockerStore } from '@/stores/dockerStore'
import {
  Loader2, Server, AlertCircle, RefreshCw, FileText, Package, Container,
  Heart, PlayCircle, Cog, Cpu,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useQueryClient } from '@tanstack/react-query'
import type { BootPhase } from '@/types/electron'

interface BackendBootProps {
  children: React.ReactNode
}

function phaseLabel(phase: BootPhase): { title: string; detail: string; icon: React.ReactNode } {
  switch (phase.phase) {
    case 'starting':
      return { title: 'Запуск...', detail: 'Инициализация', icon: <Cog className="h-6 w-6" /> }
    case 'checking-docker':
      return { title: 'Поиск Docker', detail: 'Проверяю, доступен ли Docker daemon', icon: <Container className="h-6 w-6" /> }
    case 'docker-unavailable':
      return { title: 'Docker недоступен', detail: `${phase.detail ?? 'Переключаюсь на нативный Python'}`, icon: <Container className="h-6 w-6" /> }
    case 'docker-no-image':
      return { title: 'Docker образ отсутствует', detail: 'Сейчас соберу его — это может занять 10-20 минут при первом запуске', icon: <Package className="h-6 w-6" /> }
    case 'building-image':
      return { title: 'Сборка Docker образа', detail: phase.line ?? 'Собираю контейнер для ML', icon: <Package className="h-6 w-6" /> }
    case 'starting-container':
      return { title: 'Запуск контейнера', detail: 'Поднимаю контейнер с GPU passthrough', icon: <PlayCircle className="h-6 w-6" /> }
    case 'waiting-container-health':
      return { title: 'Ожидание готовности', detail: `FastAPI внутри контейнера запускается (попытка ${phase.attempt})`, icon: <Heart className="h-6 w-6" /> }
    case 'starting-python':
      return { title: 'Запуск Python', detail: 'Нативный режим — запускаю uvicorn', icon: <Cpu className="h-6 w-6" /> }
    case 'python-starting':
      return { title: 'Python запускается', detail: phase.line ?? 'Ожидание готовности uvicorn', icon: <Cpu className="h-6 w-6" /> }
    case 'ready':
      return { title: 'Готово', detail: `Backend (${phase.mode}) на порту ${phase.port}`, icon: <Server className="h-6 w-6" /> }
    case 'failed':
      return { title: 'Не удалось запустить backend', detail: phase.error, icon: <AlertCircle className="h-6 w-6" /> }
    default:
      return { title: 'Инициализация', detail: '', icon: <Cog className="h-6 w-6" /> }
  }
}

function isFailure(phase: BootPhase): boolean {
  return phase.phase === 'failed'
}

export function BackendBoot({ children }: BackendBootProps) {
  const queryClient = useQueryClient()
  const { backendOnline, backendError, backendPort, bootPhase, checkBackend, restartBackend, subscribeToBootPhase, setBackendMode, config, loadConfig } = useDockerStore()
  const [attempts, setAttempts] = useState(0)
  const [buildLog, setBuildLog] = useState<string[]>([])
  const [timeout20, setTimeout20] = useState(false)

  useEffect(() => {
    if (backendOnline) void queryClient.invalidateQueries()
  }, [backendOnline, backendPort, queryClient])

  useEffect(() => {
    loadConfig()
    const unsub = subscribeToBootPhase()
    checkBackend()
    return unsub
  }, [])

  useEffect(() => {
    if (backendOnline) return
    let cancelled = false
    const tick = async () => {
      if (cancelled) return
      await checkBackend()
      if (cancelled) return
      setAttempts((a) => a + 1)
      if (!useDockerStore.getState().backendOnline) {
        setTimeout(tick, 1500)
      }
    }
    tick()
    const failureTimer = setTimeout(() => setTimeout20(true), 30_000)
    return () => { cancelled = true; clearTimeout(failureTimer) }
  }, [backendOnline, checkBackend])

  useEffect(() => {
    if (bootPhase.phase === 'building-image' && bootPhase.line) {
      setBuildLog((prev) => [...prev.slice(-200), bootPhase.line!])
    }
  }, [bootPhase])

  const { title, detail, icon } = useMemo(() => phaseLabel(bootPhase), [bootPhase])
  const failed = isFailure(bootPhase)
  const takingLong = timeout20 && !failed

  if (backendOnline) return <>{children}</>

  const isBuilding = bootPhase.phase === 'building-image'
  const isDocker = ['checking-docker', 'docker-no-image', 'building-image', 'starting-container', 'waiting-container-health'].includes(bootPhase.phase)

  return (
    <div className="flex h-screen w-screen items-center justify-center bg-background p-8 overflow-auto">
      <div className="flex flex-col items-center gap-6 max-w-2xl w-full text-center">
        <div className="relative">
          <div className={`flex h-20 w-20 items-center justify-center rounded-2xl ${failed ? 'bg-destructive/10' : 'bg-primary/10'}`}>
            <div className={failed ? 'text-destructive' : 'text-primary'}>
              {icon}
            </div>
          </div>
          {!failed && (
            <Loader2 className="absolute -bottom-1 -right-1 h-6 w-6 animate-spin text-primary" />
          )}
        </div>

        <div className="space-y-2">
          <h1 className="text-xl font-semibold">{title}</h1>
          <p className="text-sm text-muted-foreground max-w-md">
            {failed ? (backendError ?? detail) : detail}
          </p>
        </div>

        {!failed && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <div className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
            <span>
              {isDocker ? 'Docker mode' : 'Native mode'} · попытка {attempts}
            </span>
          </div>
        )}

        {isBuilding && buildLog.length > 0 && (
          <div className="w-full max-w-2xl rounded-lg border border-border bg-black/40 p-3 text-left text-xs font-mono text-muted-foreground max-h-56 overflow-auto">
            {buildLog.slice(-20).map((line, i) => (
              <div key={i} className="truncate">{line}</div>
            ))}
          </div>
        )}

        {takingLong && <p className="text-xs text-muted-foreground">Запуск занимает больше времени. Дождись завершения текущего этапа.</p>}
        {failed && (
          <>
            <div className="flex gap-2 flex-wrap justify-center">
              <Button onClick={() => { setTimeout20(false); setAttempts(0); restartBackend() }}>
                <RefreshCw className="h-4 w-4" />
                Перезапустить
              </Button>
              <Button
                variant="ghost"
                onClick={() => window.electronAPI?.backend.openLogs()}
              >
                <FileText className="h-4 w-4" />
                Логи
              </Button>
            </div>

            {config && (
              <div className="flex flex-col gap-2 items-center w-full max-w-md">
                <p className="text-xs text-muted-foreground">Или смени режим запуска backend:</p>
                <div className="flex gap-2">
                  {(['native', 'docker', 'auto'] as const).map((m) => (
                    <Button
                      key={m}
                      variant={config.backendMode === m ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => { setTimeout20(false); setAttempts(0); setBackendMode(m) }}
                    >
                      {m === 'native' ? 'Native' : m === 'docker' ? 'Docker' : 'Auto'}
                    </Button>
                  ))}
                </div>
              </div>
            )}

            <details className="w-full max-w-xl text-left">
              <summary className="text-xs text-muted-foreground cursor-pointer hover:text-foreground">
                Ручной запуск
              </summary>
              <div className="mt-2 rounded-lg border border-border bg-card p-4 text-xs font-mono text-muted-foreground">
                <div className="space-y-1">
                  <div><span className="text-primary">$</span> pip install fastapi "uvicorn[standard]" websockets pydantic</div>
                  <div><span className="text-primary">$</span> uvicorn backend.api:app --port {backendPort}</div>
                </div>
              </div>
            </details>
          </>
        )}
      </div>
    </div>
  )
}
