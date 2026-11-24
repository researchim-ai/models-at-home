import { useState, useRef, useEffect, type ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import {
  Bot,
  Play,
  Square,
  Send,
  Plus,
  Trash2,
  Server,
  Wrench,
  Activity,
  MessageSquare,
  Sparkles,
  RefreshCw,
  Loader2,
} from 'lucide-react'
import {
  useAgentServerStatus,
  useAgentSessions,
  useAgentTools,
  useAgentRuns,
  useAgentCapabilities,
} from '@/api/hooks'
import { api, createAgentChatStream } from '@/api/client'
import type { ChatMessage, AgentSession, AgentRun } from '@/api/types'
import { toast } from 'sonner'

function AgentSidebar() {
  const { t } = useTranslation()
  const { data: serverStatus } = useAgentServerStatus()
  const { data: toolsData } = useAgentTools()
  const [model, setModel] = useState('unsloth/Qwen3.5-9B-GGUF')
  const [quant, setQuant] = useState('9B-UD-Q4_K_XL')
  const [ctxSize, setCtxSize] = useState(262144)
  const [temperature, setTemperature] = useState(0.2)
  const [gpuLayers, setGpuLayers] = useState(-1)
  const [starting, setStarting] = useState(false)

  const isRunning = serverStatus?.running ?? false
  const tools = toolsData?.tools ?? []

  const handleStart = async () => {
    setStarting(true)
    try {
      await api.startAgentServer({
        model_repo: model,
        quant,
        ctx_size: ctxSize,
        gpu_layers: gpuLayers,
        temperature,
      })
      toast.success(t('agent.server_running'))
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t('error.generic'))
    } finally {
      setStarting(false)
    }
  }

  const handleStop = async () => {
    try {
      await api.stopAgentServer()
      toast.success(t('agent.server_stopped'))
    } catch {
      toast.error(t('error.generic'))
    }
  }

  return (
    <ScrollArea className="w-80 shrink-0 border-r border-border p-4 h-full">
      <div className="space-y-6">
        {/* Server status */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Server className="h-4 w-4 text-muted-foreground" />
            <Label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              llama-server
            </Label>
            <Badge variant={isRunning ? 'success' : 'secondary'} className="ml-auto text-[10px]">
              {isRunning ? t('agent.server_running') : t('agent.server_stopped')}
            </Badge>
          </div>

          <div className="flex gap-2">
            {isRunning ? (
              <Button variant="destructive" size="sm" className="w-full" onClick={handleStop}>
                <Square className="h-3.5 w-3.5" />
                {t('agent.stop_server')}
              </Button>
            ) : (
              <Button size="sm" className="w-full" onClick={handleStart} disabled={starting}>
                {starting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
                {starting ? t('agent.starting') : t('agent.start_server')}
              </Button>
            )}
          </div>
          {isRunning && serverStatus?.model && (
            <p className="text-[10px] text-muted-foreground truncate" title={serverStatus.model}>
              {serverStatus.model}
            </p>
          )}
        </div>

        {/* Model config */}
        <div className="space-y-3">
          <Label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {t('agent.model_gguf')}
          </Label>
          <Input
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder="HuggingFace repo"
          />
          <Input
            value={quant}
            onChange={(e) => setQuant(e.target.value)}
            placeholder="Quantization"
          />
          <Slider
            label={t('agent.context_size')}
            value={ctxSize}
            onValueChange={setCtxSize}
            min={2048} max={524288} step={2048}
            formatValue={(v) => `${(v / 1024).toFixed(0)}K`}
          />
          <Slider
            label={t('agent.gpu_layers')}
            value={gpuLayers}
            onValueChange={setGpuLayers}
            min={-1} max={100} step={1}
            formatValue={(v) => (v < 0 ? t('agent.gpu_layers_max') : String(v))}
          />
          <Slider
            label="Temperature"
            value={temperature}
            onValueChange={setTemperature}
            min={0} max={2} step={0.05}
            formatValue={(v) => v.toFixed(2)}
          />
        </div>

        {/* Tool capabilities (live) */}
        <div className="space-y-2">
          <Label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {t('agent.tools')} {tools.length > 0 && `(${tools.length})`}
          </Label>
          <div className="space-y-1">
            {tools.length === 0 ? (
              <p className="text-[11px] text-muted-foreground">{t('agent.tools_loading')}</p>
            ) : (
              tools.map((tool) => (
                <div key={tool.name} className="flex items-center gap-2 rounded-md bg-muted/50 px-2.5 py-1.5">
                  <Wrench className="h-3 w-3 text-muted-foreground shrink-0" />
                  <span className="text-xs font-mono truncate">{tool.name}</span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </ScrollArea>
  )
}

function ChatPanel() {
  const { t } = useTranslation()
  const { data: serverStatus } = useAgentServerStatus()
  const { data: sessions, refetch: refetchSessions } = useAgentSessions()
  const [activeSession, setActiveSession] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const streamRef = useRef<AbortController | null>(null)
  useEffect(() => () => streamRef.current?.abort(), [])

  const isRunning = serverStatus?.running ?? false

  const handleNewSession = async () => {
    try {
      const session = await api.createAgentSession()
      setActiveSession(session.id)
      setMessages([])
      refetchSessions()
    } catch {
      toast.error(t('error.generic'))
    }
  }

  const handleDeleteSession = async (id: string) => {
    try {
      await api.deleteAgentSession(id)
      if (activeSession === id) {
        setActiveSession(null)
        setMessages([])
      }
      refetchSessions()
      toast.success(t('agent.session_deleted'))
    } catch {
      toast.error(t('error.generic'))
    }
  }

  const handleSend = () => {
    if (!input.trim() || !activeSession || isGenerating) return
    if (!isRunning) {
      toast.error(t('agent.server_required'))
      return
    }

    const userMsg: ChatMessage = { role: 'user', content: input }
    setMessages((prev) => [...prev, userMsg])
    const sentMessage = input
    setInput('')
    setIsGenerating(true)

    let assistantContent = ''
    setMessages((prev) => [...prev, { role: 'assistant', content: '' }])

    streamRef.current = createAgentChatStream(
      activeSession,
      sentMessage,
      (text) => {
        assistantContent += text
        setMessages((prev) => {
          const copy = [...prev]
          copy[copy.length - 1] = { role: 'assistant', content: assistantContent }
          return copy
        })
      },
      (tool, args) => {
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: `🔧 [${tool}] ${JSON.stringify(args)}` },
        ])
        assistantContent = ''
        setMessages((prev) => [...prev, { role: 'assistant', content: '' }])
      },
      () => {
        setIsGenerating(false)
        // Pull the (possibly auto-generated) session name and refreshed ordering.
        refetchSessions()
      },
      (err) => {
        setIsGenerating(false)
        toast.error(err.message || t('error.generic'))
      },
    )
  }

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  return (
    <div className="flex flex-col h-full">
      {/* Session tabs */}
      <div className="flex items-center gap-2 border-b border-border px-4 py-2">
        <Button variant="outline" size="sm" onClick={handleNewSession} disabled={isGenerating}>
          <Plus className="h-3.5 w-3.5" />
          {t('agent.new_session')}
        </Button>
        <div className="flex gap-1 overflow-x-auto">
          {(sessions?.sessions ?? []).map((s: AgentSession) => (
            <div key={s.id} className="flex items-center">
              <Button
                variant={activeSession === s.id ? 'secondary' : 'ghost'}
                size="sm"
                disabled={isGenerating}
                onClick={() => { setActiveSession(s.id); setMessages(s.messages ?? []) }}
                className="rounded-r-none text-xs max-w-[180px]"
                title={s.name || s.id}
              >
                <MessageSquare className="h-3 w-3 shrink-0" />
                <span className="truncate">{s.name || s.id?.slice(0, 6) || 'session'}</span>
              </Button>
              <Button
                variant={activeSession === s.id ? 'secondary' : 'ghost'}
                size="icon"
                disabled={isGenerating}
                onClick={() => handleDeleteSession(s.id)}
                className="h-8 w-8 rounded-l-none border-l border-border/60"
              >
                <Trash2 className="h-3 w-3" />
              </Button>
            </div>
          ))}
        </div>
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 p-4" ref={scrollRef}>
        {messages.length === 0 ? (
          <div className="flex h-full items-center justify-center">
            <div className="text-center text-muted-foreground">
              <Bot className="mx-auto h-12 w-12 mb-3 opacity-20" />
              <p className="text-sm font-medium">{t('agent.title')}</p>
              <p className="text-xs mt-1">{t('agent.subtitle')}</p>
              {!activeSession && (
                <p className="text-xs mt-3 text-muted-foreground/70">{t('agent.start_session_hint')}</p>
              )}
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.filter((m) => (typeof m.content === 'string' && m.content !== '') || isGenerating).map((msg, i) => {
              const content = typeof msg.content === 'string' ? msg.content : ''
              return (
              <div
                key={i}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm ${
                    msg.role === 'user'
                      ? 'bg-primary text-primary-foreground'
                      : content.startsWith('🔧')
                        ? 'bg-amber-500/10 border border-amber-500/30 font-mono text-xs'
                        : 'bg-muted'
                  }`}
                >
                  <pre className="whitespace-pre-wrap font-sans">{content || (isGenerating && i === messages.length - 1 ? t('chat.thinking') : '')}</pre>
                </div>
              </div>
              )
            })}
          </div>
        )}
      </ScrollArea>

      {/* Input */}
      <div className="border-t border-border p-4">
        <div className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
            placeholder={t('chat.message_placeholder')}
            disabled={isGenerating || !activeSession}
          />
          <Button onClick={handleSend} disabled={isGenerating || !activeSession}>
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  )
}

function RunsPanel() {
  const { t } = useTranslation()
  const { data, isLoading, refetch, isFetching } = useAgentRuns()
  const runs = data?.runs ?? []

  const handleStop = async (runId: string) => {
    try {
      const res = await api.stopAgentRun(runId)
      if (res.stopped) toast.success(t('agent.run_stopped'))
      else toast.error(res.reason || t('error.generic'))
      refetch()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t('error.generic'))
    }
  }

  return (
    <div className="p-6 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold">{t('agent.runs_title')}</h3>
          <p className="text-xs text-muted-foreground">{t('agent.runs_subtitle')}</p>
        </div>
        <Button variant="outline" size="icon" onClick={() => refetch()}>
          <RefreshCw className={`h-4 w-4 ${isFetching ? 'animate-spin' : ''}`} />
        </Button>
      </div>

      {isLoading ? (
        <Card><CardContent className="p-8 text-center text-muted-foreground">{t('common.loading')}</CardContent></Card>
      ) : runs.length === 0 ? (
        <Card>
          <CardContent className="p-8 text-center text-muted-foreground">
            {t('agent.no_runs')}
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-2">
          {runs.map((run: AgentRun) => {
            const metrics = run.metrics ?? {}
            const status = String(metrics.status ?? (run.is_running ? 'running' : 'unknown'))
            const step = metrics.step ?? metrics.current_step
            const totalSteps = metrics.total_steps ?? metrics.max_steps
            const loss = metrics.loss ?? metrics.current_loss
            return (
              <div key={run.run_id} className="rounded-lg border border-border bg-card px-4 py-3">
                <div className="flex items-center gap-3">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-mono truncate">{run.run_id}</span>
                      <Badge variant={run.is_running ? 'success' : status === 'error' ? 'destructive' : 'secondary'} className="text-[10px]">
                        {status}
                      </Badge>
                      {run.config_summary?.stage && (
                        <Badge variant="outline" className="text-[10px]">{run.config_summary.stage}</Badge>
                      )}
                    </div>
                    <div className="text-[11px] text-muted-foreground mt-1 flex gap-3 flex-wrap">
                      {step != null && <span>step: {String(step)}{totalSteps != null ? ` / ${String(totalSteps)}` : ''}</span>}
                      {loss != null && <span>loss: {typeof loss === 'number' ? loss.toFixed(4) : String(loss)}</span>}
                      {run.config_summary?.training_backend && <span>{run.config_summary.training_backend}</span>}
                    </div>
                  </div>
                  {run.is_running && (
                    <Button variant="destructive" size="sm" onClick={() => handleStop(run.run_id)}>
                      <Square className="h-3.5 w-3.5" />
                      {t('agent.stop_run')}
                    </Button>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function ToolsPanel() {
  const { t } = useTranslation()
  const { data, isLoading } = useAgentTools()
  const groups = data?.groups ?? []

  if (isLoading) {
    return <div className="p-6"><Card><CardContent className="p-8 text-center text-muted-foreground">{t('common.loading')}</CardContent></Card></div>
  }

  return (
    <ScrollArea className="h-full">
      <div className="p-6 space-y-6">
        {groups.filter((g) => g.tools.length > 0).map((group) => (
          <div key={group.category} className="space-y-3">
            <h3 className="text-sm font-semibold">{group.title}</h3>
            <div className="grid grid-cols-2 gap-3">
              {group.tools.map((tool) => (
                <div key={tool.name} className="rounded-lg border border-border p-3">
                  <div className="flex items-center gap-2">
                    <Wrench className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                    <span className="text-sm font-mono font-medium">{tool.name}</span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">{tool.description}</p>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </ScrollArea>
  )
}

function CapabilitiesPanel() {
  const { t } = useTranslation()
  const { data, isLoading } = useAgentCapabilities()
  const capabilities = data?.capabilities ?? {}

  if (isLoading) {
    return <div className="p-6"><Card><CardContent className="p-8 text-center text-muted-foreground">{t('common.loading')}</CardContent></Card></div>
  }

  const renderValue = (value: unknown): ReactNode => {
    if (Array.isArray(value)) {
      return (
        <ul className="list-disc list-inside space-y-0.5">
          {value.map((item, i) => <li key={i} className="text-xs text-muted-foreground">{typeof item === 'string' ? item : JSON.stringify(item)}</li>)}
        </ul>
      )
    }
    if (value && typeof value === 'object') {
      return (
        <div className="space-y-1.5 pl-2 border-l border-border/60">
          {Object.entries(value as Record<string, unknown>).map(([k, v]) => (
            <div key={k}>
              <span className="text-xs font-medium">{k}:</span> {renderValue(v)}
            </div>
          ))}
        </div>
      )
    }
    return <span className="text-xs text-muted-foreground">{String(value)}</span>
  }

  return (
    <ScrollArea className="h-full">
      <div className="p-6 space-y-4">
        {Object.entries(capabilities).map(([key, value]) => (
          <Card key={key}>
            <CardHeader>
              <CardTitle className="text-sm font-mono">{key}</CardTitle>
            </CardHeader>
            <CardContent>{renderValue(value)}</CardContent>
          </Card>
        ))}
      </div>
    </ScrollArea>
  )
}

export function AgentStudio() {
  const { t } = useTranslation()
  const [activeTab, setActiveTab] = useState('chat')

  return (
    <div className="flex h-full">
      <AgentSidebar />
      <div className="flex-1 flex flex-col min-w-0">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col min-h-0">
          <div className="border-b border-border px-4">
            <TabsList className="bg-transparent">
              <TabsTrigger value="chat">
                <MessageSquare className="h-3.5 w-3.5 mr-1.5" />
                {t('agent.tab_chat')}
              </TabsTrigger>
              <TabsTrigger value="runs">
                <Activity className="h-3.5 w-3.5 mr-1.5" />
                {t('agent.tab_runs')}
              </TabsTrigger>
              <TabsTrigger value="tools">
                <Wrench className="h-3.5 w-3.5 mr-1.5" />
                {t('agent.tab_tools')}
              </TabsTrigger>
              <TabsTrigger value="capabilities">
                <Sparkles className="h-3.5 w-3.5 mr-1.5" />
                {t('agent.tab_capabilities')}
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="chat" className="flex-1 m-0 min-h-0">
            <ChatPanel />
          </TabsContent>
          <TabsContent value="runs" className="flex-1 m-0 min-h-0 overflow-auto">
            <RunsPanel />
          </TabsContent>
          <TabsContent value="tools" className="flex-1 m-0 min-h-0">
            <ToolsPanel />
          </TabsContent>
          <TabsContent value="capabilities" className="flex-1 m-0 min-h-0">
            <CapabilitiesPanel />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
