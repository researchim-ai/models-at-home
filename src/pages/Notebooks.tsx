import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { ExternalLink, FlaskConical, Check, X, RefreshCw, FileText, Copy, Play, Square } from 'lucide-react'
import { api } from '@/api/client'
import { formatBytes } from '@/lib/utils'
import { toast } from 'sonner'

interface Notebook {
  name: string
  path: string
  size_bytes: number
  modified_at: string
}

interface Template {
  name: string
  description: string
  file: string
}

export function Notebooks() {
  const { t } = useTranslation()
  const [jupyterUrl, setJupyterUrl] = useState('http://localhost:8888')
  const [token, setToken] = useState('')
  const [healthy, setHealthy] = useState<boolean | null>(null)
  const [showEmbed, setShowEmbed] = useState(false)
  const [notebooks, setNotebooks] = useState<Notebook[]>([])
  const [templates, setTemplates] = useState<Template[]>([])
  const [loading, setLoading] = useState(false)
  const [jupyterAction, setJupyterAction] = useState<'start' | 'stop' | null>(null)

  const checkHealth = async () => {
    try {
      const data = await api.jupyterStatus(jupyterUrl)
      setHealthy(data.running)
    } catch {
      setHealthy(false)
    }
  }

  const loadNotebooks = async () => {
    setLoading(true)
    try {
      const [nbData, tplData] = await Promise.all([
        api.listNotebooks(),
        api.notebookTemplates(),
      ])
      setNotebooks(nbData.notebooks ?? [])
      setTemplates(tplData.templates ?? [])
    } catch (err) {
      toast.error('Не удалось загрузить список ноутбуков')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    checkHealth()
    loadNotebooks()
  }, [])

  useEffect(() => {
    checkHealth()
  }, [jupyterUrl])

  const fullUrl = `${jupyterUrl}/lab?token=${token}`

  const startJupyter = async () => {
    setJupyterAction('start')
    try {
      const data = await api.startJupyter()
      setJupyterUrl(data.url)
      setToken(data.token)
      setHealthy(data.running)
      toast.success('JupyterLab запущен')
    } catch (err) {
      toast.error(`Не удалось запустить JupyterLab: ${err instanceof Error ? err.message : String(err)}`)
      setHealthy(false)
    } finally {
      setJupyterAction(null)
    }
  }

  const stopJupyter = async () => {
    setJupyterAction('stop')
    try {
      await api.stopJupyter()
      setHealthy(false)
      toast.success('JupyterLab остановлен')
    } catch (err) {
      toast.error(`Не удалось остановить JupyterLab: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setJupyterAction(null)
    }
  }

  const openNotebook = (path: string) => {
    const nbUrl = `${jupyterUrl}/lab/tree/${path}?token=${token}`
    const opener = window.electronAPI?.shell.openExternal
    if (opener) opener(nbUrl)
    else window.open(nbUrl, '_blank')
  }

  const copyPath = (p: string) => {
    navigator.clipboard.writeText(p)
    toast.success('Путь скопирован')
  }

  return (
    <div className="p-6 space-y-6 max-w-5xl mx-auto">
      <div>
        <h2 className="text-xl font-semibold">{t('notebooks.title')}</h2>
        <p className="text-sm text-muted-foreground mt-1">{t('notebooks.subtitle')}</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">JupyterLab Connection</CardTitle>
          <CardDescription>
            Studio может запустить JupyterLab сама. Если используешь внешний сервер, укажи URL и token вручную.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-3 items-end">
            <div className="flex-1 space-y-1">
              <label className="text-xs text-muted-foreground">URL</label>
              <Input
                value={jupyterUrl}
                onChange={(e) => setJupyterUrl(e.target.value)}
              />
            </div>
            <div className="w-48 space-y-1">
              <label className="text-xs text-muted-foreground">Token</label>
              <Input
                value={token}
                onChange={(e) => setToken(e.target.value)}
              />
            </div>
            <Button variant="outline" size="icon" onClick={checkHealth}>
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>

          <div className="flex items-center gap-3">
            <Badge variant={healthy ? 'success' : healthy === false ? 'destructive' : 'secondary'}>
              {healthy ? <Check className="h-3 w-3 mr-1" /> : healthy === false ? <X className="h-3 w-3 mr-1" /> : null}
              {healthy ? 'Connected' : healthy === false ? 'Not reachable' : 'Checking...'}
            </Badge>
          </div>

          <div className="flex gap-3">
            <Button onClick={startJupyter} disabled={jupyterAction !== null}>
              <Play className="h-4 w-4" />
              {jupyterAction === 'start' ? 'Starting...' : 'Start Jupyter'}
            </Button>
            <Button variant="outline" onClick={stopJupyter} disabled={jupyterAction !== null}>
              <Square className="h-4 w-4" />
              Stop
            </Button>
            <Button onClick={() => {
              const opener = window.electronAPI?.shell.openExternal
              if (opener) opener(fullUrl)
              else window.open(fullUrl, '_blank')
            }}>
              <ExternalLink className="h-4 w-4" />
              {t('notebooks.open')}
            </Button>
            <Button variant="outline" onClick={() => setShowEmbed(!showEmbed)}>
              <FlaskConical className="h-4 w-4" />
              {showEmbed ? 'Hide Embed' : 'Embed Here'}
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex-row items-center justify-between">
          <div>
            <CardTitle className="text-sm">Notebook Templates</CardTitle>
            <CardDescription>Quick-start notebooks shipped with the project</CardDescription>
          </div>
          <Button variant="ghost" size="icon" onClick={loadNotebooks}>
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </CardHeader>
        <CardContent>
          {templates.length === 0 && !loading && (
            <p className="text-sm text-muted-foreground text-center py-4">
              Шаблоны не найдены
            </p>
          )}
          <div className="grid grid-cols-2 gap-3">
            {templates.map((tmpl) => (
              <button
                key={tmpl.file}
                onClick={() => openNotebook(tmpl.file)}
                disabled={!healthy}
                className="flex flex-col rounded-lg border border-border p-3 text-left transition-colors hover:bg-accent disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <span className="text-sm font-medium">{tmpl.name}</span>
                <span className="text-xs text-muted-foreground mt-0.5">{tmpl.description}</span>
                <span className="text-[10px] font-mono text-muted-foreground/60 mt-1">{tmpl.file}</span>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Your Notebooks</CardTitle>
          <CardDescription>Файлы в <code className="text-xs bg-muted rounded px-1">models-at-home/notebooks/</code></CardDescription>
        </CardHeader>
        <CardContent>
          {notebooks.length === 0 && !loading && (
            <p className="text-sm text-muted-foreground text-center py-4">
              Нет ноутбуков
            </p>
          )}
          <div className="space-y-1.5">
            {notebooks.map((nb) => (
              <div
                key={nb.path}
                className="flex items-center gap-3 rounded-lg border border-border bg-card px-3 py-2 hover:bg-accent/50 transition-colors group"
              >
                <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
                <div className="flex-1 min-w-0">
                  <div className="text-sm truncate">{nb.name}</div>
                  <div className="text-[10px] text-muted-foreground">
                    {formatBytes(nb.size_bytes)} • {new Date(nb.modified_at).toLocaleString()}
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => copyPath(nb.path)}
                  className="opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <Copy className="h-3.5 w-3.5" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => openNotebook(nb.name)}
                  disabled={!healthy}
                >
                  <ExternalLink className="h-3.5 w-3.5" />
                  Open
                </Button>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {showEmbed && (
        <Card className="overflow-hidden">
          <iframe
            src={fullUrl}
            className="w-full h-[600px] border-0"
            title="JupyterLab"
          />
        </Card>
      )}
    </div>
  )
}
