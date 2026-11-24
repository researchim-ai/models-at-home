import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Globe, Moon, Sun, Server, RefreshCw, Info, Container, Cpu, Zap,
  CheckCircle2, XCircle, Package, Terminal,
} from 'lucide-react'
import { useSettingsStore } from '@/stores/settingsStore'
import { useDockerStore } from '@/stores/dockerStore'
import { toast } from 'sonner'
import type { BackendMode } from '@/types/electron'

export function SettingsPage() {
  const { t, i18n } = useTranslation()
  const { language, theme, setTheme, toggleLanguage } = useSettingsStore()
  const {
    backendOnline,
    backendMode,
    backendPort,
    status: dockerStatus,
    config,
    restartBackend,
    startDocker,
    stopDocker,
    checkDocker,
    loadConfig,
    setBackendMode,
  } = useDockerStore()

  const [dockerAvailable, setDockerAvailable] = useState<boolean | null>(null)
  const [imageExists, setImageExists] = useState<boolean | null>(null)
  const [buildingImage, setBuildingImage] = useState(false)
  const [buildLog, setBuildLog] = useState<string[]>([])

  const isElectron = !!window.electronAPI

  useEffect(() => {
    loadConfig()
    if (isElectron && window.electronAPI) {
      window.electronAPI.docker.available().then(setDockerAvailable).catch(() => setDockerAvailable(false))
      window.electronAPI.docker.imageExists().then(setImageExists).catch(() => setImageExists(false))
      const unsub = window.electronAPI.docker.onBuildProgress((line) => {
        setBuildLog((prev) => [...prev.slice(-300), line])
      })
      return unsub
    }
  }, [isElectron, loadConfig])

  const refreshDockerInfo = async () => {
    if (!window.electronAPI) return
    const [avail, img] = await Promise.all([
      window.electronAPI.docker.available(),
      window.electronAPI.docker.imageExists(),
    ])
    setDockerAvailable(avail)
    setImageExists(img)
    checkDocker()
  }

  const handleModeChange = async (mode: BackendMode) => {
    await setBackendMode(mode)
    toast.success(`Backend режим: ${mode}`)
  }

  const handleBuildImage = async () => {
    if (!window.electronAPI) return
    setBuildingImage(true)
    setBuildLog([])
    try {
      const r = await window.electronAPI.docker.build()
      if (r.success) {
        toast.success('Docker образ собран')
        setImageExists(true)
      } else {
        toast.error(`Build failed: ${r.error}`)
      }
    } finally {
      setBuildingImage(false)
    }
  }

  const handleLangChange = (lang: string) => {
    i18n.changeLanguage(lang)
    if (lang !== language) toggleLanguage()
  }

  const modes: { id: BackendMode; label: string; desc: string; icon: React.ReactNode }[] = [
    { id: 'auto', label: 'Auto', desc: 'Docker если доступен, иначе Python', icon: <Zap className="h-4 w-4" /> },
    { id: 'native', label: 'Native', desc: 'Локальный Python (быстрее, но нужны зависимости на хосте)', icon: <Cpu className="h-4 w-4" /> },
    { id: 'docker', label: 'Docker', desc: 'Полная изоляция + GPU passthrough', icon: <Container className="h-4 w-4" /> },
  ]

  return (
    <div className="p-6 space-y-6 max-w-3xl mx-auto">
      <div>
        <h2 className="text-xl font-semibold">{t('settings.title')}</h2>
        <p className="text-sm text-muted-foreground mt-1">Configure your studio preferences</p>
      </div>

      {/* Backend Mode */}
      {isElectron && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              <Server className="h-4 w-4" />
              Backend Mode
            </CardTitle>
            <CardDescription>
              Как запускать FastAPI backend. Смена режима перезапустит backend.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-3 gap-2">
              {modes.map((m) => (
                <button
                  key={m.id}
                  onClick={() => handleModeChange(m.id)}
                  className={`flex flex-col items-start gap-1 rounded-lg border p-3 text-left transition-colors ${
                    config?.backendMode === m.id
                      ? 'border-primary bg-primary/5'
                      : 'border-border hover:bg-accent'
                  }`}
                >
                  <div className="flex items-center gap-2 text-sm font-medium">
                    {m.icon}
                    {m.label}
                  </div>
                  <p className="text-xs text-muted-foreground">{m.desc}</p>
                </button>
              ))}
            </div>

            <div className="flex items-center gap-3 pt-2 border-t border-border">
              <span className="text-sm">Сейчас:</span>
              <Badge variant={backendOnline ? 'success' : 'secondary'}>
                {backendOnline ? `Online (${backendMode} :${backendPort})` : 'Offline'}
              </Badge>
              <Button size="sm" variant="outline" onClick={restartBackend} className="ml-auto">
                <RefreshCw className="h-3.5 w-3.5" />
                Restart
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Docker Detail */}
      {isElectron && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              <Container className="h-4 w-4" />
              Docker Runtime
            </CardTitle>
            <CardDescription>
              Образ собирается один раз (~10-20 мин). После этого все запуски быстрые.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="rounded-lg bg-muted/50 p-3 flex items-center gap-2">
                {dockerAvailable === null ? (
                  <RefreshCw className="h-4 w-4 animate-spin text-muted-foreground" />
                ) : dockerAvailable ? (
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                ) : (
                  <XCircle className="h-4 w-4 text-destructive" />
                )}
                <span>Docker daemon: {dockerAvailable ? 'running' : 'not running'}</span>
              </div>
              <div className="rounded-lg bg-muted/50 p-3 flex items-center gap-2">
                {imageExists === null ? (
                  <RefreshCw className="h-4 w-4 animate-spin text-muted-foreground" />
                ) : imageExists ? (
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                ) : (
                  <XCircle className="h-4 w-4 text-muted-foreground" />
                )}
                <span>Image: {imageExists ? 'built' : 'not built'}</span>
              </div>
              <div className="rounded-lg bg-muted/50 p-3 flex items-center gap-2">
                <Server className="h-4 w-4 text-muted-foreground" />
                <span>Container: {dockerStatus}</span>
              </div>
              <div className="rounded-lg bg-muted/50 p-3 flex items-center gap-2">
                <Zap className="h-4 w-4 text-muted-foreground" />
                <span>GPU: {config?.dockerGpu ? 'enabled' : 'disabled'}</span>
              </div>
            </div>

            <div className="flex gap-2 flex-wrap">
              <Button
                size="sm"
                onClick={handleBuildImage}
                disabled={!dockerAvailable || buildingImage}
              >
                <Package className="h-3.5 w-3.5" />
                {buildingImage ? 'Building...' : (imageExists ? 'Rebuild Image' : 'Build Image')}
              </Button>
              <Button size="sm" variant="outline" onClick={startDocker} disabled={!imageExists || dockerStatus === 'running'}>
                Start container
              </Button>
              <Button size="sm" variant="outline" onClick={stopDocker} disabled={dockerStatus !== 'running'}>
                Stop
              </Button>
              <Button size="sm" variant="ghost" onClick={refreshDockerInfo}>
                <RefreshCw className="h-3.5 w-3.5" />
              </Button>
            </div>

            {(buildingImage || buildLog.length > 0) && (
              <div className="rounded-lg border border-border bg-black/40 p-3 text-xs font-mono text-muted-foreground max-h-64 overflow-auto">
                <div className="flex items-center gap-2 text-foreground mb-2">
                  <Terminal className="h-3.5 w-3.5" />
                  Build log
                </div>
                {buildLog.slice(-50).map((line, i) => (
                  <div key={i} className="truncate">{line}</div>
                ))}
                {buildingImage && <div className="animate-pulse">...</div>}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Language */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm flex items-center gap-2">
            <Globe className="h-4 w-4" />
            {t('settings.language')}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-3">
            <Button variant={language === 'en' ? 'default' : 'outline'} onClick={() => handleLangChange('en')}>
              English
            </Button>
            <Button variant={language === 'ru' ? 'default' : 'outline'} onClick={() => handleLangChange('ru')}>
              Русский
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Theme */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm flex items-center gap-2">
            {theme === 'dark' ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
            {t('settings.theme')}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-3">
            <Button variant={theme === 'dark' ? 'default' : 'outline'} onClick={() => setTheme('dark')}>
              <Moon className="h-4 w-4" />
              Dark
            </Button>
            <Button variant={theme === 'light' ? 'default' : 'outline'} onClick={() => setTheme('light')}>
              <Sun className="h-4 w-4" />
              Light
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* About */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm flex items-center gap-2">
            <Info className="h-4 w-4" />
            About
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-1 text-sm text-muted-foreground">
            <p>Models at Home Studio v0.1.0</p>
            <p>Train LLMs, VLMs and more — right at home</p>
            <p className="text-xs mt-2">Apache 2.0 License</p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
