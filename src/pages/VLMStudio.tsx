import { useState, useRef, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select } from '@/components/ui/select'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import {
  Play,
  Square,
  ImageIcon,
  Download,
  Send,
  Trash2,
  CheckCircle2,
  Loader2,
} from 'lucide-react'
import { useVLMRuns, useModels, useDatasets, useGpuInfo, useRunLogs } from '@/api/hooks'
import { api } from '@/api/client'
import type { TrainingRun, VLMPreflightReport } from '@/api/types'
import { toast } from 'sonner'

function VLMConfigSidebar({
  stage, setStage,
  model, setModel,
  datasetPath, setDatasetPath,
  batchSize, setBatchSize,
  lr, setLr,
  maxSteps, setMaxSteps,
  tuningMethod, setTuningMethod, gpuId, setGpuId,
  backend, setBackend, algorithm, setAlgorithm, groupSize, setGroupSize, completionTokens, setCompletionTokens, reward, setReward,
  seqLen, setSeqLen, maxPixels, setMaxPixels, precision, setPrecision,
}: {
  stage: string; setStage: (v: string) => void
  model: string; setModel: (v: string) => void
  datasetPath: string; setDatasetPath: (v: string) => void
  batchSize: number; setBatchSize: (v: number) => void
  lr: number; setLr: (v: number) => void
  maxSteps: number; setMaxSteps: (v: number) => void
  backend: string; setBackend: (v: string) => void
  algorithm: string; setAlgorithm: (v: string) => void
  groupSize: number; setGroupSize: (v: number) => void
  completionTokens: number; setCompletionTokens: (v: number) => void
  reward: string; setReward: (v: string) => void
  tuningMethod: string; setTuningMethod: (v: string) => void
  gpuId: number; setGpuId: (v: number) => void
  seqLen: number; setSeqLen: (v: number) => void
  maxPixels: number | undefined; setMaxPixels: (v: number | undefined) => void
  precision: string; setPrecision: (v: string) => void
}) {
  const { t } = useTranslation()
  const { data: gpuData } = useGpuInfo()
  const { data: modelsData } = useModels()
  const { data: datasetsData } = useDatasets()

  return (
    <ScrollArea className="w-80 shrink-0 border-r border-border p-4 space-y-6 h-full">
      <div className="space-y-6">
        <div className="space-y-2">
          <Label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            VLM Stage
          </Label>
          <Select
            options={[
              { value: 'vlm_pretrain', label: t('vlm.stage.pretrain') },
              { value: 'vlm_sft', label: t('vlm.stage.sft') },
              { value: 'vlm_grpo', label: t('vlm.stage.grpo') },
            ]}
            value={stage}
            onChange={(e) => setStage(e.target.value)}
          />
          <p className="text-[11px] text-muted-foreground">GRPO использует наш homellm RL: несколько ответов на изображение, относительная награда и обновление политики.</p>
        </div>

        <div className="space-y-2">
          <Label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {t('sidebar.base_model')}
          </Label>
          <Select
            options={[
              { value: '', label: '-- HF repo or local --' },
              ...['Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3.5-2B', 'Qwen/Qwen3.5-4B',
                'Qwen/Qwen3-VL-2B-Instruct', 'Qwen/Qwen3-VL-4B-Instruct',
                'google/gemma-3-4b-it'].map(value => ({ value, label: value })),
              ...(modelsData?.models ?? []).map((m) => ({ value: m.path, label: m.name })),
            ]}
            value={model}
            onChange={(e) => setModel(e.target.value)}
          />
          <Input
            placeholder="e.g. Qwen/Qwen3.5-0.8B"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          />
        </div>

        <div className="space-y-2">
          <Label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Dataset
          </Label>
          <Select
            options={[
              { value: '', label: '-- select dataset or type path --' },
              ...(datasetsData?.datasets ?? []).map((d) => ({ value: d.path, label: d.name })),
            ]}
            value={datasetPath}
            onChange={(e) => setDatasetPath(e.target.value)}
          />
          <Input
            placeholder="datasets/my_vlm_dataset.jsonl"
            value={datasetPath}
            onChange={(e) => setDatasetPath(e.target.value)}
          />
          {!datasetPath && (
            <p className="text-[11px] text-amber-500">
              Для VLM-тренировки нужен dataset_path
            </p>
          )}
        </div>

        <div className="space-y-3">
          <Label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {t('sidebar.hyperparams')}
          </Label>
          <Slider
            label={t('training.batch_size')}
            value={batchSize}
            onValueChange={setBatchSize}
            min={1} max={16} step={1}
          />
          <div className="space-y-1">
            <Label className="text-xs">{t('training.learning_rate')}</Label>
            <Input
              type="number"
              step="0.000001"
              value={lr}
              onChange={(e) => setLr(parseFloat(e.target.value))}
            />
          </div>
          <Slider label={t('training.max_steps')} value={maxSteps} onValueChange={setMaxSteps}
            min={1} max={10000} step={1} />
          <Label className="text-xs">Training backend</Label>
          <Select aria-label="VLM backend" value={backend} onChange={event => setBackend(event.target.value)}
            options={[{ value: 'transformers', label: 'Transformers' }, { value: 'unsloth', label: 'Unsloth' }]} />
          {stage === 'vlm_grpo' && <div className="space-y-2">
            <Label className="text-xs">Policy objective</Label>
            <Select aria-label="VLM GRPO algorithm" value={algorithm} onChange={event => setAlgorithm(event.target.value)}
              options={['grpo', 'dr_grpo', 'dapo', 'sdpo'].map(value => ({ value, label: value.toUpperCase() }))} />
            <Label className="text-xs">Ответов на один prompt</Label>
            <Input aria-label="VLM GRPO group size" type="number" min={2} max={64} value={groupSize}
              onChange={event => setGroupSize(Number(event.target.value))} />
            <Label className="text-xs">Максимум токенов ответа</Label>
            <Input aria-label="VLM GRPO completion tokens" type="number" min={1} value={completionTokens}
              onChange={event => setCompletionTokens(Number(event.target.value))} />
            <Label className="text-xs">Reward</Label>
            <Select aria-label="VLM GRPO reward" value={reward} onChange={event => setReward(event.target.value)}
              options={[{ value: 'exact_match', label: 'Exact answer match' }, { value: 'contains', label: 'Answer contained in response' }]} />
            <p className="text-[11px] text-muted-foreground">Нужен эталонный answer. Batch size задаёт число prompt-групп; градиенты считаются по одному ответу для экономии VRAM. Одинаковые награды дают нулевой reward-сигнал.</p>
          </div>}
          <Label className="text-xs">Tuning</Label>
          <Select aria-label="VLM tuning" options={[{ value: 'full', label: 'Full fine-tuning' },
            { value: 'lora', label: 'LoRA' }, { value: 'qlora', label: 'QLoRA (4-bit)' }]}
            value={tuningMethod} onChange={event => setTuningMethod(event.target.value)} />
          <Label className="text-xs">GPU</Label>
          <Select aria-label="VLM GPU" options={gpuData?.gpus?.length ? gpuData.gpus.map(gpu => ({ value: String(gpu.id), label: `GPU ${gpu.id}` }))
            : [{ value: '0', label: 'GPU 0' }]} value={String(gpuId)} onChange={event => setGpuId(Number(event.target.value))} />
          <Label htmlFor="vlm-seq-len" className="text-xs">Максимум токенов после processor</Label>
          <Input id="vlm-seq-len" type="number" min={1} value={seqLen} onChange={event => setSeqLen(Number(event.target.value))} />
          <Label htmlFor="vlm-max-pixels" className="text-xs">Лимит пикселей изображения (опционально)</Label>
          <Input id="vlm-max-pixels" type="number" min={1} value={maxPixels ?? ''} placeholder="Processor default"
            onChange={event => setMaxPixels(event.target.value === '' ? undefined : Number(event.target.value))} />
          <Label className="text-xs">Precision</Label>
          <Select aria-label="VLM precision" options={[{ value: 'no', label: 'Float32' }, { value: 'fp16', label: 'Float16' }, { value: 'bf16', label: 'BFloat16' }]}
            value={precision} onChange={event => setPrecision(event.target.value)} />
        </div>

        {gpuData?.gpus?.map((gpu) => (
          <div key={gpu.id} className="rounded-lg bg-muted/50 p-2.5 space-y-1">
            <div className="flex justify-between text-xs">
              <span>GPU {gpu.id}</span>
              <span>{gpu.memory_used_gb.toFixed(1)} / {gpu.memory_total_gb.toFixed(1)} GB</span>
            </div>
            <div className="h-1.5 w-full rounded-full bg-secondary overflow-hidden">
              <div className="h-full bg-primary" style={{ width: `${gpu.memory_percent}%` }} />
            </div>
          </div>
        ))}
      </div>
    </ScrollArea>
  )
}

function VLMLaunchTab({ config }: { config: Record<string, unknown> }) {
  const { t } = useTranslation()
  const [starting, setStarting] = useState(false)
  const { data, refetch } = useVLMRuns()
  const activeRun = data?.runs.find(run => run.process_running ?? !['completed', 'error', 'stopped', 'unknown'].includes(run.status))
  const runId = activeRun?.run_id
  const running = starting || !!activeRun
  const [lastRunId, setLastRunId] = useState<string | null>(null)
  useEffect(() => { if (runId && !lastRunId) setLastRunId(runId) }, [runId, lastRunId])
  const observedId = lastRunId ?? runId ?? null
  const { data: metricsData } = useQuery({ queryKey: ['vlm', 'metrics', observedId],
    queryFn: () => api.runMetrics(observedId!), enabled: !!observedId, refetchInterval: 2000 })
  const { data: logs } = useRunLogs(observedId)
  const metrics = metricsData?.metrics
  const [preflight, setPreflight] = useState<VLMPreflightReport | null>(null)
  const [verifiedKey, setVerifiedKey] = useState<string | null>(null)
  const [checking, setChecking] = useState(false)
  const [checkError, setCheckError] = useState<string | null>(null)
  const configKey = JSON.stringify(config)
  const preflightReady = preflight?.ready && verifiedKey === configKey
  const currentKey = useRef(configKey)
  currentKey.current = configKey
  const checkVersion = useRef(0)
  useEffect(() => {
    checkVersion.current++
    setPreflight(null)
    setVerifiedKey(null)
    setCheckError(null)
    setChecking(false)
    return () => { checkVersion.current++ }
  }, [configKey])

  const handleCheck = async () => {
    const version = ++checkVersion.current
    setChecking(true)
    setPreflight(null)
    setCheckError(null)
    try {
      const report = await api.preflightVLM(config)
      if (version === checkVersion.current && configKey === currentKey.current) {
        setPreflight(report)
        setVerifiedKey(configKey)
      }
    } catch (error) {
      if (version === checkVersion.current) setCheckError(error instanceof Error ? error.message : String(error))
    } finally {
      if (version === checkVersion.current) setChecking(false)
    }
  }

  const handleStart = async () => {
    if (!preflightReady || checking) return
    if (!config.model_path || !config.dataset_path) {
      toast.error('Выбери модель и dataset_path')
      return
    }
    try {
      setStarting(true)
      const { run_id } = await api.startVLMTraining(config)
      setLastRunId(run_id)
      await refetch()
      toast.success(`Training started: ${run_id}`)
    } catch (e) {
      toast.error(`Failed to start: ${e}`)
    } finally {
      setStarting(false)
    }
  }

  const handleStop = async () => {
    if (!runId) return
    try {
      await api.stopVLMTraining(runId)
      toast.success('Training stopped')
      await refetch()
    } catch (e) {
      toast.error(`Failed to stop: ${e}`)
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>{t('vlm.title')}</CardTitle>
          <CardDescription>{t('vlm.subtitle')}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-3">
            <Button variant="outline" onClick={handleCheck} disabled={checking || running || !config.model_path || !config.dataset_path}>
              {checking && <Loader2 className="h-4 w-4 animate-spin" />}
              Проверить эксперимент
            </Button>
            {!running ? (
              <Button onClick={handleStart} disabled={checking || !preflightReady}>
                <Play className="h-4 w-4" />
                {t('button.start_training')}
              </Button>
            ) : (
              <Button variant="destructive" onClick={handleStop} disabled={starting || !runId}>
                <Square className="h-4 w-4" />
                {t('button.stop_training')}
              </Button>
            )}
            {runId && (
              <Badge variant="secondary" className="font-mono">{runId}</Badge>
            )}
          </div>
          <p className="mt-3 text-xs text-muted-foreground">Перед запуском проверь модель и данные. Изменение настроек сбрасывает результат проверки.</p>
        </CardContent>
      </Card>

      {(checking || preflight || checkError) && <Card>
        <CardHeader><CardTitle className="text-sm">Проверка VLM-эксперимента</CardTitle>
          <CardDescription>Без загрузки весов модели. Проверяется processor и ограниченная выборка данных.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {checking && <p role="status" className="text-sm">Проверяем окружение, изображения и маски loss…</p>}
          {checkError && <p role="alert" className="text-sm text-destructive">{checkError}</p>}
          {preflight && <>
            <Badge variant={preflight.ready ? 'success' : 'destructive'}>{preflight.ready ? 'Проверка пройдена' : 'Нужно исправить ошибки'}</Badge>
            <div className="space-y-2">
              {preflight.checks.map((check, index) => <div key={index} className="text-sm">
                <span className={check.status === 'error' ? 'text-destructive font-medium' : 'font-medium'}>{check.name}</span>
                <p className="text-xs whitespace-pre-wrap text-muted-foreground">{check.detail}</p>
              </div>)}
            </div>
            {preflight.datasets.map(dataset => <p key={dataset.kind} className="text-xs text-muted-foreground">
              {dataset.kind}: строк {dataset.scanned_rows}, проверенных примеров {dataset.valid_rows}, изображений {dataset.images_checked}.
              {!dataset.complete && ' Проверена только часть файла.'}
            </p>)}
            {preflight.sample && <div className="space-y-3">
              <div className="text-sm font-medium">Пример: строка {preflight.sample.line}</div>
              <div className="flex flex-wrap gap-3">
                {preflight.sample.images.map((image, index) => <figure key={index} className="max-w-64 space-y-1">
                  <img src={image.src} alt={`Изображение примера ${index + 1}`} className="max-h-48 rounded border border-border object-contain" />
                  <figcaption className="text-xs text-muted-foreground break-all">{image.width} × {image.height} · {image.reference}</figcaption>
                </figure>)}
              </div>
              <p className="text-xs">Токенов после processor: {preflight.sample.expanded_tokens}; целей loss: {preflight.sample.supervised_tokens}.</p>
              <p className="text-xs text-muted-foreground">Выделенные токены участвуют в loss; остальные замаскированы. Первый токен не является целью next-token loss.</p>
              <div className="flex flex-wrap gap-1 rounded bg-muted/30 p-3 max-h-64 overflow-auto font-mono text-xs">
                {preflight.sample.tokens.map(token => <span key={token.index} title={`Позиция ${token.index}, ID ${token.id}`}
                  className={token.trainable ? 'rounded bg-primary/15 px-1 text-primary' : 'px-1 text-muted-foreground'}>{token.text}</span>)}
              </div>
              {preflight.sample.omitted_tokens > 0 && <p className="text-xs text-muted-foreground">В середине скрыто {preflight.sample.omitted_tokens} токенов; показаны начало и конец.</p>}
            </div>}
          </>}
        </CardContent>
      </Card>}

      {observedId && <Card>
        <CardHeader><CardTitle className="text-sm">VLM Training · {observedId}</CardTitle></CardHeader>
        <CardContent className="space-y-3">
          <div className="flex gap-4 text-sm">
            <span>{metrics?.status ?? 'initializing'}</span>
            <span>Step {metrics?.step ?? 0}</span>
            <span>Loss {metrics?.loss?.toFixed(5) ?? '—'}</span>
          </div>
          {metrics?.error && <pre role="alert" className="max-h-48 overflow-auto whitespace-pre-wrap text-xs text-destructive">{metrics.error}</pre>}
          {metrics?.training_warning && <p role="status" className="text-xs text-amber-500">{metrics.training_warning}</p>}
          {config.stage === 'vlm_grpo' && <p className="text-xs text-muted-foreground">
            Reward: {metrics?.reward ?? '—'} · KL: {metrics?.kl_divergence ?? '—'}
          </p>}
          {metrics?.final_model_dir && <p className="break-all text-xs">Model: {metrics.final_model_dir}</p>}
          <details><summary className="cursor-pointer text-xs">Training logs</summary>
            <pre className="max-h-64 overflow-auto whitespace-pre-wrap text-xs">{logs?.stderr || logs?.stdout || 'Waiting for worker output…'}</pre>
          </details>
        </CardContent>
      </Card>}

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Config Preview</CardTitle>
        </CardHeader>
        <CardContent>
          <pre className="text-xs font-mono bg-muted/30 rounded-lg p-3 overflow-auto max-h-64">
            {JSON.stringify(config, null, 2)}
          </pre>
        </CardContent>
      </Card>
    </div>
  )
}

function VLMInferenceTab() {
  const { data: modelsData } = useModels()
  const [modelPath, setModelPath] = useState('')
  const [loaded, setLoaded] = useState(false)
  const [loading, setLoading] = useState(false)
  const [prompt, setPrompt] = useState('Describe this image.')
  const [imageData, setImageData] = useState<string | null>(null)
  const [response, setResponse] = useState('')
  const [generating, setGenerating] = useState(false)
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(256)
  const fileInputRef = useRef<HTMLInputElement>(null)
  useEffect(() => {
    let active = true
    api.vlmStatus().then(status => {
      if (active && status.loaded) { setLoaded(true); setModelPath(status.model_path ?? '') }
    }).catch(() => {})
    return () => { active = false }
  }, [])

  const handleLoad = async () => {
    if (!modelPath) { toast.error('Select a model'); return }
    setLoading(true)
    try {
      await api.loadVLM(modelPath)
      setLoaded(true)
      toast.success(`Model loaded: ${modelPath}`)
    } catch (e) {
      toast.error(`Failed to load: ${e}`)
    } finally {
      setLoading(false)
    }
  }

  const handleUnload = async () => {
    try {
      await api.unloadVLM()
      setLoaded(false)
      toast.success('Model unloaded')
    } catch { /* ignore */ }
  }

  const handleImageUpload = (file: File) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      const result = e.target?.result as string
      setImageData(result)
    }
    reader.readAsDataURL(file)
  }

  const handleGenerate = async () => {
    if (!imageData || !prompt) { toast.error('Image and prompt required'); return }
    setGenerating(true)
    setResponse('')
    try {
      const { text } = await api.generateVLM({
        prompt,
        image_base64: imageData,
        max_tokens: maxTokens,
        temperature,
      })
      setResponse(text)
    } catch (e) {
      toast.error(`Generation failed: ${e}`)
    } finally {
      setGenerating(false)
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm flex items-center gap-2">
            Model Loading
            {loaded && <CheckCircle2 className="h-4 w-4 text-green-500" />}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <div className="flex-1 space-y-1">
              <Select
                options={[
                  { value: '', label: '-- local model --' },
                  ...(modelsData?.models ?? []).map((m) => ({ value: m.path, label: m.name })),
                ]}
                value={modelPath}
                disabled={loaded || loading}
                onChange={(e) => setModelPath(e.target.value)}
              />
              <Input
                placeholder="or HF repo_id (e.g. Qwen/Qwen3.5-0.8B)"
                value={modelPath}
                disabled={loaded || loading}
                onChange={(e) => setModelPath(e.target.value)}
              />
            </div>
            {!loaded ? (
              <Button onClick={handleLoad} disabled={loading}>
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
                Load
              </Button>
            ) : (
              <Button variant="outline" onClick={handleUnload}>Unload</Button>
            )}
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Image</CardTitle>
          </CardHeader>
          <CardContent
            onDragOver={(e) => { e.preventDefault() }}
            onDrop={(e) => {
              e.preventDefault()
              const file = e.dataTransfer.files[0]
              if (file && file.type.startsWith('image/')) handleImageUpload(file)
            }}
          >
            {imageData ? (
              <div className="space-y-2">
                <img src={imageData} alt="Input" className="w-full rounded-lg max-h-96 object-contain bg-muted/30" />
                <Button variant="outline" size="sm" onClick={() => setImageData(null)}>
                  <Trash2 className="h-3.5 w-3.5" />
                  Clear
                </Button>
              </div>
            ) : (
              <div
                onClick={() => fileInputRef.current?.click()}
                className="flex flex-col items-center justify-center h-64 border-2 border-dashed border-border rounded-lg cursor-pointer hover:bg-accent/20 transition-colors"
              >
                <ImageIcon className="h-12 w-12 text-muted-foreground/40 mb-2" />
                <p className="text-sm text-muted-foreground">Click or drag to upload image</p>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (file) handleImageUpload(file)
              }}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Prompt & Settings</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="space-y-1">
              <Label className="text-xs">Prompt</Label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={4}
                className="w-full rounded-md border border-border bg-transparent px-3 py-2 text-sm resize-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              />
            </div>
            <Slider
              label="Temperature"
              value={temperature}
              onValueChange={setTemperature}
              min={0} max={2} step={0.1}
            />
            <Slider
              label="Max tokens"
              value={maxTokens}
              onValueChange={setMaxTokens}
              min={16} max={2048} step={16}
            />
            <Button
              onClick={handleGenerate}
              disabled={!loaded || !imageData || generating}
              className="w-full"
            >
              {generating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              Generate
            </Button>
          </CardContent>
        </Card>
      </div>

      {(response || generating) && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Response</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="rounded-lg bg-muted/30 p-4 text-sm whitespace-pre-wrap">
              {generating && !response ? (
                <span className="text-muted-foreground italic">Generating...</span>
              ) : (
                response
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function VLMHistoryTab() {
  const { t } = useTranslation()
  const { data, isLoading } = useVLMRuns()
  const runs = data?.runs ?? []

  const statusColor = (s: string): 'success' | 'default' | 'destructive' | 'secondary' => {
    switch (s) {
      case 'running': return 'success'
      case 'completed': return 'default'
      case 'error': return 'destructive'
      default: return 'secondary'
    }
  }

  return (
    <div className="space-y-3">
      {isLoading && <div className="text-muted-foreground text-sm">{t('common.loading')}</div>}
      {runs.length === 0 && !isLoading && (
        <div className="text-muted-foreground text-sm py-8 text-center">No VLM runs yet</div>
      )}
      {runs.map((run: TrainingRun) => (
        <Card key={run.run_id}>
          <CardContent className="p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Badge variant={statusColor(run.status)}>{run.status}</Badge>
              <span className="font-mono text-sm">{run.experiment_name || run.run_id.slice(0, 12)}</span>
              <span className="text-xs text-muted-foreground">{run.stage}</span>
            </div>
            <span className="text-xs text-muted-foreground">
              {run.current_step}/{run.total_steps}
            </span>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}

export function VLMStudio() {
  const { t } = useTranslation()
  const [activeTab, setActiveTab] = useState('launch')

  const [stage, setStage] = useState('vlm_sft')
  const [model, setModel] = useState('')
  const [datasetPath, setDatasetPath] = useState('')
  const [batchSize, setBatchSize] = useState(2)
  const [lr, setLr] = useState(2e-5)
  const [maxSteps, setMaxSteps] = useState(500)
  const [epochs] = useState(1)
  const [backend, setBackend] = useState('transformers')
  const [algorithm, setAlgorithm] = useState('grpo')
  const [groupSize, setGroupSize] = useState(4)
  const [completionTokens, setCompletionTokens] = useState(256)
  const [reward, setReward] = useState('exact_match')
  const [experimentName] = useState(() => `vlm_${Date.now().toString(36)}`)
  const [tuningMethod, setTuningMethod] = useState('lora')
  const [gpuId, setGpuId] = useState(0)
  const [seqLen, setSeqLen] = useState(2048)
  const [maxPixels, setMaxPixels] = useState<number | undefined>(undefined)
  const [precision, setPrecision] = useState('bf16')

  const config = {
    stage,
    model_path: model,
    dataset_path: datasetPath,
    batch_size: batchSize,
    learning_rate: lr,
    max_steps: maxSteps,
    training_backend: backend,
    grpo_algorithm: algorithm,
    grpo_group_size: groupSize,
    grpo_max_new_tokens: completionTokens,
    grpo_reward_fn: reward,
    num_epochs: epochs,
    use_lora: tuningMethod !== 'full',
    tuning_method: tuningMethod,
    gpu_ids: [gpuId],
    seq_len: seqLen,
    max_pixels: maxPixels,
    mixed_precision: precision,
    experiment_name: experimentName,
  }

  return (
    <div className="flex h-full">
      <VLMConfigSidebar
        stage={stage} setStage={setStage}
        model={model} setModel={setModel}
          datasetPath={datasetPath} setDatasetPath={setDatasetPath}
        batchSize={batchSize} setBatchSize={setBatchSize}
        lr={lr} setLr={setLr}
        maxSteps={maxSteps} setMaxSteps={setMaxSteps}
        backend={backend} setBackend={setBackend} algorithm={algorithm} setAlgorithm={setAlgorithm}
        groupSize={groupSize} setGroupSize={setGroupSize} completionTokens={completionTokens} setCompletionTokens={setCompletionTokens}
        reward={reward} setReward={setReward}
        tuningMethod={tuningMethod} setTuningMethod={setTuningMethod} gpuId={gpuId} setGpuId={setGpuId}
        seqLen={seqLen} setSeqLen={setSeqLen} maxPixels={maxPixels} setMaxPixels={setMaxPixels} precision={precision} setPrecision={setPrecision}
      />
      <div className="flex-1 p-6 overflow-auto">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList>
            <TabsTrigger value="launch">{t('tabs.launch')}</TabsTrigger>
            <TabsTrigger value="inference">Inference</TabsTrigger>
            <TabsTrigger value="history">{t('tabs.history')}</TabsTrigger>
          </TabsList>

          <TabsContent value="launch"><VLMLaunchTab config={config} /></TabsContent>
          <TabsContent value="inference"><VLMInferenceTab /></TabsContent>
          <TabsContent value="history"><VLMHistoryTab /></TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
