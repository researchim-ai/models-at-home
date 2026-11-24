import { useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Play, Square, Loader2, CheckCircle2, Bot, Upload } from 'lucide-react'
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts'
import { api } from '@/api/client'
import type { VLAPreflightReport, VLAPrediction } from '@/api/types'
import { useGpuInfo, useRunLogs } from '@/api/hooks'
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select } from '@/components/ui/select'
import { Label } from '@/components/ui/label'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'

const initial = {
  domain: 'robotics', architecture: 'native_vla', training_backend: 'transformers', action_type: 'continuous', model_path: '', dataset_path: '', val_data_path: '',
  action_dim: 7, action_horizon: 8, state_dim: 7, hidden_dim: 128, image_size: 64, vision_grid: 4, batch_size: 8,
  gradient_accumulation: 1, learning_rate: 0.0003, max_steps: 500, save_every: 100, eval_every: 20,
  mixed_precision: 'bf16', device: 'auto', tuning_method: 'full', seq_len: 2048, max_pixels: 65536,
  action_names: [] as string[], discrete_classes: [] as number[], discrete_names: [] as string[], discrete_labels: [] as string[][],
}
const presets = {
  robotics: { action_type: 'continuous', action_dim: 7, state_dim: 7, action_horizon: 8,
    action_names: ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'gripper'], discrete_classes: [] as number[], discrete_names: [] as string[], discrete_labels: [] as string[][] },
  simulation: { action_type: 'hybrid', action_dim: 2, state_dim: 0, action_horizon: 4,
    action_names: ['steering', 'throttle'], discrete_classes: [6], discrete_names: ['command'], discrete_labels: [['none', 'jump', 'interact', 'reload', 'menu', 'confirm']] },
  desktop: { action_type: 'hybrid', action_dim: 3, state_dim: 0, action_horizon: 4,
    action_names: ['mouse_dx', 'mouse_dy', 'scroll'], discrete_classes: [4, 16], discrete_names: ['mouse_button', 'key'], discrete_labels: [['none', 'left', 'right', 'middle'], ['none', 'up', 'down', 'left', 'right', 'enter', 'escape', 'space', 'tab', 'backspace', 'ctrl', 'shift', 'alt', 'a', 'c', 'v']] },
}

export function VLAStudio() {
  const [config, setConfig] = useState({ ...initial, ...presets.robotics })
  const [gpuId, setGpuId] = useState(0)
  const [name] = useState(() => `vla_${Date.now().toString(36)}`)
  const [tab, setTab] = useState('launch')
  const [observed, setObserved] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [checking, setChecking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [report, setReport] = useState<{ key: string; value: VLAPreflightReport } | null>(null)
  const { data: gpu } = useGpuInfo()
  const demos = useQuery({ queryKey: ['vla', 'examples'], queryFn: api.vlaExamples })
  const runs = useQuery({ queryKey: ['vla', 'runs'], queryFn: api.vlaRuns, refetchInterval: 3000 })
  const active = runs.data?.runs.find(run => run.process_running)
  useEffect(() => { if (!observed && active?.run_id) setObserved(active.run_id) }, [active?.run_id, observed])
  const selected = observed ?? active?.run_id ?? null
  const metrics = useQuery({ queryKey: ['vla', 'metrics', selected], queryFn: () => api.runMetrics(selected!),
    enabled: !!selected, refetchInterval: 2000 })
  const logs = useRunLogs(selected)
  const payload = { ...config, stage: 'vla_sft', experiment_name: name, gpu_ids: [gpuId] }
  const key = JSON.stringify(payload)
  const currentKey = useRef(key)
  currentKey.current = key
  const version = useRef(0)
  useEffect(() => { version.current++; setChecking(false); setReport(null) }, [key])
  useEffect(() => () => { version.current++ }, [])
  const ready = report?.key === key && report.value.ready
  const patch = (value: Partial<typeof config>) => setConfig(previous => ({ ...previous, ...value }))
  const fail = (value: unknown) => setError(value instanceof Error ? value.message : String(value))
  async function check() {
    const requestVersion = ++version.current
    setChecking(true); setError(null)
    try {
      const value = await api.preflightVLA(payload)
      if (currentKey.current === key && version.current === requestVersion) setReport({ key, value })
    } catch (value) { if (version.current === requestVersion) fail(value) }
    finally { if (version.current === requestVersion) setChecking(false) }
  }
  async function start() {
    if (!ready || busy || active) return
    setBusy(true); setError(null)
    try { const result = await api.startVLA(payload); setObserved(result.run_id); await runs.refetch() }
    catch (value) { fail(value) } finally { setBusy(false) }
  }
  async function stop() {
    if (!active) return
    setBusy(true)
    try { await api.stopVLA(active.run_id); await runs.refetch() } catch (value) { fail(value) } finally { setBusy(false) }
  }
  const numberField = (field: keyof typeof config, label: string, min: number, step = 1) => <div className="space-y-1" key={field}>
    <Label htmlFor={`vla-${field}`} className="text-xs">{label}</Label>
    <Input id={`vla-${field}`} type="number" min={min} step={step} value={String(config[field])}
      onChange={event => {
        const caps: Record<string, number> = { action_dim: config.action_type === 'discrete' ? 4096 : 256, state_dim: 1024, action_horizon: 256, image_size: 512, vision_grid: 8, batch_size: 128, gradient_accumulation: 128 }
        const value = Math.min(caps[field] ?? 1e9, Math.max(min, Number(event.target.value) || min))
        patch({ [field]: value, ...(field === 'action_dim' ? { action_names: [] } : {}) })
      }} />
  </div>
  const sample = report?.key === key ? report.value.sample : null
  const current = metrics.data?.metrics
  const selectedRun = runs.data?.runs.find(run => run.run_id === selected)
  const exportPath = current?.status === 'completed' && selectedRun?.process_running !== true
    ? current.final_model_dir ?? selectedRun?.final_model_path ?? '' : ''

  return <div className="flex h-full">
    <aside className="w-80 shrink-0 overflow-auto border-r border-border p-4 space-y-5">
      <div className="space-y-1"><h2 className="text-sm font-semibold flex items-center gap-2"><Bot className="h-4 w-4" />VLA Configuration</h2>
        <p className="text-xs text-muted-foreground">Изображение + инструкция + состояние → action chunk.</p></div>
      <div className="space-y-2"><Label className="text-xs">Сценарий</Label>
        <Select aria-label="VLA domain" value={config.domain} onChange={event => {
          const domain = event.target.value as keyof typeof presets
          patch({ domain, ...presets[domain] })
        }} options={[{ value: 'robotics', label: 'Робототехника' }, { value: 'simulation', label: 'Игры и симуляторы' }, { value: 'desktop', label: 'Мышь и клавиатура' }]} /></div>
      <div className="space-y-2"><Label className="text-xs">Модель</Label>
        <Select aria-label="VLA architecture" value={config.architecture} onChange={event => patch({ architecture: event.target.value,
          tuning_method: event.target.value === 'native_vla' ? 'full' : 'lora', training_backend: 'transformers', device: event.target.value === 'vlm_action' ? 'auto' : config.device })}
          options={[{ value: 'native_vla', label: 'Native VLA — с нуля' }, { value: 'vlm_action', label: 'VLM + Action Head' }]} />
        {config.architecture === 'native_vla' && <>{numberField('image_size', 'Размер изображения', 16)}{numberField('vision_grid', 'Пространственная сетка', 1)}</>}
        {config.architecture === 'vlm_action' && <>
          <Select aria-label="VLA backend" value={config.training_backend} onChange={event => patch({ training_backend: event.target.value })}
            options={[{ value: 'transformers', label: 'Transformers' }, { value: 'unsloth', label: 'Unsloth' }]} />
          <Input aria-label="VLA backbone" placeholder="Qwen/Qwen3.5-0.8B or local VLM" value={config.model_path} onChange={event => patch({ model_path: event.target.value })} />
          <Select aria-label="VLA tuning" value={config.tuning_method} onChange={event => patch({ tuning_method: event.target.value })}
            options={['full', 'lora', 'qlora'].map(value => ({ value, label: value.toUpperCase() }))} />
          {numberField('seq_len', 'Контекст VLM', 1)}{numberField('max_pixels', 'Максимум пикселей', 1)}
        </>}
      </div>
      <Button variant="outline" onClick={() => {
        const example = demos.data?.examples?.find(value => value.domain === config.domain)
        if (example) patch({ ...presets[config.domain as keyof typeof presets], dataset_path: example.dataset_path, val_data_path: example.val_data_path })
      }} disabled={!demos.data?.examples?.some(value => value.domain === config.domain)}>Синтетическое демо</Button>
      <div className="space-y-2"><Label htmlFor="vla-dataset" className="text-xs">Демонстрации JSONL</Label>
        <Input id="vla-dataset" placeholder="datasets/actions/train.jsonl" value={config.dataset_path} onChange={event => patch({ dataset_path: event.target.value })} />
        <Label htmlFor="vla-validation" className="text-xs">Holdout JSONL (опционально)</Label>
        <Input id="vla-validation" placeholder="datasets/actions/validation.jsonl" value={config.val_data_path} onChange={event => patch({ val_data_path: event.target.value })} /></div>
      <div className="space-y-2"><Label className="text-xs">Тип действий</Label>
        <Select aria-label="VLA action type" value={config.action_type} onChange={event => patch({ action_type: event.target.value,
          discrete_classes: event.target.value === 'hybrid' ? [4] : [], discrete_names: [], discrete_labels: [], action_names: [] })}
          options={[{ value: 'continuous', label: 'Непрерывные векторы' }, { value: 'discrete', label: 'Дискретные команды' }, { value: 'hybrid', label: 'Смешанные действия' }]} />
        {numberField('action_dim', config.action_type === 'discrete' ? 'Число классов' : 'Размер непрерывного вектора', config.action_type === 'discrete' ? 2 : 1)}
        {config.action_type === 'hybrid' && <><Label htmlFor="vla-classes" className="text-xs">Классов в каждой категории (через запятую)</Label>
          <Input id="vla-classes" value={config.discrete_classes.join(',')} onChange={event => patch({ discrete_classes: event.target.value.split(',').map(Number), discrete_names: [], discrete_labels: [] })} /></>}
        {numberField('action_horizon', 'Горизонт действий', 1)}{numberField('state_dim', 'Размер состояния (0 — без состояния)', 0)}
        <Label htmlFor="vla-action-names" className="text-xs">Имена координат / классов (через запятую)</Label>
        <Input id="vla-action-names" value={config.action_names.join(',')} onChange={event => patch({ action_names: event.target.value ? event.target.value.split(',').map(value => value.trim()) : [] })} />
        <p className="text-[11px] text-muted-foreground">{config.action_names.join(', ') || 'Семантика координат/классов определяется вашим датасетом.'}</p>
      </div>
      <div className="space-y-2">{numberField('batch_size', 'Batch size', 1)}{numberField('gradient_accumulation', 'Gradient accumulation', 1)}
        {numberField('learning_rate', 'Learning rate', 0.000001, 0.00001)}{numberField('max_steps', 'Optimizer steps', 1)}
        {numberField('save_every', 'Checkpoint every', 1)}{numberField('eval_every', 'Validation every', 1)}
        <Label className="text-xs">Устройство</Label><Select aria-label="VLA device" value={config.device} onChange={event => patch({ device: event.target.value })}
          options={[{ value: 'auto', label: 'Auto' }, { value: 'cuda', label: 'CUDA' }, { value: 'cpu', label: 'CPU (Native VLA)', disabled: config.architecture === 'vlm_action' }]} />
        <Select aria-label="VLA GPU" value={String(gpuId)} onChange={event => setGpuId(Number(event.target.value))}
          options={gpu?.gpus.length ? gpu.gpus.map(value => ({ value: String(value.id), label: `GPU ${value.id}` })) : [{ value: '0', label: 'GPU 0' }]} />
        <Select aria-label="VLA precision" value={config.mixed_precision} onChange={event => patch({ mixed_precision: event.target.value })}
          options={[{ value: 'bf16', label: 'BF16' }, { value: 'fp16', label: 'FP16' }, { value: 'no', label: 'Float32' }]} />
      </div>
    </aside>
    <main className="flex-1 overflow-auto p-6 space-y-4">
      <div><h2 className="text-xl font-semibold">VLA Studio</h2><p className="text-sm text-muted-foreground">Обучение action-моделей по демонстрациям — behavior cloning.</p></div>
      <Tabs value={tab} onValueChange={setTab}><TabsList><TabsTrigger value="launch">Обучение</TabsTrigger><TabsTrigger value="predict">Action preview</TabsTrigger><TabsTrigger value="history">История</TabsTrigger><TabsTrigger value="data">Формат данных</TabsTrigger></TabsList>
        <TabsContent value="launch"><div className="space-y-4">
          <Card><CardHeader><CardTitle>Action policy training</CardTitle><CardDescription>Native VLA учится с нуля; VLM + Action Head использует визуально-языковую основу и отдельный выход действий.</CardDescription></CardHeader><CardContent className="space-y-3">
            <div className="flex flex-wrap gap-2"><Button variant="outline" onClick={check} disabled={checking || !!active || !config.dataset_path}>{checking ? <Loader2 className="h-4 w-4 animate-spin" /> : <CheckCircle2 className="h-4 w-4" />}Проверить VLA</Button>
              <Button onClick={start} disabled={!ready || busy || !!active}><Play className="h-4 w-4" />Начать VLA обучение</Button>
              {active && <Button variant="destructive" onClick={stop} disabled={busy}><Square className="h-4 w-4" />Остановить</Button>}</div>
            {report?.key === key && <div className="space-y-2"><Badge variant={ready ? 'success' : 'destructive'}>{ready ? 'VLA проверка пройдена' : 'Исправьте настройки или данные'}</Badge>
              {report.value.checks.map((item, index) => <p key={index} className={`text-xs ${item.status === 'error' ? 'text-destructive' : item.status === 'warning' ? 'text-amber-500' : 'text-muted-foreground'}`}><strong>{item.name}</strong>: {item.detail}</p>)}
              {sample && <div className="grid grid-cols-[160px_1fr] gap-3 rounded-lg bg-muted/40 p-3"><img src={sample.image} alt="VLA observation" className="w-40 rounded" /><div className="space-y-2 text-xs"><p>{sample.instruction}</p><p>State: {JSON.stringify(sample.state)}</p><p>Действий в примере: {sample.valid_timesteps} / {sample.horizon}</p><pre className="max-h-36 overflow-auto">{JSON.stringify(sample.actions, null, 2)}</pre></div></div>}
            </div>}
          </CardContent></Card>
          {selected && <Card><CardHeader><CardTitle>Эксперимент {selected}</CardTitle></CardHeader><CardContent className="space-y-2 text-sm">
            <p>{current?.status ?? 'initializing'} · Step {current?.step ?? 0} · Loss {current?.loss?.toFixed(6) ?? '—'} · Val loss {current?.val_loss?.toFixed(6) ?? '—'}</p>
            <div className="h-1.5 rounded bg-muted overflow-hidden"><div className="h-full bg-primary" style={{ width: `${Math.min(100, 100 * (current?.step ?? 0) / Math.max(1, selectedRun?.total_steps ?? config.max_steps))}%` }} /></div>
            {(metrics.data?.history.length ?? 0) > 0 && <div className="h-52 w-full"><ResponsiveContainer width="100%" height="100%"><LineChart data={metrics.data?.history}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.15} /><XAxis dataKey="step" /><YAxis width={45} /><Tooltip />
              <Line dataKey="loss" name="Action loss" stroke="var(--color-primary, #8b5cf6)" dot={false} isAnimationActive={false} />
              <Line dataKey="val_loss" name="Validation loss" stroke="#22c55e" connectNulls dot={false} isAnimationActive={false} />
            </LineChart></ResponsiveContainer></div>}
            {current?.action_mae != null && <p className="text-xs">Action MAE (исходные единицы): {current.action_mae.toFixed(6)}</p>}
            {current?.action_accuracy != null && <p className="text-xs">Discrete accuracy: {(100 * current.action_accuracy).toFixed(1)}%</p>}
            {current?.error && <pre role="alert" className="whitespace-pre-wrap text-xs text-destructive">{current.error}</pre>}
            {exportPath && <p className="break-all text-xs">Policy export: {exportPath}</p>}
            <details><summary className="cursor-pointer">Логи</summary><pre className="max-h-64 overflow-auto whitespace-pre-wrap text-xs">{logs.data?.stderr || logs.data?.stdout || 'Ожидание логов…'}</pre></details>
          </CardContent></Card>}
        </div></TabsContent>
        <TabsContent value="predict"><ActionPreview initialPath={exportPath} gpuId={gpuId} stateDim={selectedRun?.config.state_dim ?? config.state_dim} /></TabsContent>
        <TabsContent value="history"><div className="space-y-3">
          {!runs.data?.runs.length && <p className="text-sm text-muted-foreground">VLA экспериментов пока нет.</p>}
          {runs.data?.runs.map(run => <Card key={run.run_id}><CardContent className="p-4 flex flex-wrap items-center gap-3"><div className="flex-1 text-sm"><strong>{run.experiment_name || run.run_id}</strong><p className="text-xs text-muted-foreground">{run.status} · {run.current_step} / {run.total_steps} · {String(run.config.domain ?? '')} · {String(run.config.action_type ?? '')}</p></div>
            <Button variant="outline" onClick={() => { setObserved(run.run_id); setTab('launch') }}>Метрики</Button>
            {!run.process_running && !!run.checkpoints?.length && <Button variant="outline" disabled={busy} onClick={async () => {
              setBusy(true); setError(null)
              try { const result = await api.continueTraining(run.run_id, { max_steps: Math.max(config.max_steps, run.current_step + 100) }); setObserved(result.run_id); setTab('launch'); await runs.refetch() }
              catch (value) { fail(value) } finally { setBusy(false) }
            }}>Продолжить до {Math.max(config.max_steps, run.current_step + 100)}</Button>}
          </CardContent></Card>)}
        </div></TabsContent>
        <TabsContent value="data"><Card><CardHeader><CardTitle>Демонстрации JSONL</CardTitle><CardDescription>Каждая строка — наблюдение, инструкция, состояние и явный action chunk. Пути изображений считаются от JSONL. Короткие конечные chunks маскируются; границы эпизодов не склеиваются.</CardDescription></CardHeader><CardContent className="space-y-4 text-sm">
          <p>Робототехника: вектор перемещений/суставов и gripper. Симуляторы: аналоговые оси и классы команд. Desktop: координаты/дельты, scroll, классы кнопок и клавиш. Единицы и таблица классов задаются вашим контрактом данных; пресеты можно менять.</p>
          <pre className="overflow-auto rounded bg-muted p-3 text-xs">{JSON.stringify({ image: 'frames/0001.png', instruction: 'move to target', state: config.state_dim ? Array(config.state_dim).fill(0) : [],
            actions: [config.action_type === 'continuous' ? Array(config.action_dim).fill(0) : config.action_type === 'discrete' ? 0 : { continuous: Array(config.action_dim).fill(0), discrete: config.discrete_classes.map(() => 0) }] }, null, 2)}</pre>
          {config.discrete_labels.length > 0 && <pre className="overflow-auto rounded bg-muted p-3 text-xs">{JSON.stringify(Object.fromEntries(config.discrete_names.map((name, index) => [name, config.discrete_labels[index].map((value, id) => `${id}: ${value}`)])), null, 2)}</pre>}
          <p>Нормализация считается только по train, сохраняется с моделью и применяется при action preview. Для оценки обобщения нужен отдельный holdout по эпизодам/сессиям.</p>
        </CardContent></Card></TabsContent>
      </Tabs>
      {error && <pre role="alert" className="whitespace-pre-wrap text-sm text-destructive">{error}</pre>}
    </main>
  </div>
}

function ActionPreview({ initialPath, gpuId, stateDim }: { initialPath: string; gpuId: number; stateDim: number }) {
  const uploadVersion = useRef(0)
  const [path, setPath] = useState(initialPath)
  const [instruction, setInstruction] = useState('move to target')
  const [state, setState] = useState(() => JSON.stringify(Array(stateDim).fill(0)))
  const [image, setImage] = useState('')
  const [result, setResult] = useState<VLAPrediction | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => { if (initialPath) setPath(initialPath) }, [initialPath])
  async function predict() {
    setBusy(true); setError(null); setResult(null)
    try {
      const values: unknown = JSON.parse(state)
      if (!Array.isArray(values) || values.some(value => typeof value !== 'number' || !Number.isFinite(value))) throw new Error('State должен быть JSON-массивом чисел')
      setResult(await api.predictVLA({ model_path: path, instruction, state: values, image_base64: image, gpu_ids: [gpuId] }))
    } catch (value) { setError(value instanceof Error ? value.message : String(value)) } finally { setBusy(false) }
  }
  return <Card><CardHeader><CardTitle>Action preview</CardTitle><CardDescription>Предсказание action chunk из экспорта. Здесь возвращаются данные для проверки политики.</CardDescription></CardHeader><CardContent className="space-y-3">
    <Label htmlFor="vla-policy">Policy export</Label><Input id="vla-policy" placeholder="out/vla_.../final_model" disabled={busy} value={path} onChange={event => { setPath(event.target.value); setResult(null) }} />
    <Label htmlFor="vla-instruction">Инструкция</Label><Input id="vla-instruction" disabled={busy} value={instruction} onChange={event => { setInstruction(event.target.value); setResult(null) }} />
    <Label htmlFor="vla-state">State JSON</Label><Input id="vla-state" disabled={busy} value={state} onChange={event => { setState(event.target.value); setResult(null) }} />
    <Label htmlFor="vla-image" className="flex items-center gap-2"><Upload className="h-4 w-4" />Изображение / скриншот</Label>
    <Input id="vla-image" disabled={busy} type="file" accept="image/*" onChange={event => {
      const file = event.target.files?.[0]
      if (!file) return
      const requestVersion = ++uploadVersion.current
      if (file.size > 14_000_000) { setError('Изображение превышает 14 MB'); setImage(''); return }
      const reader = new FileReader()
      reader.onload = () => { if (requestVersion === uploadVersion.current) { setImage(String(reader.result)); setResult(null) } }
      reader.onerror = () => { if (requestVersion === uploadVersion.current) setError('Не удалось прочитать изображение') }
      reader.readAsDataURL(file)
    }} />
    {image && <img src={image} alt="Action observation" className="max-h-56 rounded object-contain" />}
    <Button onClick={predict} disabled={busy || !path || !image || !instruction.trim()}>{busy && <Loader2 className="h-4 w-4 animate-spin" />}Предсказать действия</Button>
    {result && <div><p className="text-xs text-muted-foreground">{result.action_type} · Horizon {result.action_horizon} · {result.action_names?.join(', ')}</p><pre className="max-h-96 overflow-auto rounded bg-muted p-3 text-xs">{JSON.stringify(result.actions, null, 2)}</pre>{result.discrete_labels && <p className="text-xs text-muted-foreground">{JSON.stringify(result.discrete_labels)}</p>}</div>}
    {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
  </CardContent></Card>
}
