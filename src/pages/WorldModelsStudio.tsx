import { useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Activity, Play, Square } from 'lucide-react'
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts'
import { WorldFoundationModels } from '@/pages/WorldFoundationModels'
import { WorldResources, durationLabel } from '@/components/world/WorldResources'
import { api } from '@/api/client'
import { useGpuInfo, useRunLogs } from '@/api/hooks'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select } from '@/components/ui/select'
import { Label } from '@/components/ui/label'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'

const defaults = { architecture: 'latent_gru', dataset_path: '', val_data_path: '', action_type: 'continuous', action_dim: 2,
  discrete_classes: [] as number[], state_dim: 0, image_size: 32, sequence_length: 4, latent_dim: 32, hidden_dim: 128,
  batch_size: 8, learning_rate: .001, max_steps: 500, save_every: 100, eval_every: 20, kl_weight: .001, free_nats: 1,
  device: 'auto', mixed_precision: 'no' }
type Report = Awaited<ReturnType<typeof api.preflightWorld>>
type Trajectory = Awaited<ReturnType<typeof api.rolloutWorld>>

export function WorldModelsStudio() {
  const [config, setConfig] = useState(defaults)
  const [gpuId, setGpuId] = useState(0)
  const [selected, setSelected] = useState<string | null>(null)
  const [tab, setTab] = useState('foundation')
  const [resourceBlocked, setResourceBlocked] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [report, setReport] = useState<{ key: string; value: Report } | null>(null)
  const [modelPath, setModelPath] = useState('')
  const [image, setImage] = useState('')
  const [actions, setActions] = useState('[[2,0],[2,0],[0,2],[0,2]]')
  const [state, setState] = useState('[]')
  const [trajectory, setTrajectory] = useState<Trajectory | null>(null)
  const { data: gpu } = useGpuInfo()
  const runs = useQuery({ queryKey: ['world', 'runs'], queryFn: api.worldRuns, refetchInterval: 2500 })
  const demos = useQuery({ queryKey: ['world', 'examples'], queryFn: api.worldExamples })
  const active = runs.data?.runs.find(r => r.process_running)
  useEffect(() => { if (!selected && active) setSelected(active.run_id) }, [active, selected])
  const metrics = useQuery({ queryKey: ['world', 'metrics', selected], queryFn: () => api.runMetrics(selected!), enabled: !!selected, refetchInterval: 2000 })
  const logs = useRunLogs(selected)
  const payload = { ...config, gpu_ids: [gpuId] }
  const key = JSON.stringify(payload)
  const currentKey = useRef(key); currentKey.current = key
  const requestVersion = useRef(0)
  const imageVersion = useRef(0)
  useEffect(() => { requestVersion.current++; setReport(null) }, [key])
  useEffect(() => () => { requestVersion.current++; imageVersion.current++ }, [])
  const ready = report?.key === key && report.value.ready
  const patch = (value: Partial<typeof config>) => setConfig(c => ({ ...c, ...value }))
  const fail = (e: unknown) => setError(e instanceof Error ? e.message : String(e))
  async function check() {
    const version = ++requestVersion.current
    setBusy(true); setError('')
    try { const value = await api.preflightWorld(payload); if (currentKey.current === key && version === requestVersion.current) setReport({ key, value }) }
    catch (e) { fail(e) } finally { setBusy(false) }
  }
  async function start() {
    if (!ready || active || busy || resourceBlocked) return
    setBusy(true); setError('')
    try { const r = await api.startWorld(payload); setSelected(r.run_id); await runs.refetch() } catch (e) { fail(e) } finally { setBusy(false) }
  }
  async function stop() {
    if (!active) return
    setBusy(true)
    try { await api.stopWorld(active.run_id); await runs.refetch() } catch (e) { fail(e) } finally { setBusy(false) }
  }
  async function resume(id: string, steps: number) {
    setBusy(true); setError('')
    try { const r = await api.continueTraining(id, { max_steps: steps + config.max_steps }); setSelected(r.run_id); setTab('train'); await runs.refetch() }
    catch (e) { fail(e) } finally { setBusy(false) }
  }
  async function imagine() {
    setBusy(true); setError(''); setTrajectory(null)
    try {
      const result = await api.rolloutWorld({ model_path: modelPath, image_base64: image, actions: JSON.parse(actions), state: JSON.parse(state), device: config.device, gpu_ids: [gpuId] })
      setTrajectory(result)
    } catch (e) { fail(e) } finally { setBusy(false) }
  }
  const numeric = (field: keyof typeof config, label: string, min: number, max: number, step = 1) => <div key={field} className="space-y-1">
    <Label htmlFor={`world-${field}`} className="text-xs">{label}</Label><Input id={`world-${field}`} type="number" min={min} max={max} step={step} value={String(config[field])}
      onChange={e => patch({ [field]: Math.min(max, Math.max(min, Number(e.target.value) || min)) })} /></div>
  const selectedRun = runs.data?.runs.find(r => r.run_id === selected)
  const current = metrics.data?.metrics
  const exportPath = current?.status === 'completed' && selectedRun?.process_running !== true ? current.final_model_dir ?? selectedRun?.final_model_path : null
  const field = (label: string, value: string, change: (value: string) => void, id?: string) => <Label className="block space-y-1 text-xs">{label}<Input id={id} disabled={busy && tab === 'rollout'} value={value} onChange={e => change(e.target.value)} /></Label>
  return <div className="flex h-full">
    {tab !== 'foundation' && <aside className="w-80 shrink-0 border-r border-border overflow-auto p-4 space-y-4">
      <h2 className="font-semibold flex items-center gap-2"><Activity className="h-4 w-4" />World Models</h2>
      <p className="text-xs text-muted-foreground">Наблюдение + действия → будущие кадры, состояния, награды и вероятность завершения.</p>
      <Label className="block space-y-1 text-xs">Архитектура<Select value={config.architecture} onChange={e => patch({ architecture: e.target.value })} options={[{ value: 'latent_gru', label: 'Native Latent GRU' }, { value: 'rssm', label: 'Native Gaussian RSSM' }]} /></Label>
      <p className="text-xs text-muted-foreground">Собственный PyTorch trainer. RSSM добавляет стохастическое скрытое состояние и KL-регуляризацию.</p>
      <Button variant="outline" className="w-full" disabled={!demos.data || busy} onClick={() => { if (demos.data) { patch({ ...defaults, ...demos.data, state_dim: 2 }); setState('[10,10]') } }}>Синтетический пример</Button>
      {field('Train JSONL', config.dataset_path, dataset_path => patch({ dataset_path }), 'world-dataset')}
      {field('Validation JSONL', config.val_data_path, val_data_path => patch({ val_data_path }))}
      <Label className="block space-y-1 text-xs">Тип действий<Select value={config.action_type} onChange={e => patch({ action_type: e.target.value, discrete_classes: e.target.value === 'hybrid' ? [4] : [] })} options={['continuous', 'discrete', 'hybrid'].map(value => ({ value, label: value }))} /></Label>
      {numeric('action_dim', config.action_type === 'discrete' ? 'Число классов' : 'Координаты действия', config.action_type === 'discrete' ? 2 : 1, 256)}
      {config.action_type === 'hybrid' && field('Категории по головам, через запятую', config.discrete_classes.join(','), value => patch({ discrete_classes: value.split(',').slice(0, 16).map(v => Math.min(256, Math.max(2, Number(v) || 2))) }))}
      <div className="grid grid-cols-2 gap-3">{numeric('state_dim', 'State dim', 0, 256)}{numeric('sequence_length', 'Переходов в окне', 1, 64)}{numeric('latent_dim', 'Latent dim', 4, 256)}{numeric('hidden_dim', 'Hidden dim', 16, 512)}</div>
      <Label className="block space-y-1 text-xs">Размер кадра<Select value={String(config.image_size)} onChange={e => patch({ image_size: Number(e.target.value) })} options={[32, 64, 128].map(n => ({ value: String(n), label: `${n} × ${n}` }))} /></Label>
      <div className="grid grid-cols-2 gap-3">{numeric('batch_size', 'Batch', 1, 64)}{numeric('max_steps', 'Шагов', 1, 10000000)}{numeric('save_every', 'Checkpoint', 1, 10000000)}{numeric('eval_every', 'Validation', 1, 10000000)}{numeric('learning_rate', 'Learning rate', .000001, .1, .0001)}</div>
      {config.architecture === 'rssm' && <div className="grid grid-cols-2 gap-3">{numeric('kl_weight', 'KL weight', 0, 1, .001)}{numeric('free_nats', 'Free nats', 0, 100, .1)}</div>}
      <Label className="block space-y-1 text-xs">Устройство<Select aria-label="World device" value={config.device} onChange={e => patch({ device: e.target.value })} options={['auto', 'cpu', 'cuda'].map(value => ({ value, label: value }))} /></Label>
      <Label className="block space-y-1 text-xs">GPU<Select value={String(gpuId)} onChange={e => setGpuId(Number(e.target.value))} options={gpu?.gpus?.length ? gpu.gpus.map(g => ({ value: String(g.id), label: `${g.id}: GPU` })) : [{ value: '0', label: 'GPU 0' }]} /></Label>
      <Label className="block space-y-1 text-xs">Precision<Select value={config.mixed_precision} onChange={e => patch({ mixed_precision: e.target.value })} options={['no', 'bf16', 'fp16'].map(value => ({ value, label: value === 'no' ? 'FP32' : value }))} /></Label>
    </aside>}
    <main className="flex-1 min-w-0 overflow-auto p-6 space-y-5">
      <div><h1 className="text-xl font-semibold">World Models Studio</h1><p className="text-sm text-muted-foreground">Обучение динамики среды и предпросмотр воображаемых траекторий.</p></div>
      {error && <div role="alert" className="text-sm text-destructive whitespace-pre-wrap">{error}</div>}
      <Tabs value={tab} onValueChange={setTab}><TabsList><TabsTrigger value="foundation">Готовые модели</TabsTrigger><TabsTrigger value="train">Обучение</TabsTrigger><TabsTrigger value="rollout">Rollout</TabsTrigger><TabsTrigger value="history">История</TabsTrigger><TabsTrigger value="format">Данные</TabsTrigger></TabsList>
        <TabsContent value="foundation"><WorldFoundationModels gpuId={gpuId} onGpuChange={setGpuId} /></TabsContent>
        <TabsContent value="train" className="space-y-4">
          <WorldResources modelId={config.architecture} operation="train" config={payload} onBlockedChange={setResourceBlocked} />
          <div className="flex gap-2"><Button variant="outline" disabled={busy} onClick={check}>Проверить данные</Button><Button disabled={busy || !ready || !!active || resourceBlocked} onClick={start}><Play className="h-4 w-4 mr-2" />Обучать</Button><Button variant="outline" disabled={busy || !active} onClick={stop}><Square className="h-4 w-4 mr-2" />Остановить</Button></div>
          {busy && <p className="text-sm">Выполняется операция…</p>}
          {report?.key === key && <Card><CardHeader><CardTitle>Preflight: {report.value.ready ? 'готово' : 'ошибка'}</CardTitle></CardHeader><CardContent className="space-y-2">{report.value.checks.map((c, i) => <p key={i} className={c.status === 'error' ? 'text-sm text-destructive' : 'text-sm'}>{c.name}: {c.detail}</p>)}</CardContent></Card>}
          {selected && <Card><CardHeader><CardTitle>{selected}</CardTitle></CardHeader><CardContent className="space-y-3">
            <p className="text-sm">{current?.status ?? selectedRun?.status} · Осталось: {current?.estimated_remaining_seconds != null ? durationLabel(current.estimated_remaining_seconds) : '—'} · Шаг {current?.step ?? 0} / {selectedRun?.total_steps ?? 0} · Loss {current?.loss?.toFixed(5) ?? '—'} · Validation MSE {current?.val_loss?.toFixed(5) ?? '—'}</p>
            {current?.error && <p className="text-destructive whitespace-pre-wrap">{current.error}</p>}
            {!!metrics.data?.history.length && <div className="h-56"><ResponsiveContainer width="100%" height="100%"><LineChart data={metrics.data.history}><XAxis dataKey="step" /><YAxis /><Tooltip contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 8 }} /><Line dataKey="loss" stroke="#38bdf8" dot={false} isAnimationActive={false} /><Line dataKey="val_loss" stroke="#34d399" dot={false} connectNulls isAnimationActive={false} /></LineChart></ResponsiveContainer></div>}
            {exportPath && <div className="space-y-2"><p className="text-xs break-all">Экспорт: {exportPath}</p><Button variant="outline" onClick={() => { setModelPath(exportPath); setState(JSON.stringify(Array(selectedRun?.config.state_dim ?? 0).fill(0))); setTrajectory(null); setTab('rollout') }}>Открыть Rollout</Button></div>}
            <pre className="max-h-64 overflow-auto text-xs bg-muted rounded p-3">{logs.data?.stderr || logs.data?.stdout || 'Ожидание логов…'}</pre>
          </CardContent></Card>}
        </TabsContent>
        <TabsContent value="rollout" className="space-y-4">
          <p className="text-sm text-muted-foreground">Загрузите начальный кадр и задайте последовательность действий (до 64 переходов). Никакие действия не исполняются в среде.</p>
          {field('Папка экспорта модели', modelPath, v => { setModelPath(v); setTrajectory(null) })}
          <Label className="block space-y-1 text-xs">Начальный кадр<Input type="file" accept="image/*" disabled={busy} onChange={e => {
            const file = e.target.files?.[0]; const version = ++imageVersion.current; setImage(''); setTrajectory(null)
            if (!file) return
            if (file.size > 14_000_000) { setError('Кадр превышает 14 MB'); return }
            const reader = new FileReader(); reader.onload = () => { if (imageVersion.current === version) setImage(String(reader.result)) }; reader.onerror = () => setError('Не удалось прочитать изображение'); reader.readAsDataURL(file)
          }} /></Label>
          {image && <img src={image} alt="Начальный кадр" className="w-32 h-32 object-contain bg-muted rounded" />}
          {field('Начальное состояние, JSON', state, v => { setState(v); setTrajectory(null) })}
          <Label className="block space-y-1 text-xs">Действия, JSON<textarea disabled={busy} className="w-full min-h-24 bg-background border border-input p-3 rounded font-mono" value={actions} onChange={e => { setActions(e.target.value); setTrajectory(null) }} /></Label>
          <Button disabled={busy || !modelPath || !image || !!active} onClick={imagine}>Предсказать будущее</Button>
          {trajectory && <div className="space-y-3"><p className="text-xs text-muted-foreground">{trajectory.warning}</p><div className="flex flex-wrap gap-3">{trajectory.frames.map((frame, i) => <figure key={i}><img src={frame} alt={`Предсказание t+${i+1}`} className="w-32 h-32 rounded border border-border" /><figcaption className="text-xs">t+{i+1}</figcaption></figure>)}</div><pre className="bg-muted p-3 rounded overflow-auto text-xs">{JSON.stringify({ states: trajectory.states, rewards: trajectory.rewards, done_probabilities: trajectory.done_probabilities }, null, 2)}</pre></div>}
        </TabsContent>
        <TabsContent value="history" className="space-y-3">{runs.data?.runs.map(r => <Card key={r.run_id}><CardContent className="p-4 flex flex-wrap gap-3 items-center"><span className="text-sm">{r.run_id} · {r.status} · {r.current_step}/{r.total_steps}</span><Button variant="outline" onClick={() => { setSelected(r.run_id); setTab('train') }}>Открыть</Button><Button variant="outline" disabled={busy || !!active || !r.checkpoints.length} onClick={() => resume(r.run_id, r.total_steps)}>Продолжить +{config.max_steps} шагов</Button></CardContent></Card>)}{!runs.data?.runs.length && <p className="text-sm text-muted-foreground">Экспериментов пока нет.</p>}</TabsContent>
        <TabsContent value="format" className="space-y-4 text-sm"><p>Одна строка JSONL — независимый эпизод: T действий и T+1 кадров. Пути кадров считаются относительно JSONL. action[t] переводит frame[t] в frame[t+1].</p><pre className="bg-muted rounded p-3 overflow-auto">{JSON.stringify({ episode_id: 'session-1', frames: ['0.png', '1.png', '2.png'], actions: [[1, 0], [0, 1]], states: [[0, 0], [1, 0], [1, 1]], rewards: [0, 1], dones: [false, true] }, null, 2)}</pre><p>states, rewards, dones необязательны; для states задайте state_dim. Дискретное действие — целое число класса; смешанное — {`{"continuous":[...],"discrete":[...]}`}. Разделяйте train/validation по эпизодам. Окна не пересекают terminal-переходы.</p><p>Демо синтетическое. Эти компактные модели обучаются с нуля; готовые веса Dreamer/Genie и actor-critic сюда не подключены. Качество длинных rollout нужно проверять отдельно.</p></TabsContent>
      </Tabs>
    </main>
  </div>
}
