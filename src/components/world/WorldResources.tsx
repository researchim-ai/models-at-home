import { useEffect, useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/api/client'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

const titles={unsupported:'Режим не подключён',blocked:'Не помещается / запуск заблокирован',unknown:'Недостаточно данных о машине',risk:'Небольшой запас — нужна проба',likely_fit:'Предварительно помещается'}
function memory(n:number|null|undefined){return n==null?'неизвестно':`${n.toFixed(2)} GiB`}
function range(values:[number|null,number|null]){return values[0]==null?'не измерено':`${memory(values[0])} – ${memory(values[1])}`}
export function durationLabel(n:number){return n<60?`${Math.ceil(n)} с`:n<3600?`${Math.ceil(n/60)} мин`:`${(n/3600).toFixed(1)} ч`}

export function WorldResources({modelId,operation,config,onBlockedChange,active=false}:{modelId:string;operation:'train'|'inference';config:Record<string,unknown>;onBlockedChange?:(value:boolean)=>void;active?:boolean}){
  const payload={model_id:modelId,operation,config}
  const raw=JSON.stringify(payload)
  const [key,setKey]=useState(raw)
  useEffect(()=>{const t=setTimeout(()=>setKey(raw),300);return()=>clearTimeout(t)},[raw])
  const queryKey=['world','resources',key]
  const qc=useQueryClient()
  const query=useQuery({queryKey,queryFn:()=>api.worldResources(JSON.parse(key)),refetchInterval:10000,staleTime:3000,retry:1})
  const [running,setRunning]=useState(false),[error,setError]=useState('')
  const latest=useRef(raw);latest.current=raw
  const alive=useRef(true);useEffect(()=>{alive.current=true;return()=>{alive.current=false}},[])
  const report=key===raw?query.data:undefined
  const blocked=running||!!report?.benchmark_running||(!active&&report?.status==='blocked')
  useEffect(()=>{onBlockedChange?.(blocked)},[blocked,onBlockedChange])
  async function benchmark(){
    setRunning(true);setError('');const started=raw
    try{const result=await api.benchmarkWorldResources(payload);qc.setQueryData(['world','resources',started],result)}
    catch(e){if(alive.current&&latest.current===started)setError(e instanceof Error?e.message:String(e))}
    finally{if(alive.current)setRunning(false)}
  }
  return <Card><CardHeader className="pb-3"><CardTitle>Ресурсы и время · {operation==='train'?'обучение':'inference'}</CardTitle></CardHeader><CardContent className="space-y-3 text-xs">
    {!report&&<p className="text-muted-foreground">{query.isError?'Не удалось оценить ресурсы; проверьте подключение к backend.':'Проверка доступных мощностей…'}</p>}
    {report&&<>
      <p className={report.status==='blocked'&&!active?'text-destructive font-semibold':'font-semibold'}>{active?'Модель уже запущена':titles[report.status]}</p>
      <p className="text-muted-foreground">{report.basis}. Другие процессы могут изменить свободную память после проверки.</p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        <p>{report.device==='cuda'&&report.gpu?`GPU ${report.gpu.id}: ${report.gpu.name}`:`Устройство: ${report.device}`}{report.device==='cuda'&&<><br/>Свободно VRAM: {memory(report.gpu?.free_gib)} / {memory(report.gpu?.total_gib)}</>}</p>
        <p>RAM свободно: {memory(report.hardware.ram_available_gib)} / {memory(report.hardware.ram_total_gib)}{report.hardware.cgroup_limited?' (лимит контейнера учтён)':''}<br/>Torch: до {report.hardware.cpu_threads} потоков CPU</p>
        <p>Память модели и активаций{report.device==='cuda'?' на GPU':''}: {range(report.memory_gib)}</p>
        <p>RAM процесса: {range(report.ram_gib)}<br/>Диск для {operation==='train'?'всех checkpoints':'weights/cache'}: {memory(report.disk_required_gib)} · доступно {memory(report.disk_available_gib)}</p>
      </div>
      {!report.training_supported&&<p>Дообучение этой pretrained-модели пока не подключено. Время тюнинга не рассчитывается. Full/AdamW: минимум {memory(report.training_state_lower_bound_gib)} только для FP32 весов, градиентов и optimizer states; активации и AMP-копия дополнительно.</p>}
      {report.measurement?.status==='ok'&&<p>Проба: {report.measurement.device}{report.measurement.gpu_name?` · ${report.measurement.gpu_name}`:''}; {report.measurement.warmup_steps} прогревочных + {report.measurement.measured_steps} измеренных optimizer steps.<br/>Шаг: {report.measurement.train_step_seconds?.toFixed(3)} с{report.measurement.device?.startsWith('cuda')&&<> · GPU tensors peak: {memory(report.measurement.peak_allocated_gib)} · GPU reserved peak: {memory(report.measurement.peak_reserved_gib)}</>} · RAM peak: {memory(report.measurement.peak_rss_gib)}</p>}
      {report.measurement?.error&&<p role="alert" className="text-destructive">Проба: {report.measurement.error}</p>}
      {report.eta?<div className="rounded bg-muted p-3 space-y-1"><p className="font-semibold">Примерное время на {report.eta.remaining_steps} шагов: {durationLabel(report.eta.total_seconds_low)}{report.eta.total_seconds_high==null?' + validation':` – ${durationLabel(report.eta.total_seconds_high)}`}</p><p>{report.eta.basis}</p><p>{report.eta.note}</p></div>:operation==='train'&&report.benchmark_supported&&<p>Время ещё не измерено. Короткая проба даст оценку под выбранные batch, окно, кадры и precision на этой машине.</p>}
      {[...report.reasons,...report.warnings].map((text,i)=><p key={i} className={report.status==='blocked'?'text-destructive':'text-amber-500'}>{text}</p>)}
      <details><summary className="cursor-pointer">Подсказки и границы оценки</summary><ul className="list-disc pl-5 mt-2 space-y-1">{report.hints.map((hint,i)=><li key={i}>{hint}</li>)}</ul></details>
      {report.benchmark_supported&&<div className="space-y-1"><Button variant="outline" disabled={running||report.benchmark_running||key!==raw} onClick={benchmark}>{running?'Измерение ресурсов…':'Измерить 5 шагов'}</Button><p className="text-muted-foreground">Реальные forward/backward + AdamW, validation и checkpoint на временных синтетических данных. Пользовательские веса и запуски не изменяются.</p></div>}
    </>}
    {error&&<p role="alert" className="text-destructive whitespace-pre-wrap">{error}</p>}
  </CardContent></Card>
}
