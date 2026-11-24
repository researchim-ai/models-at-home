import { useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { WorldResources } from '@/components/world/WorldResources'
import { api } from '@/api/client'
import { useGpuInfo, useRunLogs } from '@/api/hooks'
import { LocalWorldPlay } from '@/pages/LocalWorldPlay'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select } from '@/components/ui/select'

export function WorldFoundationModels({gpuId,onGpuChange}:{gpuId:number;onGpuChange:(id:number)=>void}){
  const catalog=useQuery({queryKey:['world','foundation','models'],queryFn:api.worldFoundationModels})
  const {data:gpu}=useGpuInfo()
  const [model,setModel]=useState('diamond_csgo')
  const info=catalog.data?.models.find(m=>m.id===model)
  return <div className="space-y-4">
    <Card><CardHeader><CardTitle>Встроенные локальные world models</CardTitle></CardHeader><CardContent className="space-y-3">
      <p className="text-sm text-muted-foreground">Скачайте pretrained-веса и пробуйте модель прямо здесь. DIAMOND работает встроенным кодом Studio, без облачного inference и отдельного приложения.</p>
      {catalog.isError&&<p role="alert" className="text-destructive">Не удалось получить каталог.</p>}
      <div className="grid gap-3 sm:grid-cols-[1fr_150px]"><Label className="text-xs">Модель<Select aria-label="Foundation model" value={model} options={(catalog.data?.models??[]).map(m=>({value:m.id,label:m.name}))} onChange={e=>setModel(e.target.value)} /></Label><Label className="text-xs">GPU<Select aria-label="Foundation GPU" value={String(gpuId)} onChange={e=>onGpuChange(Number(e.target.value))} options={gpu?.gpus.length?gpu.gpus.map(g=>({value:String(g.id),label:`GPU ${g.id}`})):[{value:'0',label:'GPU 0'}]} /></Label></div>
      {info&&<div className="space-y-1 text-xs text-muted-foreground"><p>{info.description}</p><p>{info.requirements}</p><p>Лицензия: {info.license}</p><a href={info.source} target="_blank" rel="noreferrer" className="text-primary underline">Источник модели</a></div>}
    </CardContent></Card>
    {model.startsWith('diamond')?<LocalWorldPlay key={`${model}:${gpuId}`} modelId={model} gpuId={gpuId}/>:<CosmosLocal key={model} modelId={model} gpuId={gpuId} domains={catalog.data?.domains??{}}/>}
  </div>
}

type Report=Awaited<ReturnType<typeof api.preflightFoundation>>
function CosmosLocal({modelId,gpuId,domains}:{modelId:string;gpuId:number;domains:Record<string,number>}){
  const [resourceBlocked,setResourceBlocked]=useState(false)
  const [c,setC]=useState({task:'text2video',prompt:'',negative_prompt:'',model_path:'',num_frames:49,height:256,width:448,inference_steps:20,guidance_scale:6,cpu_offload:true,domain:'umi'})
  const [image,setImage]=useState(''),[actions,setActions]=useState('[]'),[busy,setBusy]=useState(false),[error,setError]=useState(''),[job,setJob]=useState<string|null>(null)
  const [report,setReport]=useState<{key:string;value:Report}|null>(null),[video,setVideo]=useState('')
  const payload=()=>({...c,model_id:modelId,image_base64:image,actions:c.task==='forward_dynamics'?JSON.parse(actions):[],gpu_ids:[gpuId]})
  const key=JSON.stringify({c,image,actions,gpuId,modelId}),latest=useRef(key);latest.current=key
  const alive=useRef(true);useEffect(()=>{alive.current=true;return()=>{alive.current=false}},[])
  const patch=(p:Partial<typeof c>)=>setC(old=>({...old,...p}))
  const metrics=useQuery({queryKey:['world','foundation','metrics',job],queryFn:()=>api.runMetrics(job!),enabled:!!job,refetchInterval:1500})
  const logs=useRunLogs(job)
  const result=useQuery({queryKey:['world','foundation','result',job],queryFn:()=>api.foundationResult(job!),enabled:!!job&&metrics.data?.metrics?.status==='completed'})
  useEffect(()=>{let live=true,url='';setVideo('');if(job&&result.data?.video_path)api.foundationVideo(job).then(blob=>{if(live){url=URL.createObjectURL(blob);setVideo(url)}}).catch(e=>{if(live)setError(String(e))});return()=>{live=false;if(url)URL.revokeObjectURL(url)}},[job,result.data?.video_path])
  const active=!!job&&!['completed','error','stopped'].includes(metrics.data?.metrics?.status??'initializing')
  async function check(){setBusy(true);setError('');try{const value=await api.preflightFoundation(payload());if(alive.current&&latest.current===key)setReport({key,value})}catch(e){if(alive.current)setError(String(e))}finally{if(alive.current)setBusy(false)}}
  async function launch(download=false){setBusy(true);setError('');try{const r=await(download?api.downloadFoundation({model_id:modelId,task:'text2video'}):api.generateFoundation(payload()));if(alive.current)setJob(r.run_id)}catch(e){if(alive.current)setError(String(e))}finally{if(alive.current)setBusy(false)}}
  return <div className="grid gap-4 xl:grid-cols-2"><Card><CardHeader><CardTitle>Cosmos · локальная генерация</CardTitle></CardHeader><CardContent className="space-y-3">
    <p className="text-xs text-muted-foreground">Дополнительный крупный локальный pipeline. Для обязательных guardrail-весов требуется доступ Hugging Face. Все вычисления выполняются локально.</p>
    <Label className="block text-xs">Задача<Select disabled={busy||active} value={c.task} onChange={e=>patch({task:e.target.value,height:256,guidance_scale:e.target.value==='forward_dynamics'?1:6})} options={[{value:'text2video',label:'Текст → видео'},{value:'image2video',label:'Кадр → видео'},{value:'forward_dynamics',label:'Кадр + действия → будущее'}]}/></Label>
    <Label className="block space-y-1 text-xs">Описание мира<textarea aria-label="World prompt" disabled={busy||active} value={c.prompt} onChange={e=>patch({prompt:e.target.value})} className="w-full min-h-24 bg-background rounded border border-input p-3"/></Label>
    <Label className="block text-xs">Локальная папка snapshot<Input value={c.model_path} disabled={busy||active} onChange={e=>patch({model_path:e.target.value})}/></Label>
    {c.task!=='text2video'&&<Label className="block text-xs">Начальный кадр<Input type="file" accept="image/*" disabled={busy||active} onChange={e=>{const f=e.target.files?.[0];if(!f)return;if(f.size>14_000_000){setError('Максимум 14 MB');return}const r=new FileReader();r.onload=()=>{if(alive.current)setImage(String(r.result))};r.readAsDataURL(f)}}/></Label>}
    {c.task==='forward_dynamics'&&<><Label className="block text-xs">Embodiment<Select disabled={busy||active} value={c.domain} options={Object.entries(domains).map(([value,n])=>({value,label:`${value} · ${n}D`}))} onChange={e=>patch({domain:e.target.value})}/></Label><textarea aria-label="Cosmos raw actions" value={actions} disabled={busy||active} onChange={e=>setActions(e.target.value)} className="w-full min-h-24 bg-background border border-input rounded p-3 font-mono"/><p className="text-xs">Raw actions [T,D] по контракту Cosmos; T кратно 4. Действия VLA не конвертируются автоматически.</p></>}
    <div className="grid grid-cols-2 gap-3">{(['num_frames','inference_steps','width'] as const).map(field=><Label className="text-xs" key={field}>{field}<Input type="number" disabled={busy||active} value={c[field]} onChange={e=>patch({[field]:Number(e.target.value)})}/></Label>)}<Label className="text-xs">Высота<Select disabled={busy||active} value={String(c.height)} options={(c.task==='forward_dynamics'?[256,480]:[256,480,720]).map(n=>({value:String(n),label:String(n)}))} onChange={e=>patch({height:Number(e.target.value)})}/></Label></div>
    <Label className="flex gap-2 text-xs"><input type="checkbox" checked={c.cpu_offload} disabled={busy||active} onChange={e=>patch({cpu_offload:e.target.checked})}/>CPU offload</Label>
  </CardContent></Card><Card><CardHeader><CardTitle>Результат</CardTitle></CardHeader><CardContent className="space-y-3">
    <WorldResources modelId={modelId} operation="inference" config={{...c,gpu_ids:[gpuId]}} onBlockedChange={setResourceBlocked}/>
    <div className="flex flex-wrap gap-2"><Button variant="outline" disabled={busy||active} onClick={()=>launch(true)}>Скачать pretrained-веса</Button><Button variant="outline" disabled={busy||active} onClick={check}>Проверить Cosmos</Button><Button disabled={busy||active||resourceBlocked||report?.key!==key||!report.value.ready} onClick={()=>launch()}>Сгенерировать видео</Button><Button variant="outline" disabled={!active||busy} onClick={()=>{if(job)void api.stopFoundation(job).catch(e=>setError(String(e)))}}>Остановить</Button></div>
    {error&&<p role="alert" className="text-destructive whitespace-pre-wrap">{error}</p>}
    {report?.key===key&&<div className="text-xs space-y-2"><p>Cosmos preflight: {report.value.ready?'готово':'ошибка'}</p>{report.value.checks.map((r,i)=><p key={i}>{r.name}: {r.detail}</p>)}</div>}
    {job&&<><p className="text-xs">{metrics.data?.metrics?.status} · Шаг {metrics.data?.metrics?.step??0}</p>{metrics.data?.metrics?.error&&<p className="text-destructive text-xs whitespace-pre-wrap">{metrics.data.metrics.error}</p>}<pre className="max-h-52 overflow-auto text-xs bg-muted p-3 rounded whitespace-pre-wrap">{logs.data?.stderr||logs.data?.stdout}</pre></>}
    {result.data?.model_path&&<Button variant="outline" onClick={()=>patch({model_path:result.data!.model_path!})}>Использовать скачанный snapshot</Button>}
    {video&&<video src={video} controls className="w-full rounded"/>}
    {result.data?.previews&&<div className="grid grid-cols-3 gap-2">{result.data.previews.map((src,i)=><img key={i} src={src} alt={`Cosmos preview ${i+1}`} className="rounded"/>)}</div>}
  </CardContent></Card></div>
}
