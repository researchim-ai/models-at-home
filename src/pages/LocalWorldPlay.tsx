import { useEffect, useRef, useState } from 'react'
import { WorldResources } from '@/components/world/WorldResources'
import { api } from '@/api/client'
import type { LocalWorldFrame } from '@/api/types'
import { Button } from '@/components/ui/button'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select } from '@/components/ui/select'

const keyMap: Record<string,string> = { KeyW:'w',KeyA:'a',KeyS:'s',KeyD:'d',Space:'space',ControlLeft:'ctrl',ShiftLeft:'shift',KeyR:'r',Digit1:'1',Digit2:'2',Digit3:'3',ArrowLeft:'look_left',ArrowRight:'look_right',ArrowUp:'look_up',ArrowDown:'look_down' }
export function LocalWorldPlay({ modelId, gpuId }: { modelId: string; gpuId: number }) {
  const [path,setPath]=useState('')
  const [game,setGame]=useState('Breakout')
  const [spawn,setSpawn]=useState(0)
  const [device,setDevice]=useState('auto')
  const [steps,setSteps]=useState(3)
  const [upscale,setUpscale]=useState(true)
  const [image,setImage]=useState('')
  const [frame,setFrame]=useState<LocalWorldFrame|null>(null)
  const [session,setSession]=useState('')
  const [resourceBlocked,setResourceBlocked]=useState(false)
  const [busy,setBusy]=useState(false)
  const [running,setRunning]=useState(false)
  const [error,setError]=useState('')
  const [job,setJob]=useState<string|null>(null)
  const [downloadStatus,setDownloadStatus]=useState('')
  const alive=useRef(true), id=useRef(''), play=useRef(false), pressed=useRef(new Set<string>()), action=useRef(0), version=useRef(0), pending=useRef(false)
  const flight=useRef<Promise<LocalWorldFrame>|null>(null)
  const fileVersion=useRef(0)
  const csgo=modelId==='diamond_csgo'
  function pause() { play.current=false;setRunning(false);pressed.current.clear();action.current=0 }
  useEffect(()=>{
    alive.current=true
    const blur=()=>pause();window.addEventListener('blur',blur)
    return ()=>{alive.current=false;play.current=false;pressed.current.clear();version.current++;window.removeEventListener('blur',blur);if(id.current){void api.stopWorldPlay(id.current).catch(()=>{});id.current=''}}
  },[])
  useEffect(()=>{
    if(!job)return
    let live=true
    const timer=setInterval(async()=>{
      try {
        const data=await api.runMetrics(job)
        if(!live)return
        setDownloadStatus(data.metrics?.status ?? '')
        if(data.metrics?.status==='error'){setError(data.metrics.error??'Ошибка скачивания');setJob(null)}
        if(data.metrics?.status==='completed'){
          const result=await api.foundationResult(job)
          if(live){if(result.model_path)setPath(result.model_path);setJob(null);setDownloadStatus('Веса скачаны — можно запускать локально')}
        }
      }catch(e){if(live){setError(String(e));setJob(null)}}
    },1500)
    return()=>{live=false;clearInterval(timer)}
  },[job])
  async function download(){
    setBusy(true);setError('')
    try{const r=await api.downloadFoundation({model_id:modelId,task:'local_play',game});if(alive.current){setJob(r.run_id);setDownloadStatus('Скачивание pretrained-весов…')}}
    catch(e){if(alive.current)setError(String(e))}finally{if(alive.current)setBusy(false)}
  }
  async function load(){
    if(resourceBlocked)return
    setBusy(true);setError('');const v=++version.current
    try{
      const r=await api.startWorldPlay({model_id:modelId,model_path:path,game,spawn,device,denoising_steps:steps,upsample:upscale,image_base64:image,gpu_ids:[gpuId],seed:42})
      if(!alive.current||v!==version.current){void api.stopWorldPlay(r.session_id);return}
      id.current=r.session_id;setSession(r.session_id);setFrame(r)
    }catch(e){if(alive.current)setError(String(e))}finally{if(alive.current)setBusy(false)}
  }
  async function tick(){
    if(pending.current||!id.current)return
    pending.current=true;const v=version.current
    try{
      const request=api.stepWorldPlay({session_id:id.current,keys:[...pressed.current],mouse_x:0,mouse_y:0,action:action.current});flight.current=request
      const r=await request
      if(alive.current&&v===version.current)setFrame(previous=>({...previous,...r}))
    }catch(e){if(alive.current){pause();setError(String(e))}}
    finally{pending.current=false;flight.current=null}
  }
  useEffect(()=>{
    if(!running)return
    let live=true
    const loop=async()=>{if(!live||!play.current)return;await tick();if(live&&play.current)setTimeout(loop,10)}
    void loop();return()=>{live=false}
  },[running])
  async function close(){
    pause();version.current++;setBusy(true)
    const old=id.current;id.current='';setSession('')
    try{await flight.current?.catch(()=>{});if(old)await api.stopWorldPlay(old)}catch(e){setError(String(e))}finally{setBusy(false)}
  }
  async function reset(){
    pause();version.current++;setBusy(true)
    try{await flight.current?.catch(()=>{});const r=await api.resetWorldPlay({session_id:id.current,spawn:csgo?spawn:null});if(alive.current)setFrame(r)}catch(e){setError(String(e))}finally{if(alive.current)setBusy(false)}
  }
  const control=(key:string,label:string)=><Button key={key} variant="outline" disabled={!session||busy} onPointerDown={e=>{e.currentTarget.setPointerCapture?.(e.pointerId);pressed.current.add(key)}} onPointerUp={()=>pressed.current.delete(key)} onPointerCancel={()=>pressed.current.delete(key)}>{label}</Button>
  return <div className="grid gap-4 xl:grid-cols-[minmax(260px,360px)_minmax(0,1fr)]">
    <Card><CardHeader><CardTitle>Локальный запуск</CardTitle></CardHeader><CardContent className="space-y-4">
      <p className="text-xs text-muted-foreground">Код архитектуры включён в Studio. Один раз скачайте веса; последующие кадры рассчитываются на вашем устройстве без сети и API-ключей.</p>
      <Label className="block text-xs space-y-1">Папка весов (пусто = локальный cache)<Input id="world-play-path" value={path} disabled={!!session||busy||!!job} onChange={e=>setPath(e.target.value)} /></Label>
      {!csgo&&<Label className="block text-xs">Игра<Select value={game} disabled={!!session||busy||!!job} onChange={e=>{setGame(e.target.value);setPath('');setImage('');fileVersion.current++}} options={['Breakout','Pong'].map(value=>({value,label:value}))} /></Label>}
      {csgo?<><Label className="block text-xs">Стартовая сцена<Select aria-label="World spawn" value={String(spawn)} disabled={busy} onChange={e=>setSpawn(Number(e.target.value))} options={Array.from({length:7},(_,i)=>({value:String(i),label:`Сцена ${i+1}`}))} /></Label><Label className="flex gap-2 items-center text-xs"><input type="checkbox" checked={upscale} disabled={!!session||busy} onChange={e=>setUpscale(e.target.checked)} />Neural upsampler 150×280</Label></>:<Label className="block text-xs space-y-1">Screenshot Atari (Breakout: встроен пример)<Input type="file" accept="image/*" disabled={!!session||busy} onChange={e=>{const file=e.target.files?.[0];const v=++fileVersion.current;if(!file)return;if(file.size>14_000_000){setError('Максимум 14 MB');return}const reader=new FileReader();reader.onload=()=>{if(alive.current&&v===fileVersion.current)setImage(String(reader.result))};reader.readAsDataURL(file)}} /></Label>}
      <Label className="block text-xs">Устройство<Select aria-label="World play device" value={device} disabled={!!session||busy} onChange={e=>setDevice(e.target.value)} options={['auto','cuda','cpu'].map(value=>({value,label:value}))} /></Label>
      <Label className="block text-xs space-y-1">Denoising steps<Input id="world-play-denoising" type="number" min={1} max={10} value={steps} disabled={!!session||busy} onChange={e=>setSteps(Math.max(1,Math.min(10,Math.round(Number(e.target.value)||1))))} /></Label>
      <Button variant="outline" disabled={busy||!!job||!!session} onClick={download}>Скачать локальные веса</Button>
      {downloadStatus&&<p className="text-xs">{downloadStatus}</p>}
      <Button className="w-full" disabled={busy||!!job||!!session||resourceBlocked||(!csgo&&game==='Pong'&&!image)} onClick={load}>Загрузить модель</Button>
      {busy&&<p className="text-xs">Выполняется операция…</p>}
      {error&&<p role="alert" className="text-sm text-destructive whitespace-pre-wrap">{error}</p>}
      <p className="text-xs text-muted-foreground">Модель может терять детали и менять геометрию сцены. FPS зависит от GPU/CPU и числа diffusion steps; это нейросимуляция, не исходная игра. Пауза более 5 минут освобождает worker.</p>
    </CardContent></Card>
    <div className="space-y-4"><WorldResources modelId={modelId} operation="inference" config={{model_path:path,device,gpu_ids:[gpuId],upsample:upscale,denoising_steps:steps,game}} active={!!session} onBlockedChange={setResourceBlocked}/><Card><CardHeader><CardTitle>Попробовать world model</CardTitle></CardHeader><CardContent className="space-y-4">
      <div tabIndex={0} aria-label="Local world viewport" role="application" className="rounded border border-border bg-black min-h-64 flex justify-center items-center outline-none focus:ring-2 focus:ring-primary" onBlur={()=>{pressed.current.clear();action.current=0}} onKeyDown={e=>{
        const key=keyMap[e.code];if(!key||!session)return;e.preventDefault()
        if(csgo)pressed.current.add(key)
        else action.current=e.code==='Space'?1:['ArrowRight','ArrowUp','KeyD','KeyW'].includes(e.code)?2:['ArrowLeft','ArrowDown','KeyA','KeyS'].includes(e.code)?3:0
      }} onKeyUp={e=>{const key=keyMap[e.code];if(key){pressed.current.delete(key);action.current=0;e.preventDefault()}}}>
        {frame?<img src={frame.frame} alt="Local world frame" draggable={false} className="w-full max-h-[540px] object-contain" style={{imageRendering:upscale&&csgo?'auto':'pixelated'}} />:<p className="text-sm text-muted-foreground p-6">Загрузите модель — стартовая сцена появится здесь.</p>}
      </div>
      <p className="text-xs">{frame?`Шаг ${frame.step} · ${frame.device??''} ${frame.seconds?`· ${(1/frame.seconds).toFixed(1)} model FPS`:''}`:'После загрузки нажмите «Играть», затем кликните по сцене для управления клавиатурой.'}</p>
      <div className="flex flex-wrap gap-2"><Button disabled={!session||busy||running} onClick={()=>{play.current=true;setRunning(true)}}>Играть</Button><Button variant="outline" disabled={!running} onClick={pause}>Пауза</Button><Button variant="outline" disabled={!session||busy||running} onClick={()=>void tick()}>Один шаг</Button><Button variant="outline" disabled={!session||busy} onClick={reset}>Сбросить сцену</Button><Button variant="outline" disabled={!session||busy} onClick={close}>Освободить модель</Button></div>
      {csgo?<><p className="text-xs text-muted-foreground">WASD — движение; стрелки — камера; Space — прыжок. Экран должен иметь фокус. Кнопки ниже можно удерживать мышью.</p><div className="flex flex-wrap gap-2">{[['w','W'],['a','A'],['s','S'],['d','D'],['space','Прыжок'],['attack','Действие'],['look_left','Камера ←'],['look_right','Камера →']].map(([k,l])=>control(k,l))}</div></>:<div className="flex flex-wrap gap-2">{frame?.action_names?.map((name,i)=><Button variant="outline" key={name} disabled={!session} onPointerDown={e=>{e.currentTarget.setPointerCapture?.(e.pointerId);action.current=i}} onPointerUp={()=>{action.current=0}} onPointerCancel={()=>{action.current=0}}>{name}</Button>)}</div>}
    </CardContent></Card></div>
  </div>
}
