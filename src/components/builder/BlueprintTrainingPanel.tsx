import { Link } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select } from '@/components/ui/select'
import type { LabHandoff } from '@/stores/labStore'

export function BlueprintTrainingPanel({ handoff, settings, patch, example, detach }: {
  handoff:LabHandoff; settings:{dataset_path:string;batch_size:number;learning_rate:number;max_steps:number;mixed_precision:string;device:string;vision_image_size:number;vision_patch_size:number;research_diagnostics:boolean}
  patch:(value:Partial<BlueprintTrainingPanelSettings>)=>void;example?:()=>void;detach:()=>void
}) {
  const field=(key:keyof BlueprintTrainingPanelSettings,label:string,type='number')=><div key={key} className="space-y-1"><Label htmlFor={`research-vlm-${key}`} className="text-xs">{label}</Label><Input id={`research-vlm-${key}`} type={type} value={String(settings[key])} step={key==='learning_rate'?'any':1} onChange={e=>patch({[key]:type==='number'?Number(e.target.value):e.target.value})}/></div>
  return <aside className="w-80 shrink-0 space-y-4 overflow-auto border-r border-border bg-card/50 p-4"><h2 className="text-sm font-semibold">VLM Stage · Research blueprint</h2>
    <div className="space-y-2 rounded-lg border border-border p-3"><p className="text-xs font-semibold">{handoff.name}</p><p className="text-[11px] leading-relaxed text-muted-foreground">Frozen backbone snapshot · {String(handoff.blueprint.hidden_size)} hidden · {String(handoff.blueprint.max_position_embeddings)} context. Native full-weight SFT, trained from scratch.</p><p className="text-[11px] text-muted-foreground">Trainable image patches + explicit byte vocabulary (259). This adapter has no pretrained vision knowledge.</p><Link to="/builder" className="text-xs text-primary underline">Open Research Lab</Link></div>
    <Label className="flex items-center gap-2 text-[11px]"><input type="checkbox" checked={settings.research_diagnostics} onChange={e=>patch({research_diagnostics:e.target.checked})}/>Capture research batch signals</Label>
    {field('dataset_path','Training dataset JSONL','text')}
    {example&&<Button variant="outline" size="sm" className="w-full" onClick={example}>Load local research demo</Button>}
    {field('batch_size','Batch size')}{field('learning_rate','Learning rate')}{field('max_steps','Max steps')}
    <div className="grid grid-cols-2 gap-2">{field('vision_image_size','Image size')}{field('vision_patch_size','Patch size')}</div>
    <Label className="block space-y-1 text-xs">Device<Select aria-label="Research VLM device" value={settings.device} options={['auto','cpu','cuda'].map(value=>({value,label:value.toUpperCase()}))} onChange={e=>patch({device:e.target.value})}/></Label>
    <Label className="block space-y-1 text-xs">Precision<Select aria-label="Research VLM precision" value={settings.mixed_precision} options={['no','bf16','fp16'].map(value=>({value,label:value==='no'?'Float32':value.toUpperCase()}))} onChange={e=>patch({mixed_precision:e.target.value})}/></Label>
    <p className="text-[11px] text-muted-foreground">Preflight checks image + text context and supervised tokens. Native SFT only; this blueprint adapter does not use Unsloth or GRPO.</p>
    <Button variant="ghost" size="sm" className="w-full" onClick={detach}>Switch to pretrained VLM workflow</Button>
  </aside>
}
export interface BlueprintTrainingPanelSettings {dataset_path:string;batch_size:number;learning_rate:number;max_steps:number;mixed_precision:string;device:string;vision_image_size:number;vision_patch_size:number;research_diagnostics:boolean}
