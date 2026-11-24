import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { LabAnalysis, LabProfile, ProbeResult, ResearchTask } from '@/lib/research'
export interface LabHandoff { name:string; path:string; blueprint:Record<string,unknown>; profile:LabProfile }
interface Baseline { id:string; name:string; note:string; blueprint:Record<string,unknown>; profile:LabProfile; parameters:number; macs:number; hash:string; probe?:Pick<ProbeResult,'device'|'precision'|'data_fingerprint'|'loss_history'|'weight_update_norm'|'causality'|'image_sensitivity'> }
interface LabState {
  handoffs:Partial<Record<ResearchTask,LabHandoff>>; baselines:Baseline[]
  handoff:(target:ResearchTask,value:LabHandoff)=>void; clearHandoff:(target:ResearchTask)=>void
  pin:(name:string,blueprint:Record<string,unknown>,analysis:LabAnalysis,probe?:ProbeResult)=>void
  note:(id:string,note:string)=>void; remove:(id:string)=>void
}
export const useLabStore=create<LabState>()(persist((set)=>({handoffs:{},baselines:[],
  handoff:(target,value)=>set(state=>({handoffs:{...state.handoffs,[target]:structuredClone(value)}})),
  clearHandoff:target=>set(state=>{const next={...state.handoffs};delete next[target];return{handoffs:next}}),
  pin:(name,blueprint,analysis,probe)=>set(state=>({baselines:[{id:crypto.randomUUID(),name,note:'',blueprint:structuredClone(blueprint),profile:structuredClone(analysis.profile),parameters:analysis.parameters,macs:analysis.dense_macs_estimate,hash:analysis.source_hash,
    probe:probe?{device:probe.device,precision:probe.precision,data_fingerprint:probe.data_fingerprint,loss_history:probe.loss_history,weight_update_norm:probe.weight_update_norm,causality:probe.causality,image_sensitivity:probe.image_sensitivity}:undefined},...state.baselines].slice(0,6)})),
  note:(id,note)=>set(state=>({baselines:state.baselines.map(item=>item.id===id?{...item,note:note.slice(0,4000)}:item)})),
  remove:id=>set(state=>({baselines:state.baselines.filter(item=>item.id!==id)})),
}),{name:'mah-research-lab'}))
