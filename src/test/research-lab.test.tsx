import { useState } from 'react'
import { act, fireEvent, render, screen, waitFor, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import '@/i18n'
import App from '@/App'
import { BlockKnowledge } from '@/components/builder/BlockKnowledge'
import { ResearchPanel } from '@/components/builder/ResearchPanel'
import { defaultLabProfile, type LabAnalysis, type LabProfile, type ProbeResult } from '@/lib/research'
import { emptyDraft } from '@/stores/builderDraft'
import { useLabStore } from '@/stores/labStore'

const blueprint={model_type:'homellm_blueprint',hidden_size:32,vocab_size:100,max_position_embeddings:128,blocks:[
  {id:'embedding',type:'token_embedding',params:{},inputs:[]},{id:'attention',type:'causal_self_attention',params:{num_heads:4},inputs:['embedding']},
]}
const analysis={blueprint,effective_blueprint:blueprint,source_hash:'a'.repeat(64),effective_hash:'b'.repeat(64),profile:defaultLabProfile,
  parameters:1000,parameter_memory_fp32_mib:.01,fp32_adam_parameter_floor_mib:.04,max_block_applications:2,loops:[],blocks:[],dense_macs_estimate:10000,output_tensor_volume_mib:.1,
  hardware:{ram_available_mib:1000,torch_cuda_available:false,fit_status:'requires_activation_measurement'},contracts:{task:'llm',text_tokens:16,vision_tokens:0,vocabulary:100,output:'logits',adapter:'none'},
  capabilities:{llm:{supported:true,modes:[],notes:[]},vlm:{supported:true,modes:[],notes:[]},vla:{supported:true,modes:[],notes:[]}},warnings:[],
} as LabAnalysis
const response=(payload:unknown)=>new Response(JSON.stringify(payload),{status:200,headers:{'Content-Type':'application/json'}})

beforeEach(()=>{localStorage.clear();useLabStore.setState({handoffs:{},baselines:[]})})
afterEach(()=>{cleanup();vi.unstubAllGlobals()})

it('links the actual primary publication and distinguishes it from the local implementation',()=>{
  render(<BlockKnowledge definition={{type:'rmsnorm',description:'',research:{summary:{ru:'RMS',en:'RMS'},formula:'x / RMS(x)',shape_contract:'[B,T,H]',origin:{ru:'Zhang and Sennrich',en:'Zhang and Sennrich'},tradeoffs:[],experiments:[],references:[{title:'Root Mean Square Layer Normalization',authors:'Biao Zhang; Rico Sennrich',year:2019,url:'https://arxiv.org/abs/1910.07467',relationship:'research'}]}}}/>)
  fireEvent.click(screen.getByText('Understand this block'))
  expect(screen.getByRole('link',{name:/Root Mean Square/})).toHaveAttribute('href','https://arxiv.org/abs/1910.07467')
  expect(screen.getByText(/Biao Zhang; Rico Sennrich/)).toHaveTextContent('2019')
})

it('keeps a stale synthetic result out of the view after the measurement profile changes',async()=>{
  let finish:(value:Response)=>void=()=>{}
  vi.stubGlobal('fetch',vi.fn(()=>new Promise<Response>(resolve=>{finish=resolve})))
  function Host(){const[profile,set]=useState(defaultLabProfile);return <ResearchPanel blueprint={blueprint} name="variant" profile={profile} setProfile={patch=>set(old=>({...old,...patch}))} analysis={analysis} inspect={()=>{}} restore={()=>{}}/>}
  render(<Host/>)
  fireEvent.click(screen.getByRole('button',{name:'Run synthetic probe'}))
  await waitFor(()=>expect(fetch).toHaveBeenCalled())
  fireEvent.change(screen.getByLabelText('Probe text tokens'),{target:{value:'32'}})
  await act(async()=>finish(response({source_hash:'old',loss_history:[1],data_fingerprint:'c'.repeat(64)} as ProbeResult)))
  await waitFor(()=>expect(screen.getByRole('button',{name:'Run synthetic probe'})).not.toBeDisabled())
  expect(screen.queryByText(/Synthetic loss:/)).not.toBeInTheDocument()
})

it('pins architecture and notes as a separate immutable baseline',()=>{
  useLabStore.getState().pin('variant',blueprint,analysis)
  const baseline=useLabStore.getState().baselines[0]
  expect(baseline.blueprint).toEqual(blueprint)
  expect(baseline.blueprint).not.toBe(blueprint)
  useLabStore.getState().note(baseline.id,'Hypothesis: reduce KV heads')
  expect(useLabStore.getState().baselines[0].note).toContain('KV heads')
})

for(const target of ['vlm','vla'] as const){
  it(`hands a frozen architecture to the real ${target.toUpperCase()} page without starting training`,async()=>{
    const requests:string[]=[]
    const profile:LabProfile={...defaultLabProfile,task:target}
    localStorage.setItem('mah-builder-draft',JSON.stringify({...emptyDraft,blocks:blueprint.blocks,hiddenSize:32,vocabSize:100,maxPositions:128,labProfile:profile}))
    vi.stubGlobal('fetch',vi.fn(async(input:RequestInfo|URL)=>{
      const url=String(input);requests.push(url)
      if(url.endsWith('/blueprints/blocks'))return response({blocks:[{type:'token_embedding',description:''},{type:'causal_self_attention',description:''}]})
      if(url.endsWith('/blueprints/research/analyze'))return response({...analysis,profile})
      if(url.endsWith('/blueprints/save'))return response({success:true,path:'/workspace/blueprints/lab.json'})
      return response({status:'ok',running:false,runs:[],gpus:[],examples:[],metrics:null,history:[],stdout:'',stderr:'',models:[],datasets:[],blueprints:[],backends:[],notebooks:[]})
    }))
    const query=new QueryClient({defaultOptions:{queries:{retry:false}}})
    render(<QueryClientProvider client={query}><MemoryRouter initialEntries={['/builder']}><App/></MemoryRouter></QueryClientProvider>)
    fireEvent.click(await screen.findByRole('button',{name:`Train in ${target.toUpperCase()} Studio`}))
    if(target==='vlm')await screen.findByText('VLM Stage · Research blueprint')
    else await waitFor(()=>expect(screen.getByRole('combobox',{name:'VLA architecture'})).toHaveValue('blueprint_vla'))
    expect(useLabStore.getState().handoffs[target]?.blueprint).toEqual(expect.objectContaining({hidden_size:32}))
    expect(requests.some(url=>url.includes('/training/start')||url.endsWith(`/${target}/start`))).toBe(false)
    query.clear()
  })
}
