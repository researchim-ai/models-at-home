import { fireEvent, render, screen, waitFor, cleanup } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, expect, it, vi } from 'vitest'
import { WorldResources } from '@/components/world/WorldResources'
import { api } from '@/api/client'
const report={status:'blocked',device:'cuda',hardware:{ram_available_gib:20,ram_total_gib:32,cpu_threads:4,disk_free_gib:100,cache_disk_free_gib:100},gpu:{id:1,name:'Selected GPU',free_gib:.1,total_gib:24},memory_gib:[1,2],ram_gib:[1,2],training_supported:true,benchmark_supported:true,reasons:['Недостаточно свободной VRAM'],warnings:[],hints:[],basis:'Предварительная оценка'}
afterEach(()=>{cleanup();vi.restoreAllMocks()})
function show(props:Record<string,unknown>={}){const qc=new QueryClient({defaultOptions:{queries:{retry:false}}});const view=render(<QueryClientProvider client={qc}><WorldResources modelId="rssm" operation="train" config={{batch_size:8}} {...props}/></QueryClientProvider>);return {...view,qc}}
it('shows selected GPU free memory and prevents launching an oversized configuration',async()=>{
 vi.spyOn(api,'worldResources').mockResolvedValue(report as never);const gate=vi.fn();const v=show({onBlockedChange:gate})
 expect(await screen.findByText('Не помещается / запуск заблокирован')).toBeInTheDocument()
 expect(screen.getByText(/GPU 1: Selected GPU/)).toHaveTextContent('0.10 GiB / 24.00 GiB')
 await waitFor(()=>expect(gate).toHaveBeenLastCalledWith(true));v.unmount();v.qc.clear()
})
it('shows a measured duration after an actual benchmark response',async()=>{
 vi.spyOn(api,'worldResources').mockResolvedValue({...report,status:'likely_fit',reasons:[]} as never)
 const measured={...report,status:'likely_fit',reasons:[],eta:{remaining_steps:500,total_seconds_low:60,total_seconds_high:120,basis:'Реальные optimizer steps',note:'Учтены сохранения'}}
 const probe=vi.spyOn(api,'benchmarkWorldResources').mockResolvedValue(measured as never);const v=show()
 fireEvent.click(await screen.findByRole('button',{name:'Измерить 5 шагов'}))
 expect(await screen.findByText('Примерное время на 500 шагов: 1 мин – 2 мин')).toBeInTheDocument()
 expect(probe).toHaveBeenCalledWith({model_id:'rssm',operation:'train',config:{batch_size:8}})
 v.unmount();v.qc.clear()
})
it('does not advertise pretrained tuning or an invented ETA',async()=>{
 vi.spyOn(api,'worldResources').mockResolvedValue({...report,status:'unsupported',training_supported:false,benchmark_supported:false,memory_gib:[null,null],training_state_lower_bound_gib:5.7} as never)
 const v=show({modelId:'diamond_csgo'});expect(await screen.findByText(/Дообучение этой pretrained-модели пока не подключено/)).toBeInTheDocument()
 expect(screen.queryByRole('button',{name:'Измерить 5 шагов'})).not.toBeInTheDocument();v.unmount();v.qc.clear()
})
