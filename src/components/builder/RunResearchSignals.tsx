import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { RuntimeSignalsData } from '@/lib/research'
import { BlockKnowledge } from './BlockKnowledge'
const value=(n:number|null)=>n==null?'—':n===0?'0':Math.abs(n)<.001?n.toExponential(2):n.toFixed(3)
export function RunResearchSignals({ signals }: {signals?:RuntimeSignalsData}) {
  const {data}=useQuery({queryKey:['research','block-catalog'],queryFn:api.listBlocks,enabled:!!signals,staleTime:Infinity})
  if(!signals)return null
  return <Card className="mt-4"><CardHeader className="pb-2"><CardTitle className="text-sm">Research · live batch signals</CardTitle><p className="text-[11px] text-muted-foreground">Captured training step {signals.step} · {signals.gradient_stage}</p></CardHeader><CardContent className="space-y-2"><details><summary className="cursor-pointer text-xs text-muted-foreground">Inspect real activations and gradients</summary><div className="max-h-80 overflow-auto"><table className="lab-table"><thead><tr><th>Block</th><th>Shape</th><th>RMS / std</th><th>Grad RMS sample</th></tr></thead><tbody>{signals.blocks.map(row=><tr key={row.path}><td>{row.path}<BlockKnowledge definition={data?.blocks.find(block=>block.type===row.type)}/></td><td className="font-mono">[{row.shape.join(',')}]</td><td>{value(row.rms)} / {value(row.std)}</td><td>{value(row.gradient_rms_sampled)}{row.parameters_without_grad>0&&<span className="block text-[10px] text-warning">{row.parameters_without_grad} params without grad</span>}</td></tr>)}</tbody></table></div></details><p className="text-[10px] text-muted-foreground">{signals.note}</p></CardContent></Card>
}
