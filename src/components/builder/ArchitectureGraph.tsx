import type { BuilderBlock } from '@/lib/builder'
export function ArchitectureGraph({ blocks, inspect }: {blocks:BuilderBlock[];inspect:(id:string)=>void}) {
  const levels=new Map<string,number>();const layers:BuilderBlock[][]=[]
  blocks.forEach((block,index)=>{const parents=block.inputs.length?block.inputs:index?[blocks[index-1].id]:[];const level=parents.length?Math.max(...parents.map(parent=>levels.get(parent)??-1))+1:0;levels.set(block.id,level);(layers[level]??=[]).push(block)})
  const width=Math.max(440,Math.max(1,...layers.map(layer=>layer.length))*180)
  const positions=new Map<string,{x:number;y:number}>()
  layers.forEach((layer,y)=>layer.forEach((block,x)=>positions.set(block.id,{x:(x+.5)*width/layer.length,y:y*74+30})))
  return <div className="max-h-80 overflow-auto rounded-lg border border-border bg-muted/20"><svg role="img" aria-label="Architecture dataflow" viewBox={`0 0 ${width} ${layers.length*74+20}`} style={{minWidth:width,width:'100%'}}>
    {blocks.flatMap((block,index)=>{const to=positions.get(block.id)!;return(block.inputs.length?block.inputs:index?[blocks[index-1].id]:[]).map(parent=>{const from=positions.get(parent);return from?<path key={`${parent}-${block.id}`} d={`M ${from.x} ${from.y+19} C ${from.x} ${from.y+45}, ${to.x} ${to.y-40}, ${to.x} ${to.y-19}`} fill="none" stroke="var(--color-muted-foreground)" strokeWidth="1.2" opacity=".5"/>:null})})}
    {blocks.map(block=>{const point=positions.get(block.id)!;return <g key={block.id} role="button" tabIndex={0} aria-label={`Inspect ${block.id}`} className="cursor-pointer" onClick={()=>inspect(block.id)} onKeyDown={e=>{if(['Enter',' '].includes(e.key)){e.preventDefault();inspect(block.id)}}}>
      <rect x={point.x-76} y={point.y-20} width={152} height={40} rx={8} fill="var(--color-card)" stroke={block.type==='loop'?'var(--color-primary)':'var(--color-border)'} />
      <text x={point.x} y={point.y-1} fill="var(--color-foreground)" fontSize={11} textAnchor="middle">{block.type}{block.type==='loop'?` ×${block.params.iterations??4}`:''}</text>
      <text x={point.x} y={point.y+12} fill="var(--color-muted-foreground)" fontSize={8} textAnchor="middle">{block.id.length>23?block.id.slice(0,20)+'…':block.id}</text>
    </g>})}
  </svg></div>
}
