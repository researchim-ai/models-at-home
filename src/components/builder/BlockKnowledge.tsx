import { useTranslation } from 'react-i18next'
import { BookOpen, ExternalLink } from 'lucide-react'
import type { BlockDefinition } from '@/lib/builder'

export function BlockKnowledge({ definition }: { definition?:BlockDefinition }) {
  const { i18n }=useTranslation()
  const ru=i18n.language.startsWith('ru')
  const info=definition?.research
  if(!info) return null
  return <details className="rounded-lg border border-border bg-muted/30 p-3" data-testid="block-knowledge">
    <summary className="flex cursor-pointer items-center gap-2 text-xs font-semibold"><BookOpen className="h-3.5 w-3.5" />{ru?'Что это за блок?':'Understand this block'}<span className="ml-auto text-[10px] font-normal text-muted-foreground">{info.references.length} sources</span></summary>
    <div className="mt-3 space-y-3 text-[11px] leading-relaxed">
      <p>{info.summary[ru?'ru':'en']}</p>
      <code className="block overflow-auto rounded-md bg-muted p-2 text-[10px]">{info.formula}</code>
      <p className="font-mono text-[10px] text-muted-foreground">{info.shape_contract}</p>
      <p className="text-muted-foreground">{info.origin[ru?'ru':'en']}</p>
      <div className="space-y-2">{info.references.map(ref=><a key={ref.url} href={ref.url} target="_blank" rel="noopener noreferrer" className="block rounded-md border border-border p-2 hover:bg-accent" onClick={event=>{const open=window.electronAPI?.shell.openExternal;if(open){event.preventDefault();void open(ref.url)}}}>
        <span className="flex items-start gap-1 font-medium">{ref.title}<ExternalLink className="mt-0.5 h-3 w-3 shrink-0" /></span>
        <span className="text-[10px] text-muted-foreground">{ref.authors}{ref.year?` · ${ref.year}`:''} · {ref.relationship}</span>
      </a>)}</div>
      <div><p className="mb-1 font-semibold">{ru?'На что смотреть':'What to measure'}</p>{info.tradeoffs.map(text=><p key={text} className="mb-1 text-muted-foreground">• {text}</p>)}</div>
      <div><p className="mb-1 font-semibold">{ru?'Идеи экспериментов':'Experiment ideas'}</p>{info.experiments.map(text=><p key={text} className="mb-1 text-muted-foreground">• {text}</p>)}</div>
    </div>
  </details>
}
