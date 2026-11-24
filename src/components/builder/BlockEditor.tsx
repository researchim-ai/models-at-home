import { BlockKnowledge } from './BlockKnowledge'
import { useEffect, useId, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { ArrowDown, ArrowUp, Copy, Plus, Trash2 } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Select } from '@/components/ui/select'
import { Label } from '@/components/ui/label'
import { initialParams, type BlockDefinition, type BodyBlock, type Dimensions } from '@/lib/builder'

const options: Record<string, string[]> = {
  input_injection: ['none', 'add', 'concat'], state_init: ['input', 'zeros'], step_embedding: ['none', 'sinusoidal'],
  activation: ['silu', 'gelu', 'relu', 'tanh', 'sigmoid', 'leaky_relu'], expert_type: ['mlp', 'swiglu'],
}
const hiddenOnly = (item: BlockDefinition) => !['token_embedding', 'positional_embedding', 'add', 'multiply', 'concat'].includes(item.type)

function JsonParams({ value, onApply }: { value: Record<string, unknown>; onApply: (value: Record<string, unknown>) => void }) {
  const [draft, setDraft] = useState(JSON.stringify(value, null, 2))
  const [error, setError] = useState('')
  const id = useId()
  useEffect(() => { setDraft(JSON.stringify(value, null, 2)); setError('') }, [value])
  return <details className="border-t border-border pt-2"><summary className="cursor-pointer text-xs text-muted-foreground">Parameters JSON</summary>
    <Label htmlFor={id} className="sr-only">Parameters JSON</Label>
    <textarea id={id} className="mt-2 min-h-40 w-full rounded-md border border-input bg-background p-2 font-mono text-[11px]" value={draft} onChange={e => setDraft(e.target.value)} />
    {error && <p role="alert" className="text-xs text-destructive">{error}</p>}
    <Button size="sm" variant="outline" onClick={() => { try { const parsed = JSON.parse(draft); if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('Expected a JSON object'); onApply(parsed); setError('') } catch (e) { setError(String(e)) } }}>Apply parameters</Button>
  </details>
}

export function BlockEditor({ block, definitions, dimensions, onChange, depth = 0 }: {
  block: BodyBlock; definitions: BlockDefinition[]; dimensions: Dimensions; onChange: (block: BodyBlock) => void; depth?: number
}) {
  const id = useId()
  const { i18n } = useTranslation()
  const ru = i18n.language.startsWith('ru')
  const definition = definitions.find(item => item.type === block.type)
  const set = (key: string, value: unknown) => onChange({ ...block, params: { ...block.params, [key]: value } })
  const body = Array.isArray(block.params.body) ? block.params.body as BodyBlock[] : []
  const scalar = Object.entries(block.params).filter(([key, value]) => !['body', 'block_type', 'block_params'].includes(key) && (typeof value !== 'object' || value === null))
  const advanced = block.type === 'loop' ? ['min_iterations', 'random_depth', 'gradient_steps', 'checkpoint', 'step_embedding'] : []
  const renderFields = (items: [string, unknown][]) => (<div className="grid grid-cols-2 gap-2">{items.map(([key, value]) => <div key={key} className={typeof value === 'boolean' ? 'col-span-2' : 'min-w-0 space-y-1'}>
      {typeof value === 'boolean' ? <Label className="flex items-center gap-2 text-[11px]"><input type="checkbox" checked={value} onChange={e => set(key, e.target.checked)} />{key}</Label> : <>
        <Label htmlFor={`${id}-${key}`} className="break-words text-[10px]">{key}</Label>
        {options[key] ? <Select id={`${id}-${key}`} className="h-8 text-xs" value={String(value)} options={options[key].map(v => ({ value: v, label: v }))} onChange={e => set(key, e.target.value)} />
          : key === 'code' ? <textarea id={`${id}-${key}`} className="min-h-32 w-full rounded-md border border-input bg-background p-2 font-mono text-[11px]" value={String(value ?? '')} onChange={e => set(key, e.target.value)} /> : <Input id={`${id}-${key}`} className="h-8 text-xs" type={typeof value === 'number' ? 'number' : 'text'} step="any" value={String(value ?? '')} onChange={e => set(key, typeof value === 'number' && e.target.value !== '' ? Number(e.target.value) : e.target.value)} />}
      </>}
    </div>)}</div>)
  return <div className="min-w-0 space-y-3">
    <div className="space-y-1"><Label htmlFor={`${id}-type`} className="text-xs">{ru ? 'Тип блока' : 'Block type'}</Label>
      <Select id={`${id}-type`} value={block.type} options={(depth ? definitions.filter(hiddenOnly) : definitions).filter(item => depth < 4 || !['loop', 'residual', 'repeater'].includes(item.type)).map(item => ({ value: item.type, label: item.type }))}
        onChange={e => onChange({ type: e.target.value, params: initialParams(e.target.value, dimensions, definitions) })} />
      <p className="text-[11px] leading-relaxed text-muted-foreground">{definition?.description}</p>
    </div>
    <BlockKnowledge definition={definition} />
    {definition?.category === 'Advanced' && <p className="rounded-md bg-muted p-2 text-[11px] text-muted-foreground">{ru ? 'Для локальных Python-операторов нужна ручная проверка forward/backward. Анализатор не исполняет пользовательский код и не подтверждает causal-контракт.' : 'Local Python operators need a manual forward/backward probe. Analysis does not execute user code or verify its causal contract.'}</p>}
    {block.type === 'loop' && <p className="rounded-md bg-muted p-2 text-[11px] leading-relaxed">{ru ? 'Общие веса: тело хранится один раз. Больше итераций = больше вычислений и активаций, а не больше параметров. gradient_steps = 0: полный BPTT.' : 'Shared weights: one physical body. More iterations mean more compute and activations, not more parameters. gradient_steps = 0: full BPTT.'}</p>}
    {renderFields(scalar.filter(([key]) => !advanced.includes(key)))}
    {scalar.some(([key]) => advanced.includes(key)) && <details className="rounded-md border border-border p-2"><summary className="cursor-pointer text-[11px] text-muted-foreground">{ru ? 'Глубина и градиенты' : 'Depth sampling & gradients'}</summary><div className="mt-3">{renderFields(scalar.filter(([key]) => advanced.includes(key)))}</div></details>}
    {['loop', 'residual'].includes(block.type) && <fieldset className="min-w-0 space-y-2 border-l-2 border-primary/30 pl-3">
      <legend className="mb-2 text-xs font-semibold">{ru ? 'Тело контейнера' : 'Container body'} · {body.length}</legend>
      {body.map((child, index) => <div key={index} className="space-y-2 rounded-lg border border-border p-2">
        <div className="flex items-center justify-between"><span className="text-[10px] text-muted-foreground">{index + 1}. {child.type}</span><div className="flex">
          {[[-1, ArrowUp], [1, ArrowDown]] .map(([offset, Icon]) => { const IconComponent = Icon as typeof ArrowUp; const next = index + Number(offset); return <Button key={String(offset)} variant="ghost" size="icon" className="h-6 w-6" aria-label={Number(offset) < 0 ? 'Move body block up' : 'Move body block down'} disabled={next < 0 || next >= body.length} onClick={() => { const copy = [...body]; [copy[index], copy[next]] = [copy[next], copy[index]]; set('body', copy) }}><IconComponent className="h-3 w-3" /></Button> })}
          <Button variant="ghost" size="icon" className="h-6 w-6" aria-label="Duplicate body block" onClick={() => set('body', [...body.slice(0, index + 1), structuredClone(child), ...body.slice(index + 1)])}><Copy className="h-3 w-3" /></Button>
          <Button variant="ghost" size="icon" className="h-6 w-6" aria-label="Remove body block" onClick={() => set('body', body.filter((_, i) => i !== index))}><Trash2 className="h-3 w-3" /></Button>
        </div></div>
        {depth < 4 ? <BlockEditor block={child} definitions={definitions} dimensions={dimensions} depth={depth + 1} onChange={value => set('body', body.map((b, i) => i === index ? value : b))} /> : <p className="text-xs text-warning">Nesting limit reached</p>}
      </div>)}
      <Select aria-label="Add body block" value="" placeholder={ru ? '+ Добавить блок в тело' : '+ Add a body block'} options={definitions.filter(hiddenOnly).map(item => ({ value: item.type, label: item.type }))}
        onChange={e => set('body', [...body, { type: e.target.value, params: initialParams(e.target.value, dimensions, definitions) }])} />
    </fieldset>}
    {block.type === 'repeater' && depth < 4 && <fieldset className="min-w-0 space-y-2 border-l-2 border-border pl-3"><legend className="text-xs font-semibold">{ru ? 'Независимые копии' : 'Independent copies'}</legend>
      <BlockEditor block={{ type: String(block.params.block_type ?? 'llama_block'), params: (block.params.block_params ?? {}) as Record<string, unknown> }} definitions={definitions.filter(hiddenOnly)} dimensions={dimensions} depth={depth + 1}
        onChange={child => onChange({ ...block, params: { ...block.params, block_type: child.type, block_params: child.params } })} />
    </fieldset>}
    <JsonParams value={block.params} onApply={params => onChange({ ...block, params })} />
  </div>
}
