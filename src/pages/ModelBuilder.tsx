import { ResearchPanel } from '@/components/builder/ResearchPanel'
import { defaultLabProfile, type LabAnalysis } from '@/lib/research'
import { useLabStore } from '@/stores/labStore'
import { useBuilderDraft } from '@/stores/builderDraft'
import { BlockEditor } from '@/components/builder/BlockEditor'
import { Select } from '@/components/ui/select'
import { initialParams, recipe, duplicateSequenceBlock, type BlockDefinition, type BlueprintAnalysis } from '@/lib/builder'
import { useState, useCallback, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import {
  Save,
  Search,
  Layers,
  Cpu,
  Repeat,
  Shuffle,
  ChevronRight,
  Plus,
  Copy,
  Trash2,
  FolderOpen,
  FileText,
  RefreshCw,
  ArrowUp,
  ArrowDown,
  Play,
  CheckCircle2,
  X,
  Undo2,
  Redo2,
} from 'lucide-react'
import { api } from '@/api/client'
import { isSequence, connectSequence, syncBlueprintDimensions } from '@/lib/blueprint'
import { toast } from 'sonner'
import { useTrainingStore } from '@/stores/trainingStore'

type BlockDef = BlockDefinition

interface BlockInstance {
  id: string
  type: string
  params: Record<string, unknown>
  inputs: string[]
}

interface SavedBlueprint {
  name: string
  path: string
  hidden_size: number
  vocab_size: number
  num_blocks: number
  modified_at: string
}

const ICON_BY_TYPE: Record<string, { icon: typeof Cpu; color: string }> = {
  token_embedding: { icon: Layers, color: 'bg-blue-500/20 text-blue-400' },
  positional_embedding: { icon: Layers, color: 'bg-blue-500/20 text-blue-400' },
  attention: { icon: Search, color: 'bg-purple-500/20 text-purple-400' },
  causal_self_attention: { icon: Search, color: 'bg-purple-500/20 text-purple-400' },
  mlp: { icon: Cpu, color: 'bg-green-500/20 text-green-400' },
  swiglu: { icon: Cpu, color: 'bg-green-500/20 text-green-400' },
  liger_swiglu: { icon: Cpu, color: 'bg-green-600/20 text-green-300' },
  rmsnorm: { icon: Repeat, color: 'bg-yellow-500/20 text-yellow-400' },
  liger_rmsnorm: { icon: Repeat, color: 'bg-yellow-600/20 text-yellow-300' },
  layernorm: { icon: Repeat, color: 'bg-orange-500/20 text-orange-400' },
  liger_layernorm: { icon: Repeat, color: 'bg-orange-600/20 text-orange-300' },
  moe: { icon: Layers, color: 'bg-red-500/20 text-red-400' },
  add: { icon: Plus, color: 'bg-cyan-500/20 text-cyan-400' },
  multiply: { icon: Shuffle, color: 'bg-pink-500/20 text-pink-400' },
  linear: { icon: ChevronRight, color: 'bg-teal-500/20 text-teal-400' },
  llama_block: { icon: Layers, color: 'bg-indigo-500/20 text-indigo-400' },
  liger_llama_block: { icon: Layers, color: 'bg-indigo-600/20 text-indigo-300' },
  loop: { icon: Repeat, color: 'bg-primary/15 text-primary' },
  repeater: { icon: Layers, color: 'bg-primary/15 text-primary' },
  residual: { icon: Shuffle, color: 'bg-primary/15 text-primary' },
  dropout: { icon: Shuffle, color: 'bg-gray-500/20 text-gray-400' },
}

function getIconFor(type: string) {
  return ICON_BY_TYPE[type] ?? { icon: Cpu, color: 'bg-muted text-muted-foreground' }
}


export function ModelBuilder() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const draft = useBuilderDraft()
  const { blocks, blueprintName, hiddenSize, vocabSize, maxPositions, autoProject, blueprintExtras } = draft
  const setBlocks = (value: React.SetStateAction<BlockInstance[]>) => draft.setField('blocks', value)
  const setBlueprintName = (value: string) => draft.setField('blueprintName', value)


  const [search, setSearch] = useState('')
  const [analyzing, setAnalyzing] = useState(false)
  const [analysis, setAnalysis] = useState<{ key: string; value: LabAnalysis } | null>(null)
  const [analysisError, setAnalysisError] = useState('')
  const [available, setAvailable] = useState<BlockDef[]>([])
  const [saved, setSaved] = useState<SavedBlueprint[]>([])
  const [selectedBlock, setSelectedBlock] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [jsonDraft, setJsonDraft] = useState<string | null>(null)
  const [jsonError, setJsonError] = useState<string | null>(null)

  const loadBlocks = useCallback(async () => {
    try {
      const data = await api.listBlocks()
      setAvailable(data.blocks ?? [])
    } catch {
      toast.error('Не удалось загрузить список блоков')
    }
  }, [])

  const loadSaved = useCallback(async () => {
    try {
      const data = await api.listBlueprints()
      setSaved(data.blueprints ?? [])
    } catch { /* ignore */ }
  }, [])

  useEffect(() => {
    loadBlocks()
    loadSaved()
  }, [loadBlocks, loadSaved])

  const addBlock = useCallback((type: string) => {
    const id = `${type}_${crypto.randomUUID()}`
    setBlocks((prev) => {
      const previousId = prev.length > 0 ? prev[prev.length - 1].id : null
      return [...prev, {
        id,
        type,
        params: initialParams(type, { hidden: hiddenSize, vocab: vocabSize, positions: maxPositions }, available),
        inputs: previousId ? [previousId] : [],
      }]
    })
    setSelectedBlock(id)
  }, [hiddenSize, vocabSize, maxPositions, available])

  const removeBlock = (id: string) => {
    if (!isSequence(blocks) && blocks.some(b => b.inputs.includes(id))) {
      toast.error('Блок используется графом. Сначала измени связи в blueprint JSON.')
      return
    }
    setBlocks(prev => isSequence(prev)
      ? connectSequence(prev.filter(b => b.id !== id))
      : prev.filter(b => b.id !== id))
    if (selectedBlock === id) setSelectedBlock(null)
  }

  const moveBlock = (id: string, dir: -1 | 1) => {
    if (!isSequence(blocks)) {
      toast.error('Перестановка доступна для последовательной архитектуры; граф сохранён без изменений.')
      return
    }
    setBlocks((prev) => {
      const idx = prev.findIndex((b) => b.id === id)
      if (idx === -1) return prev
      const next = idx + dir
      if (next < 0 || next >= prev.length) return prev
      const copy = [...prev]
      ;[copy[idx], copy[next]] = [copy[next], copy[idx]]
      return connectSequence(copy)
    })
  }


  const blueprint = {
    ...blueprintExtras,
    model_type: 'homellm_blueprint',
    vocab_size: vocabSize,
    hidden_size: hiddenSize,
    max_position_embeddings: maxPositions,
    auto_project: autoProject,
    blocks: blocks.map((b) => ({
      id: b.id,
      type: b.type,
      params: b.params,
      inputs: b.inputs,
    })),
    training: blueprintExtras.training ?? {
      optimizer: 'adamw',
      lr: 0.001,
      weight_decay: 0.01,
      loss_fn: 'cross_entropy',
    },
  }

  const labProfile = draft.labProfile ?? defaultLabProfile
  const fingerprint = JSON.stringify({blueprint,profile:labProfile})
  const currentAnalysis = analysis?.key === fingerprint ? analysis.value : null
  const dimensions = { hidden: hiddenSize, vocab: vocabSize, positions: maxPositions }
  const analyze = async () => {
    setAnalyzing(true); setAnalysisError('')
    try { setAnalysis({ key: fingerprint, value: await api.researchAnalyze(blueprint,labProfile) }) }
    catch (error) { setAnalysisError(error instanceof Error ? error.message : String(error)) }
    finally { setAnalyzing(false) }
  }
  const duplicate = (id: string) => {
    if (!isSequence(blocks)) { toast.error('For a branched graph, duplicate branches through JSON or add a block and select its inputs.'); return }
    setBlocks(current => duplicateSequenceBlock(current, id))
  }
  const wrapInLoop = (id: string) => {
    const original = blocks.find(b => b.id === id)
    if (!original || ['token_embedding', 'positional_embedding', 'add', 'multiply', 'concat'].includes(original.type)) return
    setBlocks(current => current.map(b => b.id === id ? { ...b, type: 'loop', params: {
      ...initialParams('loop', dimensions, available), body: [{ type: b.type, params: structuredClone(b.params) }],
    } } : b))
  }

  const handleSave = async () => {
    if (blocks.length === 0) {
      toast.error('Нечего сохранять — добавь блоки')
      return
    }
    setSaving(true)
    try {
      await api.saveBlueprint(blueprintName, blueprint)
      toast.success(`Blueprint "${blueprintName}" сохранён`)
      loadSaved()
    } catch (e) {
      toast.error(`Не удалось сохранить: ${e}`)
    } finally {
      setSaving(false)
    }
  }

  const handleTrain = async () => {
    setSaving(true)
    try {
      if(labProfile.task!=='llm') await api.researchAnalyze(blueprint,labProfile)
      const result = await api.saveBlueprint(blueprintName, blueprint)
      useLabStore.getState().handoff(labProfile.task,{name:blueprintName,path:result.path,blueprint,profile:labProfile})
      if(labProfile.task!=='llm') {
        navigate('/'+labProfile.task)
        toast.success(`Blueprint snapshot sent to ${labProfile.task.toUpperCase()} Studio. Choose data and run preflight.`)
        return
      }
      const store = useTrainingStore.getState()
      store.setStage('pretrain')
      store.updateConfig({ model_type: 'blueprint', model_blueprint: result.path,
        research_diagnostics:true, model_blueprint_spec: blueprint, hidden_size: hiddenSize, vocab_size: vocabSize,
        seq_len: maxPositions, training_backend: 'models-at-home', tuning_method: 'full', use_lora: false,
        base_model_path: '', model_path: '' })
      navigate('/')
      toast.success('Схема передана в Training Studio. Выбери токенизатор и данные для обучения.')
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error))
    } finally { setSaving(false) }
  }

  const handleLoad = async (name: string) => {
    try {
      const data = await api.getBlueprint(name)
      const bp = data.blueprint as any
      draft.patch({ blueprintName: name, hiddenSize: bp.hidden_size ?? 512, vocabSize: bp.vocab_size ?? 50257,
        maxPositions: bp.max_position_embeddings ?? 2048, autoProject: bp.auto_project ?? true,
        blueprintExtras: { dtype: bp.dtype, training: bp.training },
        blocks: (bp.blocks ?? []).map((b: any) => ({ id: b.id, type: b.type, params: b.params ?? {}, inputs: b.inputs ?? [] })) })
      toast.success(`Загружен blueprint "${name}"`)
    } catch {
      toast.error('Не удалось загрузить blueprint')
    }
  }

  const applyJson = async () => {
    if (jsonDraft === null) return
    setJsonError(null)
    try {
      const parsed = JSON.parse(jsonDraft)
      const { blueprint: value } = await api.validateBlueprint(parsed)
      draft.patch({ hiddenSize: value.hidden_size as number, vocabSize: value.vocab_size as number,
        maxPositions: value.max_position_embeddings as number, autoProject: value.auto_project as boolean,
        blueprintExtras: { dtype: value.dtype, training: value.training }, blocks: value.blocks as BlockInstance[] })
      setJsonDraft(null)
    } catch (error) {
      setJsonError(error instanceof Error ? error.message : String(error))
    }
  }

  const handleDelete = async (name: string) => {
    if (!confirm(`Удалить blueprint "${name}"?`)) return
    try {
      await api.deleteBlueprint(name)
      toast.success('Удалено')
      loadSaved()
    } catch {
      toast.error('Ошибка удаления')
    }
  }

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(blueprint, null, 2))
      toast.success('Blueprint скопирован')
    } catch {
      toast.error('Не удалось скопировать blueprint в буфер обмена')
    }
  }

  const currentBlock = blocks.find((b) => b.id === selectedBlock)

  const resize = (changes: Partial<{ hidden: number; vocab: number; positions: number }>) => {
    const before = { hidden: hiddenSize, vocab: vocabSize, positions: maxPositions }
    const after = { ...before, ...changes }
    if (Object.values(after).some(value => !Number.isInteger(value) || value <= 0)) return
    draft.patch({ hiddenSize: after.hidden, vocabSize: after.vocab, maxPositions: after.positions,
      blocks: syncBlueprintDimensions(blocks, before, after) })
  }

  return (
    <div className="builder-workspace flex h-full">
      <div className="builder-palette min-h-0 w-64 shrink-0 border-r border-border flex flex-col">
        <ScrollArea className="flex-1 p-4">
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="builder-recipe" className="text-xs">Start from a recipe</Label>
              <Select id="builder-recipe" disabled={!available.length} value="" placeholder="Choose architecture…" options={[{ value: 'transformer', label: 'Transformer · independent layers' }, { value: 'looped', label: 'Looped Transformer · shared core' }, { value: 'moe', label: 'Sparse MoE · residual branches' }, { value:'lab_tiny',label:'CPU Research Lab · small shared core' }]}
                onChange={e => { const small=e.target.value==='lab_tiny'; const dims=small?{hidden:32,vocab:259,positions:128}:dimensions; draft.patch({ blocks: recipe(small?'looped':e.target.value, dims, available), hiddenSize:dims.hidden,vocabSize:dims.vocab,maxPositions:dims.positions,blueprintName:e.target.value+'_model',blueprintExtras:{} }); setSelectedBlock(null); setJsonDraft(null) }} />
              <p className="text-[10px] leading-relaxed text-muted-foreground">A recipe replaces the current draft. Save your design before switching.</p>
            </div>
            <div>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  {t('builder.blocks')}
                </h3>
                <Button variant="ghost" size="icon" onClick={loadBlocks}>
                  <RefreshCw className="h-3 w-3" />
                </Button>
              </div>
              <Input aria-label="Search blocks" placeholder="Search blocks…" className="mb-3 h-8 text-xs" value={search} onChange={e => setSearch(e.target.value)} />
              <div className="space-y-1">
                {available.filter(item => `${item.type} ${item.description} ${item.category ?? ''}`.toLowerCase().includes(search.toLowerCase())).map((def) => {
                  const { icon: Icon, color } = getIconFor(def.type)
                  return (
                    <button
                      key={def.type}
                      onClick={() => addBlock(def.type)}
                      title={def.description}
                      className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-xs transition-colors hover:bg-accent text-left"
                    >
                      <div className={`flex h-6 w-6 items-center justify-center rounded ${color}`}>
                        <Icon className="h-3 w-3" />
                      </div>
                      <span className="truncate">{def.type}</span>
                      <Plus className="ml-auto h-3 w-3 text-muted-foreground shrink-0" />
                    </button>
                  )
                })}
              </div>
            </div>

            <div className="space-y-2 pt-2 border-t border-border">
              <Label htmlFor="builder-hidden" className="text-xs">Hidden Size</Label>
              <Input
                id="builder-hidden" type="number"
                value={hiddenSize}
                onChange={(e) => resize({ hidden: Number(e.target.value) })}
                className="h-8"
              />
              <Label htmlFor="builder-vocab" className="text-xs">Vocab Size</Label>
              <Input
                id="builder-vocab" type="number"
                value={vocabSize}
                onChange={(e) => resize({ vocab: Number(e.target.value) })}
                className="h-8"
              />
              <Label htmlFor="builder-positions" className="text-xs">Max Positions</Label>
              <Input
                id="builder-positions" type="number"
                value={maxPositions}
                onChange={(e) => resize({ positions: Number(e.target.value) })}
                className="h-8"
              />
            </div>
          </div>
        </ScrollArea>
      </div>

      <div className="min-h-0 min-w-0 flex-1 flex flex-col">
        <div className="flex flex-col items-stretch px-6 py-3 border-b border-border gap-3">
          <div className="flex min-w-0 flex-wrap items-center gap-3 flex-1">
            <Input
              value={blueprintName}
              onChange={(e) => setBlueprintName(e.target.value)}
              className="max-w-xs h-9"
              placeholder="Blueprint name"
            />
            <Badge variant="secondary">{currentAnalysis ? `${currentAnalysis.parameters.toLocaleString()} params` : 'Parameters: analyze to count'}</Badge>
            <Badge variant="outline">{blocks.length} blocks</Badge>
            <span className="text-[10px] text-muted-foreground">{draft.persisted ? 'Draft saved on this device' : 'Draft storage unavailable: save a blueprint'}</span>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button variant="ghost" size="sm" onClick={draft.undo} disabled={!draft.canUndo || jsonDraft !== null} aria-label="Undo architecture edit"><Undo2 className="h-3.5 w-3.5" /></Button>
            <Button variant="ghost" size="sm" onClick={draft.redo} disabled={!draft.canRedo || jsonDraft !== null} aria-label="Redo architecture edit"><Redo2 className="h-3.5 w-3.5" /></Button>
            <Button variant="outline" size="sm" disabled={!blocks.length || analyzing || jsonDraft !== null} onClick={analyze}><CheckCircle2 className="h-3.5 w-3.5" />{analyzing ? 'Analyzing…' : 'Analyze architecture'}</Button>
            <Button variant="outline" size="sm" onClick={() => { setJsonDraft(JSON.stringify(blueprint, null, 2)); setJsonError(null) }}>
              Edit JSON
            </Button>
            <Button variant="outline" size="sm" onClick={handleTrain} disabled={saving || !blocks.length || jsonDraft !== null}>
              <Play className="h-3.5 w-3.5" />
              {labProfile.task==='llm'?t('builder.train'):`Train in ${labProfile.task.toUpperCase()} Studio`}
            </Button>
            <Button variant="outline" size="sm" onClick={handleCopy}>
              <Copy className="h-3.5 w-3.5" />
              Copy JSON
            </Button>
            <Button size="sm" onClick={handleSave} disabled={saving || jsonDraft !== null}>
              <Save className="h-3.5 w-3.5" />
              {saving ? 'Saving...' : t('builder.save')}
            </Button>
          </div>
        </div>

        <ScrollArea className="flex-1 p-6">
          {analysisError && <p role="alert" className="mb-4 rounded-lg border border-destructive/40 bg-destructive/5 p-3 text-xs text-destructive">{analysisError}</p>}
          <ResearchPanel blueprint={blueprint} name={blueprintName} profile={labProfile} setProfile={patch=>draft.patch({labProfile:{...labProfile,...patch}})} analysis={currentAnalysis} inspect={setSelectedBlock}
            restore={(value,profile,name)=>{draft.patch({blocks:value.blocks as BlockInstance[],hiddenSize:Number(value.hidden_size),vocabSize:Number(value.vocab_size),maxPositions:Number(value.max_position_embeddings),autoProject:Boolean(value.auto_project??true),blueprintExtras:{training:value.training,dtype:value.dtype},blueprintName:name,labProfile:profile});setSelectedBlock(null);setJsonDraft(null)}} />
          {jsonDraft !== null && <div className="mb-4 space-y-2">
            <Label htmlFor="blueprint-json">Blueprint JSON</Label>
            <textarea id="blueprint-json" value={jsonDraft} onChange={event => setJsonDraft(event.target.value)}
              className="w-full min-h-80 rounded border border-border bg-muted/30 p-3 font-mono text-xs" />
            {jsonError && <p role="alert" className="text-xs text-destructive">{jsonError}</p>}
            <div className="flex gap-2">
              <Button size="sm" onClick={applyJson}>Apply JSON</Button>
              <Button size="sm" variant="outline" onClick={() => setJsonDraft(null)}>Cancel</Button>
            </div>
          </div>}
          {blocks.length === 0 ? (
            <div className="flex h-96 items-center justify-center rounded-xl border-2 border-dashed border-border">
              <div className="text-center text-muted-foreground">
                <Layers className="mx-auto h-12 w-12 mb-3 opacity-30" />
                <p className="text-sm">Добавь блоки из палитры слева</p>
                <p className="text-xs mt-1">Или открой blueprint из списка сохранённых</p>
              </div>
            </div>
          ) : (
            <div className="space-y-2 max-w-3xl">
              {blocks.map((block, idx) => {
                const { icon: Icon, color } = getIconFor(block.type)
                const isSelected = selectedBlock === block.id
                return (
                  <Card
                    key={block.id}
                    role="group" tabIndex={0} aria-label={`${block.type} block ${idx + 1}`}
                    onKeyDown={e => { if (e.target === e.currentTarget && ['Enter', ' '].includes(e.key)) { e.preventDefault(); setSelectedBlock(block.id) } }}
                    className={`group cursor-pointer transition-all ${isSelected ? 'ring-2 ring-primary' : ''}`}
                    onClick={() => setSelectedBlock(block.id)}
                  >
                    <CardContent className="p-3 flex items-center gap-3">
                      <div className="flex h-6 w-6 items-center justify-center rounded bg-muted text-xs font-mono text-muted-foreground">
                        {idx}
                      </div>
                      <div className={`flex h-8 w-8 items-center justify-center rounded-md ${color}`}>
                        <Icon className="h-4 w-4" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium">{block.type}</div>
                        <div className="text-xs text-muted-foreground font-mono truncate">
                          {block.id} · {block.type === 'loop' ? `shared body × ${block.params.iterations ?? 4}` : Object.entries(block.params).filter(([, v]) => typeof v !== 'object').map(([k, v]) => `${k}=${v}`).join(', ')}
                        </div>
                        <p className="mt-1 truncate text-[10px] text-muted-foreground" title={(block.inputs.length ? block.inputs : [idx ? blocks[idx - 1].id : 'input']).join(', ')}>← {(block.inputs.length ? block.inputs : [idx ? blocks[idx - 1].id : 'input']).join(', ')}</p>
                      </div>
                      <div className="flex flex-wrap gap-1 opacity-60 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
                        <Button variant="ghost" size="icon" aria-label="Duplicate block" title="Duplicate with independent weights" onClick={e => { e.stopPropagation(); duplicate(block.id) }}><Copy className="h-3 w-3" /></Button>
                        {!['token_embedding', 'positional_embedding', 'add', 'multiply', 'concat'].includes(block.type) && <Button variant="ghost" size="icon" aria-label="Wrap block in loop" title="Wrap in a shared-weight loop" onClick={e => { e.stopPropagation(); wrapInLoop(block.id); setSelectedBlock(block.id) }}><Repeat className="h-3 w-3" /></Button>}
                        <Button variant="ghost" size="icon" aria-label="Move block up" onClick={(e) => { e.stopPropagation(); moveBlock(block.id, -1) }}>
                          <ArrowUp className="h-3 w-3" />
                        </Button>
                        <Button variant="ghost" size="icon" aria-label="Move block down" onClick={(e) => { e.stopPropagation(); moveBlock(block.id, 1) }}>
                          <ArrowDown className="h-3 w-3" />
                        </Button>
                        <Button variant="ghost" size="icon" aria-label="Remove block" onClick={(e) => { e.stopPropagation(); removeBlock(block.id) }}>
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                )
              })}
            </div>
          )}
        </ScrollArea>
      </div>

      <div className="builder-saved min-h-0 w-[328px] shrink-0 border-l border-border flex flex-col">
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{currentBlock ? 'Block inspector' : 'Saved Blueprints'}</h3>
          <Button variant="ghost" size="icon" aria-label={currentBlock ? 'Close block inspector' : 'Refresh blueprints'} onClick={() => currentBlock ? setSelectedBlock(null) : loadSaved()}>{currentBlock ? <X className="h-4 w-4" /> : <RefreshCw className="h-3.5 w-3.5" />}</Button>
        </div>
        <ScrollArea className="min-h-0 flex-1">
          {currentBlock ? <div className="space-y-4 p-4">
            <BlockEditor block={currentBlock} definitions={available} dimensions={dimensions} onChange={value => setBlocks(current => current.map(b => b.id === currentBlock.id ? { ...b, ...value } : b))} />
            <fieldset className="space-y-2 border-t border-border pt-3"><legend className="text-xs font-semibold">Inputs · earlier blocks</legend>
              <p className="text-[10px] text-muted-foreground">No explicit inputs: use the previous block. Graph joins need at least two branches.</p>
              {['input', ...blocks.slice(0, blocks.findIndex(b => b.id === currentBlock.id)).map(b => b.id)].map(parent => <Label key={parent} className="flex min-w-0 items-center gap-2 text-[11px]">
                <input type="checkbox" checked={currentBlock.inputs.includes(parent)} onChange={e => setBlocks(current => current.map(b => b.id === currentBlock.id ? { ...b, inputs: e.target.checked ? [...b.inputs, parent] : b.inputs.filter(id => id !== parent) } : b))} /><span className="truncate" title={parent}>{parent}</span>
              </Label>)}
            </fieldset>
          </div> : <div className="space-y-1 p-3">
            {saved.length === 0 && <p className="py-4 text-center text-xs text-muted-foreground">No saved models yet</p>}
            {saved.map(item => <div key={item.path} className="group flex items-start gap-2 rounded-md p-2 hover:bg-accent">
              <FileText className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" /><button className="min-w-0 flex-1 text-left" onClick={() => { handleLoad(item.name); setSelectedBlock(null) }}><div className="truncate text-xs font-medium">{item.name}</div><div className="text-[10px] text-muted-foreground">{item.num_blocks} blocks · h={item.hidden_size}</div></button>
              <Button variant="ghost" size="icon" className="h-6 w-6" aria-label={`Delete ${item.name}`} onClick={() => handleDelete(item.name)}><Trash2 className="h-3 w-3" /></Button>
            </div>)}
          </div>}
        </ScrollArea>
        <details className="border-t border-border"><summary className="cursor-pointer px-3 py-2 text-xs font-semibold text-muted-foreground">JSON Preview</summary><pre className="max-h-72 overflow-auto bg-muted/50 p-3 font-mono text-[10px] text-muted-foreground">{JSON.stringify(blueprint, null, 2)}</pre></details>
      </div>
    </div>
  )
}
