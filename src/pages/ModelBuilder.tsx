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
} from 'lucide-react'
import { api } from '@/api/client'
import { isSequence, connectSequence, syncBlueprintDimensions } from '@/lib/blueprint'
import { toast } from 'sonner'
import { useTrainingStore } from '@/stores/trainingStore'

interface BlockDef {
  type: string
  description: string
}

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
  dropout: { icon: Shuffle, color: 'bg-gray-500/20 text-gray-400' },
}

function getIconFor(type: string) {
  return ICON_BY_TYPE[type] ?? { icon: Cpu, color: 'bg-muted text-muted-foreground' }
}

function getDefaultParams(type: string, hiddenSize: number, vocabSize: number, maxPositions: number): Record<string, number | string | boolean> {
  switch (type.replace(/^liger_/, '')) {
    case 'token_embedding': return { vocab_size: vocabSize, hidden_size: hiddenSize }
    case 'positional_embedding': return { max_position_embeddings: maxPositions, hidden_size: hiddenSize }
    case 'attention': return { num_heads: 8, dropout: 0 }
    case 'causal_self_attention': return { num_heads: 8, use_rope: true }
    case 'mlp': return { intermediate_size: hiddenSize * 4, activation: 'gelu' }
    case 'swiglu': return { intermediate_size: hiddenSize * 4 }
    case 'rmsnorm': return { eps: 1e-6 }
    case 'layernorm': return { eps: 1e-5 }
    case 'moe': return { num_experts: 8, num_select: 2, expert_type: 'swiglu' }
    case 'llama_block': return { num_heads: 8, intermediate_size: hiddenSize * 4 }
    case 'linear': return { in_features: hiddenSize, out_features: hiddenSize, bias: false }
    case 'dropout': return { p: 0.1 }
    default: return {}
  }
}

export function ModelBuilder() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const [blocks, setBlocks] = useState<BlockInstance[]>([])
  const [blueprintName, setBlueprintName] = useState('my_model')
  const [hiddenSize, setHiddenSize] = useState(512)
  const [vocabSize, setVocabSize] = useState(50257)
  const [maxPositions, setMaxPositions] = useState(2048)
  const [autoProject, setAutoProject] = useState(true)

  const [available, setAvailable] = useState<BlockDef[]>([])
  const [saved, setSaved] = useState<SavedBlueprint[]>([])
  const [selectedBlock, setSelectedBlock] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [jsonDraft, setJsonDraft] = useState<string | null>(null)
  const [jsonError, setJsonError] = useState<string | null>(null)
  const [blueprintExtras, setBlueprintExtras] = useState<Record<string, unknown>>({})

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
        params: getDefaultParams(type, hiddenSize, vocabSize, maxPositions),
        inputs: previousId ? [previousId] : [],
      }]
    })
  }, [hiddenSize, vocabSize, maxPositions])

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

  const updateParam = (blockId: string, key: string, value: string | number | boolean) => {
    setBlocks((prev) => prev.map((b) =>
      b.id === blockId ? { ...b, params: { ...b.params, [key]: value } } : b,
    ))
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

  const totalParams = blocks.reduce((acc, b) => {
    if (b.type === 'token_embedding') return acc + vocabSize * hiddenSize
    if (b.type === 'positional_embedding') return acc + maxPositions * hiddenSize
    if (b.type === 'attention' || b.type === 'causal_self_attention') return acc + 4 * hiddenSize * hiddenSize
    if (b.type === 'mlp' || b.type === 'swiglu' || b.type === 'liger_swiglu')
      return acc + 3 * hiddenSize * hiddenSize * 4
    if (b.type === 'llama_block' || b.type === 'liger_llama_block')
      return acc + 4 * hiddenSize * hiddenSize + 3 * hiddenSize * hiddenSize * 4 + 2 * hiddenSize
    if (b.type === 'linear') return acc + Number(b.params.in_features ?? hiddenSize) * Number(b.params.out_features ?? hiddenSize)
    if (b.type === 'moe') return acc + Number(b.params.num_experts ?? 8) * hiddenSize * hiddenSize * 4 * 3
    return acc
  }, 0)

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
      const result = await api.saveBlueprint(blueprintName, blueprint)
      const store = useTrainingStore.getState()
      store.setStage('pretrain')
      store.updateConfig({ model_type: 'blueprint', model_blueprint: result.path,
        model_blueprint_spec: blueprint, hidden_size: hiddenSize, vocab_size: vocabSize,
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
      setBlueprintName(name)
      setHiddenSize(bp.hidden_size ?? 512)
      setVocabSize(bp.vocab_size ?? 50257)
      setMaxPositions(bp.max_position_embeddings ?? 2048)
      setAutoProject(bp.auto_project ?? true)
      setBlueprintExtras({ dtype: bp.dtype, training: bp.training })
      setBlocks((bp.blocks ?? []).map((b: any) => ({
        id: b.id,
        type: b.type,
        params: b.params ?? {},
        inputs: b.inputs ?? [],
      })))
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
      setHiddenSize(value.hidden_size as number)
      setVocabSize(value.vocab_size as number)
      setMaxPositions(value.max_position_embeddings as number)
      setAutoProject(value.auto_project as boolean)
      setBlueprintExtras({ dtype: value.dtype, training: value.training })
      setBlocks(value.blocks as BlockInstance[])
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
    setBlocks(current => syncBlueprintDimensions(current, before, after))
    setHiddenSize(after.hidden)
    setVocabSize(after.vocab)
    setMaxPositions(after.positions)
  }

  return (
    <div className="flex h-full">
      <div className="w-64 shrink-0 border-r border-border flex flex-col">
        <ScrollArea className="flex-1 p-4">
          <div className="space-y-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  {t('builder.blocks')}
                </h3>
                <Button variant="ghost" size="icon" onClick={loadBlocks}>
                  <RefreshCw className="h-3 w-3" />
                </Button>
              </div>
              <div className="space-y-1">
                {available.map((def) => {
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
              <Label className="text-xs">Hidden Size</Label>
              <Input
                type="number"
                value={hiddenSize}
                onChange={(e) => resize({ hidden: Number(e.target.value) })}
                className="h-8"
              />
              <Label className="text-xs">Vocab Size</Label>
              <Input
                type="number"
                value={vocabSize}
                onChange={(e) => resize({ vocab: Number(e.target.value) })}
                className="h-8"
              />
              <Label className="text-xs">Max Positions</Label>
              <Input
                type="number"
                value={maxPositions}
                onChange={(e) => resize({ positions: Number(e.target.value) })}
                className="h-8"
              />
            </div>
          </div>
        </ScrollArea>
      </div>

      <div className="flex-1 flex flex-col">
        <div className="flex items-center justify-between px-6 py-3 border-b border-border gap-3">
          <div className="flex items-center gap-3 flex-1">
            <Input
              value={blueprintName}
              onChange={(e) => setBlueprintName(e.target.value)}
              className="max-w-xs h-9"
              placeholder="Blueprint name"
            />
            <Badge variant="secondary">
              ~{totalParams >= 1e9 ? `${(totalParams / 1e9).toFixed(2)}B` : `${(totalParams / 1e6).toFixed(1)}M`} params
            </Badge>
            <Badge variant="outline">{blocks.length} blocks</Badge>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => { setJsonDraft(JSON.stringify(blueprint, null, 2)); setJsonError(null) }}>
              Edit JSON
            </Button>
            <Button variant="outline" size="sm" onClick={handleTrain} disabled={saving || !blocks.length || jsonDraft !== null}>
              <Play className="h-3.5 w-3.5" />
              {t('builder.train')}
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
                <p className="text-xs mt-1">Или выбери сохранённый blueprint справа</p>
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
                          {block.id} · {Object.entries(block.params).map(([k, v]) => `${k}=${v}`).join(', ')}
                        </div>
                      </div>
                      <div className="opacity-0 group-hover:opacity-100 transition-opacity flex gap-1">
                        <Button variant="ghost" size="icon" onClick={(e) => { e.stopPropagation(); moveBlock(block.id, -1) }}>
                          <ArrowUp className="h-3 w-3" />
                        </Button>
                        <Button variant="ghost" size="icon" onClick={(e) => { e.stopPropagation(); moveBlock(block.id, 1) }}>
                          <ArrowDown className="h-3 w-3" />
                        </Button>
                        <Button variant="ghost" size="icon" onClick={(e) => { e.stopPropagation(); removeBlock(block.id) }}>
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

      <div className="w-72 shrink-0 border-l border-border flex flex-col">
        <div className="px-4 py-3 border-b border-border flex items-center justify-between">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground flex items-center gap-2">
            <FolderOpen className="h-3.5 w-3.5" />
            Saved Blueprints
          </h3>
          <Button variant="ghost" size="icon" onClick={loadSaved}>
            <RefreshCw className="h-3 w-3" />
          </Button>
        </div>

        <ScrollArea className="flex-1">
          <div className="p-3 space-y-1">
            {saved.length === 0 && (
              <p className="text-xs text-muted-foreground text-center py-4">
                Нет сохранённых
              </p>
            )}
            {saved.map((s) => (
              <div
                key={s.path}
                className="group flex items-start gap-2 rounded-md p-2 hover:bg-accent transition-colors"
              >
                <FileText className="h-3.5 w-3.5 mt-0.5 shrink-0 text-muted-foreground" />
                <button
                  className="flex-1 min-w-0 text-left"
                  onClick={() => handleLoad(s.name)}
                >
                  <div className="text-xs font-medium truncate">{s.name}</div>
                  <div className="text-[10px] text-muted-foreground">
                    {s.num_blocks} blocks · h={s.hidden_size}
                  </div>
                </button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="opacity-0 group-hover:opacity-100 transition-opacity h-6 w-6"
                  onClick={() => handleDelete(s.name)}
                >
                  <Trash2 className="h-3 w-3" />
                </Button>
              </div>
            ))}
          </div>

          {currentBlock && (
            <div className="border-t border-border p-3 space-y-2">
              <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Params: {currentBlock.type}
              </div>
              {Object.entries(currentBlock.params).map(([k, v]) => (
                <div key={k} className="space-y-1">
                  <Label className="text-[10px]">{k}</Label>
                  {typeof v === 'boolean' ? <input
                    type="checkbox" checked={v} aria-label={k}
                    onChange={e => updateParam(currentBlock.id, k, e.target.checked)}
                  /> : typeof v === 'object' ? <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(v, null, 2)}</pre> : <Input
                    value={String(v)}
                    onChange={(e) => {
                      const val = e.target.value
                      const parsed = val === '' ? '' : isNaN(Number(val)) ? val : Number(val)
                      updateParam(currentBlock.id, k, parsed)
                    }}
                    className="h-7 text-xs"
                  />}
                </div>
              ))}
            </div>
          )}

          <details className="border-t border-border">
            <summary className="px-3 py-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground cursor-pointer hover:bg-accent/50">
              JSON Preview
            </summary>
            <pre className="text-[10px] font-mono text-muted-foreground bg-muted/50 p-3 overflow-auto max-h-72">
              {JSON.stringify(blueprint, null, 2)}
            </pre>
          </details>
        </ScrollArea>
      </div>
    </div>
  )
}
