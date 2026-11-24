import { defaultLabProfile, type LabProfile } from '@/lib/research'
import { useEffect, useReducer, useState, type SetStateAction } from 'react'
import type { BuilderBlock } from '@/lib/builder'

export interface BuilderDraft {
  blocks: BuilderBlock[]; blueprintName: string; hiddenSize: number; vocabSize: number; maxPositions: number
  labProfile: LabProfile; autoProject: boolean; blueprintExtras: Record<string, unknown>
}
export const emptyDraft: BuilderDraft = { blocks: [], blueprintName: 'my_model', hiddenSize: 512, vocabSize: 50257, maxPositions: 2048, labProfile: defaultLabProfile, autoProject: true, blueprintExtras: {} }
interface History { present: BuilderDraft; past: BuilderDraft[]; future: BuilderDraft[] }
type Action = { type: 'edit'; update: (value: BuilderDraft) => BuilderDraft } | { type: 'undo' | 'redo' }
export function builderHistory(state: History, action: Action): History {
  if (action.type === 'undo') return state.past.length ? { present: state.past[state.past.length - 1], past: state.past.slice(0, -1), future: [state.present, ...state.future] } : state
  if (action.type === 'redo') return state.future.length ? { present: state.future[0], past: [...state.past, state.present].slice(-50), future: state.future.slice(1) } : state
  if (action.type !== 'edit') return state
  const present = action.update(state.present)
  return { present, past: [...state.past, state.present].slice(-50), future: [] }
}
function readDraft(): History {
  try {
    const value = JSON.parse(localStorage.getItem('mah-builder-draft') ?? 'null')
    if (value && Array.isArray(value.blocks) && value.blocks.length <= 256 && value.blocks.every((b: BuilderBlock) => typeof b.id === 'string' && typeof b.type === 'string' && b.params && typeof b.params === 'object' && Array.isArray(b.inputs)) && ['hiddenSize', 'vocabSize', 'maxPositions'].every(key => Number.isInteger(value[key]) && value[key] > 0) && typeof value.blueprintName === 'string') {
      return { present: { ...emptyDraft, ...value }, past: [], future: [] }
    }
  } catch { /* A broken draft must not prevent opening the builder. */ }
  return { present: structuredClone(emptyDraft), past: [], future: [] }
}
export function useBuilderDraft() {
  const [history, dispatch] = useReducer(builderHistory, undefined, readDraft)
  const [persisted, setPersisted] = useState(true)
  useEffect(() => {
    try { localStorage.setItem('mah-builder-draft', JSON.stringify(history.present)); setPersisted(true) }
    catch { setPersisted(false) }
  }, [history.present])
  function setField<K extends keyof BuilderDraft>(key: K, value: SetStateAction<BuilderDraft[K]>) {
    dispatch({ type: 'edit', update: current => ({ ...current, [key]: typeof value === 'function' ? (value as (old: BuilderDraft[K]) => BuilderDraft[K])(current[key]) : value }) })
  }
  const patch = (update: Partial<BuilderDraft>) => dispatch({ type: 'edit', update: current => ({ ...current, ...update }) })
  return { ...history.present, setField, patch, persisted, canUndo: !!history.past.length, canRedo: !!history.future.length,
    undo: () => dispatch({ type: 'undo' }), redo: () => dispatch({ type: 'redo' }) }
}
