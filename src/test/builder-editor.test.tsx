import { useState } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import '@/i18n'
import { BlockEditor } from '@/components/builder/BlockEditor'
import { duplicateSequenceBlock, initialParams, recipe, type BlockDefinition, type BodyBlock } from '@/lib/builder'
import { builderHistory, emptyDraft } from '@/stores/builderDraft'

const definitions: BlockDefinition[] = [
  { type: 'loop', description: 'Shared body', defaults: { iterations: 4, eval_iterations: 4, input_injection: 'none', checkpoint: false, body: [{ type: 'llama_block', params: { num_heads: 8, intermediate_size: 2048 } }] } },
  { type: 'llama_block', description: '', defaults: { num_heads: 8, intermediate_size: 2048 } },
  { type: 'mlp', description: '', defaults: { intermediate_size: 2048, activation: 'gelu' } },
  { type: 'rmsnorm', description: '', defaults: { eps: 1e-6 } },
  { type: 'token_embedding', description: '', defaults: { hidden_size: 512, vocab_size: 50257 } },
]
const dimensions = { hidden: 64, vocab: 100, positions: 32 }
function Editor() {
  const [block, setBlock] = useState<BodyBlock>(() => ({ type: 'loop', params: initialParams('loop', dimensions, definitions) }))
  return <><BlockEditor block={block} definitions={definitions} dimensions={dimensions} onChange={setBlock} /><output data-testid="spec">{JSON.stringify(block)}</output></>
}
const read = () => JSON.parse(screen.getByTestId('spec').textContent ?? '{}')

describe('composable block editor', () => {
  it('edits shared depth and body type without replacing the surrounding loop', () => {
    render(<Editor />)
    fireEvent.change(screen.getByLabelText('iterations'), { target: { value: '7' } })
    fireEvent.change(screen.getByLabelText('input_injection'), { target: { value: 'concat' } })
    fireEvent.change(screen.getAllByLabelText('Block type')[1], { target: { value: 'mlp' } })
    const result = read()
    expect(result.type).toBe('loop')
    expect(result.params.iterations).toBe(7)
    expect(result.params.input_injection).toBe('concat')
    expect(result.params.body[0]).toEqual({ type: 'mlp', params: { intermediate_size: 256, activation: 'gelu' } })
  })

  it('adds, duplicates, moves and removes body blocks', () => {
    render(<Editor />)
    fireEvent.change(screen.getByLabelText('Add body block'), { target: { value: 'rmsnorm' } })
    fireEvent.click(screen.getAllByRole('button', { name: 'Duplicate body block' })[0])
    expect(read().params.body.map((b: BodyBlock) => b.type)).toEqual(['llama_block', 'llama_block', 'rmsnorm'])
    fireEvent.click(screen.getAllByRole('button', { name: 'Move body block up' })[2])
    expect(read().params.body[1].type).toBe('rmsnorm')
    fireEvent.click(screen.getAllByRole('button', { name: 'Remove body block' })[0])
    expect(read().params.body.map((b: BodyBlock) => b.type)).toEqual(['rmsnorm', 'llama_block'])
  })

  it('does not apply a broken parameter JSON draft', () => {
    render(<Editor />)
    const field = screen.getAllByLabelText('Parameters JSON').at(-1)!
    fireEvent.change(field, { target: { value: '{broken' } })
    fireEvent.click(screen.getAllByRole('button', { name: 'Apply parameters' }).at(-1)!)
    expect(screen.getByRole('alert')).toBeInTheDocument()
    expect(read().params.iterations).toBe(4)
  })
})

it('duplicates sequential layers with independent editable specifications and rewires successors', () => {
  const blocks = recipe('looped', dimensions, definitions)
  const duplicated = duplicateSequenceBlock(blocks, blocks[1].id)
  expect(duplicated).toHaveLength(blocks.length + 1)
  expect(duplicated[2].id).not.toBe(blocks[1].id)
  expect(duplicated[2].params).not.toBe(blocks[1].params)
  expect(duplicated[3].inputs).toEqual([duplicated[2].id])
  expect(blocks[2].inputs).toEqual([blocks[1].id])
})

it('undoes an entire architecture edit including dimensions and invalidates redo after a new edit', () => {
  const before = { present: structuredClone(emptyDraft), past: [], future: [] }
  const edited = builderHistory(before, { type: 'edit', update: current => ({ ...current, hiddenSize: 64, blocks: recipe('looped', dimensions, definitions) }) })
  const undone = builderHistory(edited, { type: 'undo' })
  expect(undone.present).toEqual(before.present)
  expect(builderHistory(undone, { type: 'redo' }).present).toEqual(edited.present)
  const branched = builderHistory(undone, { type: 'edit', update: current => ({ ...current, blueprintName: 'alternative' }) })
  expect(branched.future).toHaveLength(0)
})
