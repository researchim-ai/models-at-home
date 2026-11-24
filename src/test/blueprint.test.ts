import { expect, it } from 'vitest'
import { connectSequence, isSequence, syncBlueprintDimensions } from '@/lib/blueprint'

it('rebuilds connections after moving or removing a sequential block', () => {
  const original = [{ id: 'a', inputs: [] }, { id: 'b', inputs: ['a'] }, { id: 'c', inputs: ['b'] }]
  expect(isSequence(original)).toBe(true)
  const moved = connectSequence([original[1], original[0], original[2]])
  expect(moved.map(b => b.inputs)).toEqual([[], ['b'], ['a']])
  expect(connectSequence(original.filter(b => b.id !== 'b')).map(b => b.inputs)).toEqual([[], ['a']])
  expect(original[2].inputs).toEqual(['b'])
})

it('synchronizes default dimensions while preserving custom widths', () => {
  const blocks: Array<{ id: string; inputs: string[]; type: string; params: Record<string, number | string | boolean> }> = [
    { id: 'embed', inputs: [], type: 'token_embedding', params: { hidden_size: 512, vocab_size: 100 } },
    { id: 'ffn', inputs: ['embed'], type: 'mlp', params: { intermediate_size: 2048 } },
    { id: 'custom', inputs: ['ffn'], type: 'linear', params: { in_features: 512, out_features: 96 } },
  ]
  const next = syncBlueprintDimensions(blocks, { hidden: 512, vocab: 100, positions: 64 }, { hidden: 256, vocab: 200, positions: 128 })
  expect(next[0].params).toEqual({ hidden_size: 256, vocab_size: 200 })
  expect(next[1].params.intermediate_size).toBe(1024)
  expect(next[2].params).toEqual({ in_features: 256, out_features: 96 })
  expect(blocks[0].params.hidden_size).toBe(512)
})

it('recognizes graphs that cannot safely be reordered as a sequence', () => {
  expect(isSequence([{ id: 'a', inputs: [] }, { id: 'b', inputs: [] }, { id: 'c', inputs: ['a', 'b'] }])).toBe(false)
})
