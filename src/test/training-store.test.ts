import { expect, it } from 'vitest'
import { useTrainingStore } from '@/stores/trainingStore'

it('keeps monitoring history while revisiting the same run and clears it for a different run', () => {
  const store = useTrainingStore.getState()
  store.setActiveRunId('first')
  store.pushMetrics({ step: 1, loss: 0.5, learning_rate: 0.01, timestamp: 1 })
  store.setActiveRunId('first')
  expect(useTrainingStore.getState().metricsHistory).toHaveLength(1)
  store.pushMetrics({ step: 1, loss: 0.4, learning_rate: 0.01, timestamp: 2 })
  expect(useTrainingStore.getState().metricsHistory).toHaveLength(1)
  expect(useTrainingStore.getState().currentMetrics?.loss).toBe(0.4)
  store.setActiveRunId('second')
  expect(useTrainingStore.getState().metricsHistory).toHaveLength(0)
  store.setActiveRunId(null)
})

it('hydrates disk history without overwriting newer live metrics or another selected run', () => {
  const store = useTrainingStore.getState()
  const point = (step: number, timestamp: number, loss: number) => ({ step, timestamp, loss, learning_rate: 0.01 })
  store.setActiveRunId('restore')
  store.pushMetrics(point(3, 30, 0.3))
  store.hydrateMetrics('restore', [point(1, 0, 0.8), point(2, 0, 0.6)], point(3, 20, 0.4))
  expect(useTrainingStore.getState().metricsHistory.map(m => m.step)).toEqual([1, 2, 3])
  expect(useTrainingStore.getState().currentMetrics?.loss).toBe(0.3)
  store.hydrateMetrics('restore', [], point(3, 40, 0.2))
  expect(useTrainingStore.getState().metricsHistory.at(-1)?.loss).toBe(0.2)
  store.pushMetrics(point(2, 50, 0.7))
  expect(useTrainingStore.getState().currentMetrics?.step).toBe(3)
  store.setActiveRunId('different')
  store.hydrateMetrics('restore', [point(1, 0, 1)], point(3, 60, 0.1))
  expect(useTrainingStore.getState().metricsHistory).toEqual([])
  store.setActiveRunId(null)
})
