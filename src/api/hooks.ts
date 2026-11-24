import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from './client'
import type { HFDownloadRequest, HFDatasetDownloadRequest } from './types'

export function useSystemHealth() {
  return useQuery({
    queryKey: ['system', 'health'],
    queryFn: api.health,
    refetchInterval: 10000,
  })
}

export function useGpuInfo() {
  return useQuery({
    queryKey: ['system', 'gpu'],
    queryFn: api.gpuInfo,
    refetchInterval: 5000,
  })
}

export function useConfigs() {
  return useQuery({
    queryKey: ['system', 'configs'],
    queryFn: api.configs,
  })
}

export function useTrainingRuns() {
  return useQuery({
    queryKey: ['training', 'runs'],
    queryFn: api.listRuns,
    refetchInterval: 5000,
  })
}

export function useTrainingRun(runId: string | null) {
  return useQuery({
    queryKey: ['training', 'run', runId],
    queryFn: () => api.getRun(runId!),
    enabled: !!runId,
    refetchInterval: 3000,
  })
}

export function useStartTraining() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: api.startTraining,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['training', 'runs'] }),
  })
}

export function useStopTraining() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: api.stopTraining,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['training', 'runs'] }),
  })
}

export function useDeleteRun() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: api.deleteRun,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['training', 'runs'] }),
  })
}

export function useContinueTraining() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (runId: string) => api.continueTraining(runId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['training', 'runs'] }),
  })
}

export function useRunLogs(runId: string | null, enabled = true) {
  return useQuery({
    queryKey: ['training', 'logs', runId],
    queryFn: () => api.runLogs(runId!),
    enabled: !!runId && enabled,
    refetchInterval: 3000,
  })
}

export function useModels() {
  return useQuery({
    queryKey: ['models', 'local'],
    queryFn: api.listModels,
  })
}

export function useTrainedModels() {
  return useQuery({
    queryKey: ['models', 'trained'],
    queryFn: api.listTrainedModels,
  })
}

export function useDownloadModel() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: HFDownloadRequest) => api.downloadModel(data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['models'] }),
  })
}

export function useDeleteLocalModel() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => api.deleteLocalModel(name),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['models'] }),
  })
}

export function useDatasets() {
  return useQuery({
    queryKey: ['datasets', 'local'],
    queryFn: api.listDatasets,
  })
}

export function useDownloadDataset() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: HFDatasetDownloadRequest) => api.downloadDataset(data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['datasets'] }),
  })
}

export function useDeleteDataset() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => api.deleteDataset(name),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['datasets'] }),
  })
}

export function useDatasetPreview(name: string | null) {
  return useQuery({
    queryKey: ['datasets', 'preview', name],
    queryFn: () => api.previewDataset(name!),
    enabled: !!name,
  })
}

export function useEstimateMemory() {
  return useMutation({
    mutationFn: api.estimateMemory,
  })
}

export function useAgentServerStatus() {
  return useQuery({
    queryKey: ['agent', 'server-status'],
    queryFn: api.agentServerStatus,
    refetchInterval: 5000,
  })
}

export function useAgentSessions() {
  return useQuery({
    queryKey: ['agent', 'sessions'],
    queryFn: api.listAgentSessions,
  })
}

export function useAgentTools() {
  return useQuery({
    queryKey: ['agent', 'tools'],
    queryFn: api.agentTools,
    staleTime: 60_000,
  })
}

export function useAgentCapabilities() {
  return useQuery({
    queryKey: ['agent', 'capabilities'],
    queryFn: api.agentCapabilities,
    staleTime: 60_000,
  })
}

export function useAgentRuns() {
  return useQuery({
    queryKey: ['agent', 'runs'],
    queryFn: () => api.agentRuns('all'),
    refetchInterval: 5000,
  })
}

export function useVLMRuns() {
  return useQuery({
    queryKey: ['vlm', 'runs'],
    queryFn: api.listVLMRuns,
    refetchInterval: 5000,
  })
}
