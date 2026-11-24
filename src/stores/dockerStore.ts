import { create } from 'zustand'
import type { BackendMode, BootPhase } from '@/types/electron'

interface DockerState {
  status: 'unknown' | 'running' | 'stopped' | 'building' | 'error'
  backendOnline: boolean
  backendPort: number
  backendError: string | null
  backendMode: 'native' | 'docker' | 'offline'
  bootPhase: BootPhase
  containerId: string | null
  error: string | null
  config: {
    backendMode: BackendMode
    dockerAutoBuild: boolean
    dockerGpu: boolean
  } | null

  setStatus: (status: DockerState['status']) => void
  setBackendOnline: (online: boolean) => void
  setError: (error: string | null) => void
  setBootPhase: (phase: BootPhase) => void

  checkBackend: () => Promise<void>
  checkDocker: () => Promise<void>
  startDocker: () => Promise<void>
  stopDocker: () => Promise<void>
  restartBackend: () => Promise<void>
  loadConfig: () => Promise<void>
  setBackendMode: (mode: BackendMode) => Promise<void>
  subscribeToBootPhase: () => () => void
}

export const useDockerStore = create<DockerState>((set) => ({
  status: 'unknown',
  backendOnline: false,
  backendPort: 8000,
  backendError: null,
  backendMode: 'offline',
  bootPhase: { phase: 'starting' },
  containerId: null,
  error: null,
  config: null,

  setStatus: (status) => set({ status }),
  setBackendOnline: (backendOnline) => set({ backendOnline }),
  setError: (error) => set({ error }),
  setBootPhase: (bootPhase) => set({ bootPhase }),

  checkBackend: async () => {
    if (window.electronAPI) {
      try {
        const result = await window.electronAPI.backend.status()
        set({
          backendOnline: result.running,
          backendPort: result.port ?? 8000,
          backendError: result.error ?? null,
          backendMode: result.mode ?? 'offline',
          bootPhase: result.phase ?? { phase: 'starting' },
        })
      } catch {
        set({ backendOnline: false })
      }
    } else {
      try {
        const res = await fetch('/api/system/health')
        set({ backendOnline: res.ok })
      } catch {
        set({ backendOnline: false })
      }
    }
  },

  checkDocker: async () => {
    if (window.electronAPI) {
      const result = await window.electronAPI.docker.status()
      set({
        status: result.running ? 'running' : 'stopped',
        containerId: result.containerId ?? null,
        error: result.error ?? null,
      })
    }
  },

  startDocker: async () => {
    set({ status: 'building' })
    if (window.electronAPI) {
      const result = await window.electronAPI.docker.start()
      set({ status: result.success ? 'running' : 'error', error: result.error ?? null })
    }
  },

  stopDocker: async () => {
    if (window.electronAPI) {
      await window.electronAPI.docker.stop()
      set({ status: 'stopped' })
    }
  },

  restartBackend: async () => {
    if (window.electronAPI) {
      set({ backendOnline: false, backendError: null, bootPhase: { phase: 'starting' } })
      const result = await window.electronAPI.backend.restart()
      set({
        backendOnline: result.success,
        backendPort: result.port ?? 8000,
        backendError: result.error ?? null,
      })
    }
  },

  loadConfig: async () => {
    if (window.electronAPI) {
      const cfg = await window.electronAPI.config.get()
      set({ config: cfg })
    }
  },

  setBackendMode: async (mode) => {
    if (!window.electronAPI) return
    set({ backendOnline: false, bootPhase: { phase: 'starting' } })
    const cfg = await window.electronAPI.config.setBackendMode(mode)
    set({ config: cfg })
  },

  subscribeToBootPhase: () => {
    if (!window.electronAPI) return () => {}
    return window.electronAPI.backend.onBootPhase((phase) => {
      set({ bootPhase: phase })
      if (phase.phase === 'ready') {
        set({
          backendOnline: true,
          backendPort: phase.port,
          backendMode: phase.mode,
          backendError: null,
        })
      } else if (phase.phase === 'failed') {
        set({ backendOnline: false, backendError: phase.error })
      }
    })
  },
}))
