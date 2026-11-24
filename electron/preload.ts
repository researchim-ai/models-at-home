import { contextBridge, ipcRenderer, type IpcRendererEvent } from 'electron'

interface OpenDialogResult {
  canceled: boolean
  filePaths: string[]
}

interface FileFilter {
  name: string
  extensions: string[]
}

export type BackendMode = 'native' | 'docker' | 'auto'

export interface AppConfigShape {
  backendMode: BackendMode
  dockerAutoBuild: boolean
  dockerGpu: boolean
}

export type BootPhase =
  | { phase: 'starting' }
  | { phase: 'checking-docker' }
  | { phase: 'docker-unavailable'; detail?: string }
  | { phase: 'docker-no-image' }
  | { phase: 'building-image'; line?: string }
  | { phase: 'starting-container' }
  | { phase: 'waiting-container-health'; attempt: number }
  | { phase: 'starting-python' }
  | { phase: 'python-starting'; line?: string }
  | { phase: 'ready'; mode: 'native' | 'docker'; port: number }
  | { phase: 'failed'; error: string }

type Unsubscribe = () => void

const api = {
  backend: {
    status: () =>
      ipcRenderer.invoke('backend:status') as Promise<{
        running: boolean
        port: number
        error?: string | null
        mode?: 'native' | 'docker' | 'offline'
        phase?: BootPhase
      }>,
    restart: () =>
      ipcRenderer.invoke('backend:restart') as Promise<{
        success: boolean
        error?: string
        port: number
        mode?: string
      }>,
    port: () => ipcRenderer.invoke('backend:port') as Promise<number>,
    token: () => ipcRenderer.invoke('backend:token') as Promise<string>,
    bootPhase: () => ipcRenderer.invoke('backend:boot-phase') as Promise<BootPhase>,
    openLogs: () => ipcRenderer.invoke('backend:open-logs') as Promise<void>,
    onBootPhase: (cb: (phase: BootPhase) => void): Unsubscribe => {
      const handler = (_e: IpcRendererEvent, phase: BootPhase) => cb(phase)
      ipcRenderer.on('backend:boot-phase', handler)
      return () => ipcRenderer.removeListener('backend:boot-phase', handler)
    },
  },
  config: {
    get: () => ipcRenderer.invoke('config:get') as Promise<AppConfigShape>,
    set: (patch: Partial<AppConfigShape>) =>
      ipcRenderer.invoke('config:set', patch) as Promise<AppConfigShape>,
    setBackendMode: (mode: BackendMode) =>
      ipcRenderer.invoke('config:set-backend-mode', mode) as Promise<AppConfigShape>,
  },
  docker: {
    status: () =>
      ipcRenderer.invoke('docker:status') as Promise<{
        running: boolean
        containerId?: string
        hostPort?: number
        error?: string
      }>,
    available: () => ipcRenderer.invoke('docker:available') as Promise<boolean>,
    imageExists: () => ipcRenderer.invoke('docker:image-exists') as Promise<boolean>,
    start: () =>
      ipcRenderer.invoke('docker:start') as Promise<{
        success: boolean
        error?: string
        hostPort?: number
      }>,
    stop: () => ipcRenderer.invoke('docker:stop') as Promise<{ success: boolean; error?: string }>,
    build: () => ipcRenderer.invoke('docker:build') as Promise<{ success: boolean; error?: string }>,
    logs: (tail?: number) => ipcRenderer.invoke('docker:logs', tail) as Promise<string>,
    onBuildProgress: (cb: (line: string) => void): Unsubscribe => {
      const handler = (_e: IpcRendererEvent, line: string) => cb(line)
      ipcRenderer.on('docker:build-progress', handler)
      return () => ipcRenderer.removeListener('docker:build-progress', handler)
    },
  },
  app: {
    version: () => ipcRenderer.invoke('app:version') as Promise<string>,
    platform: () => ipcRenderer.invoke('app:platform') as Promise<NodeJS.Platform>,
    paths: () =>
      ipcRenderer.invoke('app:paths') as Promise<{
        userData: string
        logs: string
        temp: string
      }>,
  },
  shell: {
    openExternal: (url: string) => ipcRenderer.invoke('shell:openExternal', url) as Promise<void>,
    showInFolder: (p: string) => ipcRenderer.invoke('shell:showInFolder', p) as Promise<void>,
  },
  dialog: {
    pickDirectory: () => ipcRenderer.invoke('dialog:pickDirectory') as Promise<OpenDialogResult>,
    pickFile: (filters?: FileFilter[]) =>
      ipcRenderer.invoke('dialog:pickFile', filters) as Promise<OpenDialogResult>,
  },
}

contextBridge.exposeInMainWorld('electronAPI', api)

export type ElectronAPI = typeof api
