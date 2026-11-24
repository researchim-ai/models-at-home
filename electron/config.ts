import { app } from 'electron'
import fs from 'fs'
import path from 'path'

export type BackendMode = 'native' | 'docker' | 'auto'

export interface AppConfig {
  backendMode: BackendMode
  dockerAutoBuild: boolean
  dockerGpu: boolean
}

const DEFAULT_CONFIG: AppConfig = {
  backendMode: 'auto',
  dockerAutoBuild: true,
  dockerGpu: true,
}

function validatePatch(value: unknown): Partial<AppConfig> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid app configuration')
  const raw = value as Record<string, unknown>
  const next: Partial<AppConfig> = {}
  if ('backendMode' in raw) {
    if (!['native', 'docker', 'auto'].includes(String(raw.backendMode))) throw new Error('Invalid backend mode')
    next.backendMode = raw.backendMode as BackendMode
  }
  for (const key of ['dockerAutoBuild', 'dockerGpu'] as const) {
    if (key in raw) {
      if (typeof raw[key] !== 'boolean') throw new Error(`Invalid ${key}`)
      next[key] = raw[key]
    }
  }
  return next
}

function configPath(): string {
  return path.join(app.getPath('userData'), 'config.json')
}

export function loadConfig(): AppConfig {
  try {
    const raw = fs.readFileSync(configPath(), 'utf-8')
    return { ...DEFAULT_CONFIG, ...validatePatch(JSON.parse(raw)) }
  } catch {
    return { ...DEFAULT_CONFIG }
  }
}

export function saveConfig(patch: Partial<AppConfig>): AppConfig {
  const current = loadConfig()
  const next = { ...current, ...validatePatch(patch) }
  try {
    fs.mkdirSync(path.dirname(configPath()), { recursive: true })
    const temp = configPath() + '.tmp'
    fs.writeFileSync(temp, JSON.stringify(next, null, 2))
    fs.renameSync(temp, configPath())
  } catch (err) {
    console.error('[config] save error:', err)
    throw err
  }
  return next
}
