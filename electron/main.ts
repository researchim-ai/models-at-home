import { app, BrowserWindow, ipcMain, shell, dialog, Menu, nativeImage } from 'electron'
import path from 'path'
import fs from 'fs'
import net from 'net'
import { randomBytes } from 'crypto'
import { terminateChild } from './process'
import { resolveRuntime, prepareWorkspace, externalUrl } from './paths'
import { spawn, spawnSync, type ChildProcess } from 'child_process'
import { DockerManager } from './docker'
import { loadConfig, saveConfig, type AppConfig, type BackendMode } from './config'

if (process.platform === 'linux') {
  app.disableHardwareAcceleration()
  app.commandLine.appendSwitch('disable-gpu')
  app.commandLine.appendSwitch('disable-gpu-compositing')
  app.commandLine.appendSwitch('disable-accelerated-2d-canvas')
}

let mainWindow: BrowserWindow | null = null
let dockerManager: DockerManager | null = null
let backendProcess: ChildProcess | null = null
let backendPort = 8000
const apiToken = randomBytes(32).toString('hex')
let backendStartError: string | null = null
let currentMode: 'native' | 'docker' | 'offline' = 'offline'
let booting = false
let startup: Promise<void> | null = null
let operations: Promise<unknown> = Promise.resolve()
let runtimeRoot = ''
let workspaceRoot = ''

// Boot phases — streamed to renderer for splash screen feedback
type BootPhase =
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

let currentBootPhase: BootPhase = { phase: 'starting' }

function emitBootPhase(p: BootPhase) {
  currentBootPhase = p
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('backend:boot-phase', p)
  }
}

const isDev = !!process.env.VITE_DEV_SERVER_URL
const VITE_DEV_SERVER_URL = process.env.VITE_DEV_SERVER_URL

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

function getResourcePath(...segments: string[]): string {
  if (isDev) {
    return path.join(__dirname, '..', ...segments)
  }
  return path.join(process.resourcesPath, ...segments)
}

function getIconPath(): string | undefined {
  const candidates = isDev
    ? [
        path.join(__dirname, '..', 'build', 'icons', '512x512.png'),
        path.join(__dirname, '..', 'build', 'icon.png'),
      ]
    : [
        path.join(process.resourcesPath, 'icons', '512x512.png'),
        path.join(process.resourcesPath, 'icon.png'),
      ]
  for (const p of candidates) {
    if (fs.existsSync(p)) return p
  }
  return undefined
}

// ---------------------------------------------------------------------------
// Port helpers
// ---------------------------------------------------------------------------

function findAvailablePort(start = 8000): Promise<number> {
  return new Promise((resolve) => {
    const srv = net.createServer()
    srv.unref()
    srv.on('error', () => resolve(findAvailablePort(start + 1)))
    srv.listen(start, '127.0.0.1', () => {
      const addr = srv.address()
      const port = typeof addr === 'object' && addr ? addr.port : start
      srv.close(() => resolve(port))
    })
  })
}

// ---------------------------------------------------------------------------
// Python resolver
// ---------------------------------------------------------------------------

function resolvePythonCommand(): string | null {
  const explicit = process.env.MAH_PYTHON
  const localEnvironment = process.platform === 'win32'
    ? path.join(getResourcePath(), '.venv', 'Scripts', 'python.exe')
    : path.join(getResourcePath(), '.venv', 'bin', 'python')
  const candidates = explicit
    ? [explicit]
    : [localEnvironment, ...(process.platform === 'win32' ? ['python', 'py', 'python3'] : ['python3', 'python'])]
  for (const cmd of candidates) {
    try {
      const result = spawnSync(cmd, ['-c', 'import fastapi, uvicorn, psutil, accelerate; from transformers import AutoTokenizer, AutoModelForCausalLM'], { stdio: 'pipe', timeout: 20000 })
      if (result.status === 0) return cmd
      if (result.stderr?.length) backendStartError = result.stderr.toString().trim().split('\n').slice(-2).join(' ')
    } catch { /* continue */ }
  }
  return null
}

// ---------------------------------------------------------------------------
// Window
// ---------------------------------------------------------------------------

function createWindow() {
  const iconPath = getIconPath()

  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1024,
    minHeight: 700,
    title: 'Models at Home Studio',
    icon: iconPath ? nativeImage.createFromPath(iconPath) : undefined,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    backgroundColor: '#09090b',
    show: false,
  })

  mainWindow.once('ready-to-show', () => mainWindow?.show())

  if (VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(VITE_DEV_SERVER_URL)
    mainWindow.webContents.openDevTools({ mode: 'detach' })
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'))
  }

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    try { void shell.openExternal(externalUrl(url)) } catch { /* unsupported scheme */ }
    return { action: 'deny' }
  })

  mainWindow.webContents.on('will-navigate', (event, url) => {
    const current = mainWindow?.webContents.getURL()
    if (url !== current) event.preventDefault()
  })

  mainWindow.webContents.on('did-finish-load', () => {
    emitBootPhase(currentBootPhase)
  })

  mainWindow.on('closed', () => { mainWindow = null })
}

// ---------------------------------------------------------------------------
// Application menu
// ---------------------------------------------------------------------------

function buildAppMenu() {
  const isMac = process.platform === 'darwin'

  const template: Electron.MenuItemConstructorOptions[] = [
    ...(isMac
      ? [{
          label: app.name,
          submenu: [
            { role: 'about' as const },
            { type: 'separator' as const },
            { role: 'services' as const },
            { type: 'separator' as const },
            { role: 'hide' as const },
            { role: 'hideOthers' as const },
            { role: 'unhide' as const },
            { type: 'separator' as const },
            { role: 'quit' as const },
          ],
        }]
      : []),
    {
      label: 'File',
      submenu: [
        isMac ? { role: 'close' as const } : { role: 'quit' as const },
      ],
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        { role: 'selectAll' },
      ],
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' },
      ],
    },
    {
      label: 'Backend',
      submenu: [
        {
          label: 'Restart Backend',
          click: async () => {
            await restartBackend().catch((err) => console.error('Restart failed:', err))
          },
        },
        {
          label: 'Show Backend Logs',
          click: () => {
            const logPath = path.join(app.getPath('logs'), 'backend.log')
            shell.showItemInFolder(logPath)
          },
        },
      ],
    },
    {
      role: 'help',
      submenu: [
        {
          label: 'Documentation',
          click: () => shell.openExternal('https://github.com/models-at-home'),
        },
        {
          label: 'About',
          click: () => {
            dialog.showMessageBox(mainWindow!, {
              type: 'info',
              title: 'Models at Home Studio',
              message: `Models at Home Studio v${app.getVersion()}`,
              detail: 'Desktop studio for training ML models at home.',
              icon: getIconPath() ? nativeImage.createFromPath(getIconPath()!) : undefined,
            })
          },
        },
      ],
    },
  ]

  Menu.setApplicationMenu(Menu.buildFromTemplate(template))
}

// ---------------------------------------------------------------------------
// Backend lifecycle — dispatcher based on config.backendMode
// ---------------------------------------------------------------------------

function startBackend(): Promise<void> {
  if (!startup) startup = startBackendOnce().finally(() => { startup = null })
  return startup
}

function restartBackend(): Promise<void> {
  const task = operations.catch(() => {}).then(async () => {
    await startup?.catch(() => {})
    await stopBackend()
    await startBackend()
  })
  operations = task
  return task
}

async function startBackendOnce(): Promise<void> {
  if (booting || backendProcess || currentMode === 'docker') return
  booting = true
  backendStartError = null
  emitBootPhase({ phase: 'starting' })

  if (!runtimeRoot) { booting = false; throw new Error(backendStartError ?? 'ML runtime missing') }
  const config = loadConfig()
  const mode = config.backendMode

  try {
    if (mode === 'docker') {
      const ok = await startBackendDocker(config)
      if (!ok) throw new Error(backendStartError ?? 'Docker start failed')
      return
    }

    if (mode === 'auto') {
      emitBootPhase({ phase: 'checking-docker' })
      const dockerOk = dockerManager ? await dockerManager.isAvailable() : false
      if (dockerOk && dockerManager?.hasDockerfile()) {
        const ok = await startBackendDocker(config)
        if (ok) return
        console.warn('[backend] Docker attempt failed, falling back to native')
      } else {
        emitBootPhase({
          phase: 'docker-unavailable',
          detail: dockerOk ? 'No Dockerfile' : 'Docker daemon not reachable',
        })
      }
    }

    if (quitting) return
    await startBackendNative()
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    backendStartError = msg
    emitBootPhase({ phase: 'failed', error: msg })
    throw err
  } finally {
    booting = false
  }
}

async function startBackendDocker(config: AppConfig): Promise<boolean> {
  if (!dockerManager) return false

  const ok = await dockerManager.isAvailable()
  if (!ok) {
    emitBootPhase({
      phase: 'docker-unavailable',
      detail: 'Docker daemon not reachable',
    })
    return false
  }

  if (!dockerManager.hasDockerfile()) {
    emitBootPhase({
      phase: 'docker-unavailable',
      detail: 'Dockerfile not found',
    })
    return false
  }

  const existing = await dockerManager.getStatus()

  if (!(await dockerManager.imageExists())) {
    emitBootPhase({ phase: 'docker-no-image' })
    if (!config.dockerAutoBuild) {
      backendStartError = 'Docker image missing and auto-build is disabled'
      return false
    }
    emitBootPhase({ phase: 'building-image' })
    const build = await dockerManager.buildImage((line) =>
      emitBootPhase({ phase: 'building-image', line }),
    )
    if (!build.success) {
      backendStartError = build.error ?? 'Image build failed'
      return false
    }
  }

  const hostPort = existing.hostPort ?? (await findAvailablePort(8000))
  const jupyterPort = await findAvailablePort(8888)
  emitBootPhase({ phase: 'starting-container' })
  const started = await dockerManager.startContainer({
    hostPort,
    jupyterPort,
    gpu: config.dockerGpu,
    apiToken,
  })
  if (!started.success) {
    backendStartError = started.error ?? 'Container start failed'
    return false
  }

  backendPort = started.hostPort ?? hostPort
  const healthy = await dockerManager.waitForHealth(backendPort, {
    timeoutMs: 120_000,
    apiToken,
    onTick: (attempt) =>
      emitBootPhase({ phase: 'waiting-container-health', attempt }),
  })
  if (!healthy) {
    backendStartError = 'Container did not become healthy in 2 minutes'
    await dockerManager.stopContainer()
    return false
  }

  currentMode = 'docker'
  emitBootPhase({ phase: 'ready', mode: 'docker', port: backendPort })
  return true
}

async function startBackendNative(): Promise<void> {
  emitBootPhase({ phase: 'starting-python' })
  const pythonCmd = resolvePythonCommand()
  if (!pythonCmd) throw new Error(`Python ML/API imports failed. Use a compatible .venv or set MAH_PYTHON. ${backendStartError ?? ''}`)
  backendPort = await findAvailablePort(8000)
  const projectRoot = getResourcePath()
  const child = spawn(pythonCmd, [
    '-m', 'uvicorn', 'backend.api:app', '--host', '127.0.0.1', '--port', String(backendPort),
  ], {
    cwd: projectRoot,
    detached: process.platform !== 'win32',
    env: {
      ...process.env, MAH_API_TOKEN: apiToken, PYTHONUNBUFFERED: '1', MODELS_AT_HOME_ROOT: workspaceRoot,
      PYTHONPATH: [projectRoot, runtimeRoot, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  backendProcess = child
  let spawnError: Error | null = null
  const logPath = path.join(app.getPath('logs'), 'backend.log')
  fs.mkdirSync(path.dirname(logPath), { recursive: true })
  const log = fs.createWriteStream(logPath, { flags: 'a' })
  log.on('error', err => console.error('[backend] log write failed:', err))
  log.write(`\n--- Session start ${new Date().toISOString()} ---\n`)
  const onData = (buf: Buffer) => {
    const line = buf.toString()
    log.write(line)
    if (currentMode !== 'native') emitBootPhase({ phase: 'python-starting', line: line.trim().slice(-500) })
  }
  child.stdout?.on('data', onData)
  child.stderr?.on('data', onData)
  child.on('error', err => {
    spawnError = err
    log.end()
    if (backendProcess === child) backendProcess = null
  })
  child.once('exit', (code, signal) => {
    log.end(`--- Exit code=${code} signal=${signal} ---\n`)
    if (backendProcess === child) {
      backendProcess = null
      currentMode = 'offline'
      backendStartError = `Backend exited (code ${code})`
      emitBootPhase({ phase: 'failed', error: backendStartError })
    }
  })
  const deadline = Date.now() + 60_000
  while (Date.now() < deadline) {
    if (quitting) { await terminateChild(child); return }
    if (spawnError) throw spawnError
    if (child.exitCode !== null || child.signalCode !== null) throw new Error('Backend exited before becoming healthy; see backend.log')
    try {
      const response = await fetch(`http://127.0.0.1:${backendPort}/api/system/health`, { signal: AbortSignal.timeout(1500), headers: { 'X-Studio-Token': apiToken } })
      if (response.ok && backendProcess === child) {
        currentMode = 'native'
        backendStartError = null
        emitBootPhase({ phase: 'ready', mode: 'native', port: backendPort })
        return
      }
    } catch { /* not ready */ }
    await new Promise(resolve => setTimeout(resolve, 250))
  }
  await terminateChild(child)
  throw new Error('Backend did not become healthy in 60 seconds')
}

async function stopBackend(): Promise<void> {
  const child = backendProcess
  if (dockerManager && (currentMode === 'docker' || ['starting-container', 'waiting-container-health'].includes(currentBootPhase.phase))) {
    const result = await dockerManager.stopContainer()
    if (!result.success) throw new Error(result.error ?? 'Could not stop Docker backend')
  }
  currentMode = 'offline'
  if (child) {
    // Detach state before exit, but keep the captured child until termination completes.
    if (backendProcess === child) backendProcess = null
    await terminateChild(child)
  }
}

// ---------------------------------------------------------------------------
// App lifecycle
// ---------------------------------------------------------------------------

const gotSingleInstanceLock = app.requestSingleInstanceLock()
if (!gotSingleInstanceLock) {
  app.quit()
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore()
      mainWindow.focus()
    }
  })
}

app.whenReady().then(async () => {
  buildAppMenu()
  try {
    runtimeRoot = resolveRuntime(getResourcePath(), isDev)
    workspaceRoot = process.env.MODELS_AT_HOME_ROOT ?? (isDev ? runtimeRoot : path.join(app.getPath('userData'), 'workspace'))
    prepareWorkspace(runtimeRoot, workspaceRoot)
    dockerManager = new DockerManager(runtimeRoot, getResourcePath(), workspaceRoot)
  } catch (error) {
    backendStartError = String(error)
    emitBootPhase({ phase: 'failed', error: backendStartError })
  }
  registerIpcHandlers()

  createWindow()

  if (runtimeRoot) startBackend().catch((err) => {
    console.error('[backend] Could not auto-start:', err)
  })

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

let quitting = false
app.on('before-quit', (event) => {
  if (quitting) return
  event.preventDefault()
  quitting = true
  void (async () => {
    await stopBackend()
  })().catch(err => console.error('[backend] shutdown:', err)).finally(() => app.quit())
})

// ---------------------------------------------------------------------------
// IPC handlers
// ---------------------------------------------------------------------------

function registerIpcHandlers() {
  ipcMain.handle('backend:status', async () => {
    try {
      const res = await fetch(`http://127.0.0.1:${backendPort}/api/system/health`, { signal: AbortSignal.timeout(2000), headers: { 'X-Studio-Token': apiToken } })
      return {
        running: res.ok,
        port: backendPort,
        error: backendStartError,
        mode: currentMode,
        phase: currentBootPhase,
      }
    } catch {
      return {
        running: false,
        port: backendPort,
        error: backendStartError,
        mode: currentMode,
        phase: currentBootPhase,
      }
    }
  })

  ipcMain.handle('backend:restart', async () => {
    try {
      await restartBackend()
      return { success: true, port: backendPort, mode: currentMode }
    } catch (err) {
      return { success: false, error: String(err), port: backendPort, mode: currentMode }
    }
  })

  ipcMain.handle('backend:port', () => backendPort)
  ipcMain.handle('backend:token', () => apiToken)

  ipcMain.handle('backend:boot-phase', () => currentBootPhase)

  ipcMain.handle('backend:open-logs', () => {
    const logPath = path.join(app.getPath('logs'), 'backend.log')
    shell.showItemInFolder(logPath)
  })

  // Config
  ipcMain.handle('config:get', () => loadConfig())
  ipcMain.handle('config:set', (_e, patch: Partial<AppConfig>) => saveConfig(patch))
  ipcMain.handle('config:set-backend-mode', async (_e, mode: BackendMode) => {
    const next = saveConfig({ backendMode: mode })
    void restartBackend().catch((err) => console.error('[backend] Restart after mode change failed:', err))
    return next
  })

  // Docker
  ipcMain.handle('docker:status', async () => {
    return dockerManager?.getStatus() ?? { running: false }
  })
  ipcMain.handle('docker:available', async () =>
    dockerManager ? dockerManager.isAvailable() : false,
  )
  ipcMain.handle('docker:image-exists', async () =>
    dockerManager ? dockerManager.imageExists() : false,
  )
  ipcMain.handle('docker:start', async () => {
    saveConfig({ backendMode: 'docker' })
    try {
      await restartBackend()
      return { success: true, hostPort: backendPort }
    } catch (err) { return { success: false, error: String(err) } }
  })
  ipcMain.handle('docker:stop', async () => {
    try {
      await stopBackend()
      return { success: true }
    } catch (err) { return { success: false, error: String(err) } }
  })
  ipcMain.handle('docker:build', async () => {
    if (!dockerManager) return { success: false, error: 'Docker not initialized' }
    return dockerManager.buildImage((line) =>
      mainWindow?.webContents.send('docker:build-progress', line),
    )
  })
  ipcMain.handle('docker:logs', async (_event, tail?: number) =>
    dockerManager?.getContainerLogs(tail),
  )

  ipcMain.handle('app:version', () => app.getVersion())
  ipcMain.handle('app:platform', () => process.platform)
  ipcMain.handle('app:paths', () => ({
    userData: app.getPath('userData'),
    logs: app.getPath('logs'),
    temp: app.getPath('temp'),
  }))

  ipcMain.handle('shell:openExternal', (_event, url: string) => shell.openExternal(externalUrl(url)))
  ipcMain.handle('shell:showInFolder', (_event, p: string) => shell.showItemInFolder(p))

  ipcMain.handle('dialog:pickDirectory', async () => {
    if (!mainWindow) return { canceled: true, filePaths: [] }
    return dialog.showOpenDialog(mainWindow, {
      properties: ['openDirectory'],
    })
  })

  ipcMain.handle('dialog:pickFile', async (_event, filters?: Electron.FileFilter[]) => {
    if (!mainWindow) return { canceled: true, filePaths: [] }
    return dialog.showOpenDialog(mainWindow, {
      properties: ['openFile'],
      filters,
    })
  })
}
