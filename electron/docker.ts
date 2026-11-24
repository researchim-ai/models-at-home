import Docker from 'dockerode'
import fs from 'fs'
import path from 'path'

const BASE_IMAGE_NAME = 'models-at-home-base:latest'
const IMAGE_NAME = 'models-at-home-studio:latest'
const CONTAINER_NAME = 'mah-studio-backend'
const CONTAINER_PORT = 8000
const OWNER_LABEL = 'com.modelsathome.studio'

export type BuildProgress = (line: string) => void

export interface DockerStatus {
  running: boolean
  containerId?: string
  hostPort?: number
  error?: string
}

export class DockerManager {
  private docker: Docker
  private containerId: string | null = null
  private hostPort: number | null = null
  private projectRoot: string
  private studioRoot: string
  private workspaceRoot: string

  constructor(projectRoot?: string, studioRoot?: string, workspaceRoot?: string) {
    this.docker = new Docker()
    this.projectRoot =
      projectRoot ?? path.resolve(__dirname, '..', '..', 'models-at-home')
    this.studioRoot = studioRoot ?? path.dirname(this.projectRoot)
    this.workspaceRoot = workspaceRoot ?? this.projectRoot
  }

  // -----------------------------------------------------------------------
  // Availability
  // -----------------------------------------------------------------------

  async isAvailable(): Promise<boolean> {
    try {
      await this.docker.ping()
      return true
    } catch {
      return false
    }
  }

  hasDockerfile(): boolean {
    return (
      fs.existsSync(path.join(this.projectRoot, 'Dockerfile')) &&
      fs.existsSync(path.join(this.studioRoot, 'backend', 'Dockerfile'))
    )
  }

  async imageExists(): Promise<boolean> {
    try {
      const images = await this.docker.listImages({
        filters: { reference: [IMAGE_NAME] },
      })
      return images.length > 0
    } catch {
      return false
    }
  }

  // -----------------------------------------------------------------------
  // Status
  // -----------------------------------------------------------------------

  async getStatus(): Promise<DockerStatus> {
    try {
      const containers = await this.docker.listContainers({
        all: true,
        filters: { name: [CONTAINER_NAME] },
      })

      if (containers.length > 0) {
        const c = containers.find(c => c.Names?.includes(`/${CONTAINER_NAME}`))
        if (!c) return { running: false }
        if (c.Labels?.[OWNER_LABEL] !== '1') return { running: false, error: 'Container name is already used by an unmanaged container' }
        this.containerId = c.Id
        const mapped = c.Ports?.find((p) => p.PrivatePort === CONTAINER_PORT)
        const port = mapped?.PublicPort ?? this.hostPort ?? undefined
        if (port) this.hostPort = port
        return {
          running: c.State === 'running',
          containerId: c.Id,
          hostPort: port,
        }
      }
      return { running: false }
    } catch (error) {
      return { running: false, error: String(error) }
    }
  }

  // -----------------------------------------------------------------------
  // Build
  // -----------------------------------------------------------------------

  async buildImage(
    onProgress?: BuildProgress,
  ): Promise<{ success: boolean; error?: string }> {
    if (!this.hasDockerfile()) {
      return {
        success: false,
        error:
          `Dockerfile not found. Expected ${this.projectRoot}/Dockerfile ` +
          `and ${this.studioRoot}/backend/Dockerfile`,
      }
    }

    try {
      await this.buildTaggedImage(
        { context: this.projectRoot, src: ['.'] },
        {
          t: BASE_IMAGE_NAME,
          dockerfile: 'Dockerfile',
          buildargs: { LLAMA_CPP_BACKEND: 'vulkan' },
        },
        onProgress,
        'base',
      )

      await this.buildTaggedImage(
        { context: this.studioRoot, src: ['backend', 'examples', 'requirements-world-foundation.txt', 'requirements-world-local.txt'] },
        {
          t: IMAGE_NAME,
          dockerfile: 'backend/Dockerfile',
        },
        onProgress,
        'backend',
      )

      return { success: true }
    } catch (error) {
      return { success: false, error: String(error) }
    }
  }

  private async buildTaggedImage(
    context: { context: string; src: string[] },
    opts: Record<string, unknown>,
    onProgress?: BuildProgress,
    label = 'image',
  ): Promise<void> {
    onProgress?.(`[${label}] Building ${String(opts.t ?? 'image')}`)
    const stream = await this.docker.buildImage(
      context,
      opts,
    )

    await new Promise<void>((resolve, reject) => {
      this.docker.modem.followProgress(
        stream,
        (err: Error | null) => (err ? reject(err) : resolve()),
        (event: Record<string, unknown>) => {
          if (!onProgress) return
          const stream = event.stream as string | undefined
          const status = event.status as string | undefined
          const progress = event.progress as string | undefined
          const errorDetail = event.errorDetail as { message?: string } | undefined
          const text =
            errorDetail?.message ||
            (stream ? stream.trimEnd() : '') ||
            (status ? `${status}${progress ? ' ' + progress : ''}` : '')
          if (text) onProgress(`[${label}] ${text}`)
        },
      )
    })
  }

  // -----------------------------------------------------------------------
  // Run
  // -----------------------------------------------------------------------

  async ensureContainerRemoved(): Promise<void> {
    const containers = await this.docker.listContainers({ all: true, filters: { name: [CONTAINER_NAME] } })
    for (const c of containers) {
      if (!c.Names?.includes(`/${CONTAINER_NAME}`)) continue
      if (c.Labels?.[OWNER_LABEL] !== '1') throw new Error('Container name is already used by an unmanaged container; stop or rename it explicitly')
      const container = this.docker.getContainer(c.Id)
      if (c.State === 'running') await container.stop({ t: 5 })
      await container.remove()
    }
  }

  async startContainer(opts: {
    hostPort: number
    jupyterPort?: number
    gpu?: boolean
    apiToken?: string
  }): Promise<{ success: boolean; hostPort?: number; error?: string }> {
    try {
      await this.ensureContainerRemoved()
      for (const dir of ['datasets', 'out', '.runs', 'blueprints', 'models', 'configs', 'notebooks', 'study_materials']) {
        fs.mkdirSync(path.join(this.workspaceRoot, dir), { recursive: true })
      }

      const deviceRequests = opts.gpu
        ? [{ Count: -1, Capabilities: [['gpu']] }]
        : undefined

      const container = await this.docker.createContainer({
        Image: IMAGE_NAME,
        name: CONTAINER_NAME,
        Labels: { [OWNER_LABEL]: '1' },
        Cmd: [
          'python',
          '-m',
          'uvicorn',
          'backend.api:app',
          '--host',
          '0.0.0.0',
          '--port',
          String(CONTAINER_PORT),
        ],
        ExposedPorts: { [`${CONTAINER_PORT}/tcp`]: {}, '8888/tcp': {} },
        HostConfig: {
          PortBindings: {
            [`${CONTAINER_PORT}/tcp`]: [{ HostIp: '127.0.0.1', HostPort: String(opts.hostPort) }],
            '8888/tcp': [{ HostIp: '127.0.0.1', HostPort: String(opts.jupyterPort ?? 8888) }],
          },
          Binds: [
            // Keep API source current while ML dependencies remain in the image.
            `${this.studioRoot}/backend:/app/backend:ro`,
            `${this.workspaceRoot}/datasets:/app/datasets`,
            `${this.workspaceRoot}/out:/app/out`,
            `${this.workspaceRoot}/.runs:/app/.runs`,
            `${this.workspaceRoot}/blueprints:/app/blueprints`,
            `${this.workspaceRoot}/models:/app/models`,
            `${this.workspaceRoot}/configs:/app/configs`,
            `${this.workspaceRoot}/notebooks:/app/notebooks`,
            `${this.workspaceRoot}/study_materials:/app/study_materials:ro`,
          ],
          ShmSize: 8 * 1024 * 1024 * 1024,
          DeviceRequests: deviceRequests,
          RestartPolicy: { Name: 'unless-stopped' },
        },
        Env: [
          'PYTHONUNBUFFERED=1',
          'MODELS_AT_HOME_ROOT=/app',
          ...['HF_TOKEN'].filter(name => process.env[name]).map(name => `${name}=${process.env[name]}`),
          ...(process.env.MAH_DOCKER_WORLD_PYTHON ? [`MAH_WORLD_PYTHON=${process.env.MAH_DOCKER_WORLD_PYTHON}`] : []),
          `MAH_API_TOKEN=${opts.apiToken ?? ''}`,
          'JUPYTER_BIND_HOST=0.0.0.0',
          'MKL_THREADING_LAYER=GNU',
          'HOME=/tmp',
          'HF_HOME=/tmp/.cache/huggingface',
          'NVIDIA_VISIBLE_DEVICES=all',
          'NVIDIA_DRIVER_CAPABILITIES=all',
          `JUPYTER_PUBLIC_URL=http://127.0.0.1:${opts.jupyterPort ?? 8888}`,
        ],
      })

      await container.start()
      this.containerId = container.id
      this.hostPort = opts.hostPort
      return { success: true, hostPort: opts.hostPort }
    } catch (error) {
      return { success: false, error: String(error) }
    }
  }

  async stopContainer(): Promise<{ success: boolean; error?: string }> {
    try {
      const status = await this.getStatus()
      if (!status.running || !status.containerId) return { success: true }

      const container = this.docker.getContainer(status.containerId)
      await container.stop({ t: 10 })
      return { success: true }
    } catch (error) {
      return { success: false, error: String(error) }
    }
  }

  async waitForHealth(
    hostPort: number,
    opts: { timeoutMs?: number; apiToken?: string; onTick?: (attempt: number) => void } = {},
  ): Promise<boolean> {
    const timeoutMs = opts.timeoutMs ?? 120_000
    const start = Date.now()
    let attempt = 0
    while (Date.now() - start < timeoutMs) {
      attempt += 1
      opts.onTick?.(attempt)
      try {
        const res = await fetch(
          `http://127.0.0.1:${hostPort}/api/system/health`,
          { signal: AbortSignal.timeout(2000), headers: { 'X-Studio-Token': opts.apiToken ?? '' } },
        )
        if (res.ok) return true
      } catch { /* not ready */ }
      await new Promise((r) => setTimeout(r, 1000))
    }
    return false
  }

  async getContainerLogs(tail = 200): Promise<string> {
    try {
      const status = await this.getStatus()
      if (!status.containerId) return ''

      const container = this.docker.getContainer(status.containerId)
      const logs = await container.logs({
        stdout: true,
        stderr: true,
        tail,
        timestamps: true,
      })
      return logs.toString()
    } catch {
      return ''
    }
  }
}
