import fs from 'fs'
import path from 'path'

export function resolveRuntime(studioRoot: string, development: boolean): string {
  const candidates = [process.env.MAH_RUNTIME_ROOT, path.join(studioRoot, 'models-at-home'),
    ...(development ? [path.resolve(studioRoot, '..', 'models-at-home')] : [])]
  const runtime = candidates.find(p => p && fs.existsSync(path.join(p, 'homellm')))
  if (!runtime) throw new Error('ML runtime missing. Set MAH_RUNTIME_ROOT or provide models-at-home/homellm before starting Studio.')
  return path.resolve(runtime)
}

export function prepareWorkspace(runtime: string, workspace: string): void {
  fs.mkdirSync(workspace, { recursive: true })
  for (const folder of ['configs', 'blueprints', 'notebooks', 'study_materials']) {
    const source = path.join(runtime, folder)
    if (fs.existsSync(source) && path.resolve(runtime) !== path.resolve(workspace)) {
      fs.cpSync(source, path.join(workspace, folder), { recursive: true, force: false, errorOnExist: false })
    }
  }
}

export function externalUrl(value: string): string {
  const url = new URL(value)
  if (!['https:', 'http:', 'mailto:'].includes(url.protocol)) throw new Error('Unsupported external URL scheme')
  return url.href
}
