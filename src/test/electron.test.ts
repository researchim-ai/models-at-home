import { describe, expect, it } from 'vitest'
import { spawn } from 'child_process'
import { once } from 'events'
import { mkdtempSync, mkdirSync, rmSync } from 'fs'
import { tmpdir } from 'os'
import path from 'path'
import { terminateChild } from '../../electron/process'
import { externalUrl, resolveRuntime } from '../../electron/paths'

describe('desktop regressions', () => {
  it('force-stops the captured process without killing a newer child', async () => {
    if (process.platform === 'win32') return
    const old = spawn(process.execPath, ['-e', 'process.on("SIGTERM",()=>{}); console.log("ready"); setInterval(()=>{},1000)'], { detached: true })
    const next = spawn(process.execPath, ['-e', 'console.log("ready"); setInterval(()=>{},1000)'], { detached: true })
    try {
      await Promise.all([once(old.stdout!, 'data'), once(next.stdout!, 'data')])
      await terminateChild(old, 25)
      expect(old.signalCode).toBe('SIGKILL')
      expect(next.exitCode).toBeNull()
      expect(next.signalCode).toBeNull()
    } finally {
      await terminateChild(old, 25)
      await terminateChild(next, 25)
    }
  })

  it('resolves a sibling runtime in development and rejects a missing packaged runtime', () => {
    const folder = mkdtempSync(path.join(tmpdir(), 'mah-runtime-'))
    try {
      mkdirSync(path.join(folder, 'studio'))
      mkdirSync(path.join(folder, 'models-at-home', 'homellm'), { recursive: true })
      expect(resolveRuntime(path.join(folder, 'studio'), true)).toBe(path.join(folder, 'models-at-home'))
      expect(() => resolveRuntime(path.join(folder, 'studio'), false)).toThrow('ML runtime missing')
    } finally { rmSync(folder, { recursive: true, force: true }) }
  })

  it('rejects shell/file protocols in renderer openExternal requests', () => {
    expect(externalUrl('https://example.com')).toBe('https://example.com/')
    for (const url of ['file:///etc/passwd', 'javascript:alert(1)', 'ms-settings:test']) {
      expect(() => externalUrl(url)).toThrow()
    }
  })
})
