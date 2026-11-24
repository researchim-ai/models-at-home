import { spawn, type ChildProcess } from 'child_process'

/** Wait for the captured child to exit; a later restart cannot change the target. */
export async function terminateChild(child: ChildProcess, graceMs = 5000): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) return
  await new Promise<void>((resolve, reject) => {
    let killTimer: ReturnType<typeof setTimeout>
    let deadline: ReturnType<typeof setTimeout>
    const cleanup = () => {
      clearTimeout(killTimer)
      clearTimeout(deadline)
      child.removeListener('exit', onExit)
    }
    const onExit = () => { cleanup(); resolve() }
    const signal = (force: boolean) => {
      try {
        if (process.platform === 'win32') {
          const killer = spawn('taskkill', ['/pid', String(child.pid), '/t', ...(force ? ['/f'] : [])])
          killer.on('error', () => child.kill())
        } else if (child.pid) {
          process.kill(-child.pid, force ? 'SIGKILL' : 'SIGTERM')
        }
      } catch (err) {
        if ((err as NodeJS.ErrnoException).code !== 'ESRCH') {
          cleanup(); reject(err)
        }
      }
    }
    child.once('exit', onExit)
    killTimer = setTimeout(() => signal(true), graceMs)
    deadline = setTimeout(() => { cleanup(); reject(new Error('Backend did not exit after SIGKILL')) }, graceMs + 3000)
    signal(false)
  })
}
