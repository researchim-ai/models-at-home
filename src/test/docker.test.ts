import { beforeEach, expect, it, vi } from 'vitest'

const driver = vi.hoisted(() => ({ listContainers: vi.fn(), getContainer: vi.fn(), stop: vi.fn(), remove: vi.fn() }))
vi.mock('dockerode', () => ({ default: class {
  listContainers = driver.listContainers
  getContainer = driver.getContainer
} }))
import { DockerManager } from '../../electron/docker'

beforeEach(() => {
  vi.clearAllMocks()
  driver.getContainer.mockReturnValue({ stop: driver.stop, remove: driver.remove })
})

it('does not delete a container with a similar name', async () => {
  driver.listContainers.mockResolvedValue([
    { Id: 'other', Names: ['/old-mah-studio-backend'], State: 'running', Labels: {} },
    { Id: 'owned', Names: ['/mah-studio-backend'], State: 'running', Labels: { 'com.modelsathome.studio': '1' } },
  ])
  await new DockerManager('/tmp/runtime').ensureContainerRemoved()
  expect(driver.getContainer).toHaveBeenCalledExactlyOnceWith('owned')
  expect(driver.remove).toHaveBeenCalledOnce()
})

it('refuses to replace an unmanaged container even with the exact name', async () => {
  driver.listContainers.mockResolvedValue([{ Id: 'unmanaged', Names: ['/mah-studio-backend'], State: 'running', Labels: {} }])
  await expect(new DockerManager('/tmp/runtime').ensureContainerRemoved()).rejects.toThrow('unmanaged container')
  expect(driver.stop).not.toHaveBeenCalled()
  expect(driver.remove).not.toHaveBeenCalled()
})
