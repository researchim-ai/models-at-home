import { lazy, Suspense, type ComponentType } from 'react'
import { Routes, Route } from 'react-router-dom'
import { Layout } from '@/components/layout/Layout'
import { BackendBoot } from '@/components/BackendBoot'
import { Loader2 } from 'lucide-react'

// A dynamic import can fail when the dev server rebuilt the module graph (HMR)
// or after a new deploy — the browser holds a stale chunk URL. This is not a
// real bug in the page, so self-heal by reloading the window once before
// surfacing the error to the boundary.
function lazyWithReload<T extends ComponentType<unknown>>(
  factory: () => Promise<{ default: T }>,
) {
  return lazy(async () => {
    try {
      const mod = await factory()
      sessionStorage.removeItem('chunk-reload')
      return mod
    } catch (err) {
      if (!sessionStorage.getItem('chunk-reload')) {
        sessionStorage.setItem('chunk-reload', '1')
        window.location.reload()
        // Never resolve — the page is reloading.
        return new Promise<{ default: T }>(() => {})
      }
      throw err
    }
  })
}

const TrainingStudio = lazyWithReload(() => import('@/pages/TrainingStudio').then((m) => ({ default: m.TrainingStudio })))
const WorldModelsStudio = lazyWithReload(() => import('@/pages/WorldModelsStudio').then(m => ({ default: m.WorldModelsStudio })))
const VLAStudio = lazyWithReload(() => import('@/pages/VLAStudio').then(m => ({ default: m.VLAStudio })))
const VLMStudio = lazyWithReload(() => import('@/pages/VLMStudio').then((m) => ({ default: m.VLMStudio })))
const ModelBuilder = lazyWithReload(() => import('@/pages/ModelBuilder').then((m) => ({ default: m.ModelBuilder })))
const StudyCenter = lazyWithReload(() => import('@/pages/StudyCenter').then((m) => ({ default: m.StudyCenter })))
const Notebooks = lazyWithReload(() => import('@/pages/Notebooks').then((m) => ({ default: m.Notebooks })))
const AgentStudio = lazyWithReload(() => import('@/pages/AgentStudio').then((m) => ({ default: m.AgentStudio })))
const SettingsPage = lazyWithReload(() => import('@/pages/Settings').then((m) => ({ default: m.SettingsPage })))

function PageFallback() {
  return (
    <div className="flex h-full w-full items-center justify-center">
      <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
    </div>
  )
}

export default function App() {
  return (
    <BackendBoot>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Suspense fallback={<PageFallback />}><TrainingStudio /></Suspense>} />
          <Route path="/world-models" element={<Suspense fallback={<PageFallback />}><WorldModelsStudio /></Suspense>} />
          <Route path="/vla" element={<Suspense fallback={<PageFallback />}><VLAStudio /></Suspense>} />
          <Route path="/vlm" element={<Suspense fallback={<PageFallback />}><VLMStudio /></Suspense>} />
          <Route path="/builder" element={<Suspense fallback={<PageFallback />}><ModelBuilder /></Suspense>} />
          <Route path="/study" element={<Suspense fallback={<PageFallback />}><StudyCenter /></Suspense>} />
          <Route path="/notebooks" element={<Suspense fallback={<PageFallback />}><Notebooks /></Suspense>} />
          <Route path="/agent" element={<Suspense fallback={<PageFallback />}><AgentStudio /></Suspense>} />
          <Route path="/settings" element={<Suspense fallback={<PageFallback />}><SettingsPage /></Suspense>} />
        </Route>
      </Routes>
    </BackendBoot>
  )
}
