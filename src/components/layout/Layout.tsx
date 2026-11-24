import { Outlet, useLocation } from 'react-router-dom'
import { Sidebar } from './Sidebar'
import { Header } from './Header'
import { Toaster } from 'sonner'
import { ErrorBoundary } from '@/components/ErrorBoundary'

export function Layout() {
  const location = useLocation()
  return (
    <div className="studio-shell flex h-screen w-screen overflow-hidden bg-background" data-space={location.pathname.split('/')[1] || 'training'}>
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <Header />
        <main className="studio-content min-h-0 flex-1 overflow-auto">
          <ErrorBoundary resetKey={location.pathname}>
            <Outlet />
          </ErrorBoundary>
        </main>
      </div>
      <Toaster
        position="bottom-right"
        toastOptions={{
          className: 'bg-card text-card-foreground border-border',
        }}
      />
    </div>
  )
}
