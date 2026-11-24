import { Outlet, useLocation } from 'react-router-dom'
import { Sidebar } from './Sidebar'
import { Header } from './Header'
import { Toaster } from 'sonner'
import { ErrorBoundary } from '@/components/ErrorBoundary'

export function Layout() {
  const location = useLocation()
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background">
      <Sidebar />
      <div className="flex flex-1 flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-auto">
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
