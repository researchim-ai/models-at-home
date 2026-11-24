import { useEffect } from 'react'
import { NavLink } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'
import {
  GraduationCap,
  Eye,
  Blocks,
  BookOpen,
  FlaskConical,
  Bot,
  Settings,
  Activity,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { useDockerStore } from '@/stores/dockerStore'

const navItems = [
  { path: '/', icon: GraduationCap, labelKey: 'nav.training' },
  { path: '/vlm', icon: Eye, labelKey: 'nav.vlm' },
  { path: '/vla', icon: Bot, labelKey: 'nav.vla' },
  { path: '/world-models', icon: Activity, labelKey: 'nav.worldModels' },
  { path: '/builder', icon: Blocks, labelKey: 'nav.builder' },
  { path: '/study', icon: BookOpen, labelKey: 'nav.study' },
  { path: '/notebooks', icon: FlaskConical, labelKey: 'nav.notebooks' },
  { path: '/agent', icon: Bot, labelKey: 'nav.agent' },
  { path: '/settings', icon: Settings, labelKey: 'nav.settings' },
]

export function Sidebar() {
  const { t } = useTranslation()
  const backendOnline = useDockerStore((s) => s.backendOnline)
  const checkBackend = useDockerStore((s) => s.checkBackend)

  useEffect(() => {
    checkBackend()
    const interval = setInterval(checkBackend, 5000)
    return () => clearInterval(interval)
  }, [checkBackend])

  return (
    <aside className="flex h-full w-[220px] flex-col border-r border-border bg-card">
      <div className="flex items-center gap-3 px-5 py-5">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground font-bold text-sm">
          M
        </div>
        <div className="flex flex-col">
          <span className="text-sm font-semibold">Models at Home</span>
          <span className="text-[10px] text-muted-foreground">Training Studio</span>
        </div>
      </div>

      <nav className="flex-1 space-y-0.5 px-3 py-2">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === '/'}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors',
                isActive
                  ? 'bg-accent text-accent-foreground font-medium'
                  : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground',
              )
            }
          >
            <item.icon className="h-4 w-4 shrink-0" />
            <span className="truncate">{t(item.labelKey)}</span>
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-border px-4 py-3">
        <div className="flex items-center gap-2">
          <Activity className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-xs text-muted-foreground">Backend</span>
          <Badge
            variant={backendOnline ? 'success' : 'secondary'}
            className="ml-auto text-[10px] px-1.5 py-0"
          >
            {backendOnline ? 'Online' : 'Offline'}
          </Badge>
        </div>
      </div>
    </aside>
  )
}
