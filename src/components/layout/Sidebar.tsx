import { useEffect } from 'react'
import { NavLink } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { GraduationCap, Eye, Blocks, BookOpen, FlaskConical, Bot, Settings, Activity } from 'lucide-react'
import { useDockerStore } from '@/stores/dockerStore'
import { BrandMark } from './BrandMark'

const groups = [
  { en: 'Train & explore', ru: 'Обучение и эксперименты', items: [
    { path: '/', icon: GraduationCap, labelKey: 'nav.training' },
    { path: '/vlm', icon: Eye, labelKey: 'nav.vlm' },
    { path: '/vla', icon: Bot, labelKey: 'nav.vla' },
    { path: '/world-models', icon: Activity, labelKey: 'nav.worldModels' },
  ] },
  { en: 'Workbench', ru: 'Инструменты', items: [
    { path: '/builder', icon: Blocks, labelKey: 'nav.builder' },
    { path: '/study', icon: BookOpen, labelKey: 'nav.study' },
    { path: '/notebooks', icon: FlaskConical, labelKey: 'nav.notebooks' },
    { path: '/agent', icon: Bot, labelKey: 'nav.agent' },
  ] },
]

export function Sidebar() {
  const { t, i18n } = useTranslation()
  const ru = i18n.language.startsWith('ru')
  const backendOnline = useDockerStore((s) => s.backendOnline)
  const checkBackend = useDockerStore((s) => s.checkBackend)
  useEffect(() => {
    checkBackend()
    const interval = setInterval(checkBackend, 5000)
    return () => clearInterval(interval)
  }, [checkBackend])

  return <aside className="studio-sidebar flex h-full flex-col border-r border-border">
    <div className="studio-brand flex items-center gap-3">
      <div className="brand-mark flex h-10 w-10 shrink-0 items-center justify-center"><BrandMark className="h-7 w-7" /></div>
      <div className="brand-copy min-w-0"><div className="text-[13px] font-semibold tracking-tight">Models at Home</div><div className="mt-1 text-[10px] tracking-[.16em] text-muted-foreground uppercase">Training studio</div></div>
    </div>
    <nav aria-label={ru ? 'Основная навигация' : 'Main navigation'} className="min-h-0 flex-1 overflow-y-auto px-3">
      {groups.map(group => <div key={group.en}>
        <p className="nav-group-label">{ru ? group.ru : group.en}</p>
        {group.items.map(item => <NavLink key={item.path} to={item.path} end={item.path === '/'} className="studio-nav-link" title={t(item.labelKey)} aria-label={t(item.labelKey)}>
          <span className="nav-icon"><item.icon className="h-4 w-4" /></span>
          <span className="nav-copy truncate">{t(item.labelKey)}</span>
        </NavLink>)}
      </div>)}
    </nav>
    <div className="px-3 pb-3 pt-2"><NavLink to="/settings" className="studio-nav-link" title={t('nav.settings')} aria-label={t('nav.settings')}><span className="nav-icon"><Settings className="h-4 w-4" /></span><span className="nav-copy">{t('nav.settings')}</span></NavLink></div>
    <div className="backend-panel flex items-center gap-2.5 border-t border-border px-5 py-4" role="status" title={`Backend: ${backendOnline ? 'Online' : 'Offline'}`}>
      <span className={`h-2 w-2 shrink-0 rounded-full ${backendOnline ? 'bg-success' : 'bg-warning'}`} />
      <div className="backend-copy flex-1"><div className="text-[11px] font-medium">{ru ? 'Локальный backend' : 'Local backend'}</div><div className="mt-0.5 text-[10px] text-muted-foreground">{backendOnline ? (ru ? 'Готов к работе' : 'Connected to workspace') : (ru ? 'Нет подключения' : 'Waiting for connection')}</div></div>
    </div>
  </aside>
}
