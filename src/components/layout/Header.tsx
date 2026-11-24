import { useTranslation } from 'react-i18next'
import { useLocation } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Globe, Moon, Sun, ChevronRight, HardDrive } from 'lucide-react'
import { useSettingsStore } from '@/stores/settingsStore'

const routeTitles: Record<string, string> = {
  '/': 'nav.training', '/vlm': 'nav.vlm', '/world-models': 'nav.worldModels', '/vla': 'nav.vla',
  '/builder': 'nav.builder', '/study': 'nav.study', '/notebooks': 'nav.notebooks', '/agent': 'nav.agent', '/settings': 'nav.settings',
}
export function Header() {
  const { t, i18n } = useTranslation()
  const location = useLocation()
  const { toggleLanguage, theme, setTheme } = useSettingsStore()
  const ru = i18n.language.startsWith('ru')
  const themeLabel = theme === 'dark' ? (ru ? 'Светлая тема' : 'Switch to light theme') : (ru ? 'Тёмная тема' : 'Switch to dark theme')
  return <header className="studio-header flex items-center justify-between gap-4 border-b border-border px-7 app-drag-region">
    <div className="flex min-w-0 items-center gap-3 text-xs"><span className="header-context text-muted-foreground">Studio</span><ChevronRight className="header-context h-3.5 w-3.5 text-muted-foreground" /><span className="truncate font-semibold">{t(routeTitles[location.pathname] ?? 'nav.training')}</span></div>
    <div className="no-drag flex shrink-0 items-center gap-1">
      <span className="header-context mr-4 flex items-center gap-2 text-[10px] text-muted-foreground"><HardDrive className="h-3.5 w-3.5" />{ru ? 'Ваше рабочее пространство' : 'Your workspace'}</span>
      <Button variant="ghost" size="icon" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} aria-label={themeLabel} title={themeLabel}>{theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}</Button>
      <Button variant="ghost" size="sm" aria-label={ru ? 'Switch to English' : 'Переключить на русский'} title={ru ? 'Switch to English' : 'Переключить на русский'} onClick={() => { i18n.changeLanguage(ru ? 'en' : 'ru'); toggleLanguage() }}><Globe className="h-3.5 w-3.5" /><span>{ru ? 'RU' : 'EN'}</span></Button>
    </div>
  </header>
}
