import { useTranslation } from 'react-i18next'
import { useLocation } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Globe } from 'lucide-react'
import { useSettingsStore } from '@/stores/settingsStore'

const routeTitles: Record<string, string> = {
  '/': 'nav.training',
  '/vlm': 'nav.vlm',
  '/world-models': 'nav.worldModels',
  '/vla': 'nav.vla',
  '/builder': 'nav.builder',
  '/study': 'nav.study',
  '/notebooks': 'nav.notebooks',
  '/agent': 'nav.agent',
  '/settings': 'nav.settings',
}

export function Header() {
  const { t, i18n } = useTranslation()
  const location = useLocation()
  const toggleLanguage = useSettingsStore((s) => s.toggleLanguage)

  const titleKey = routeTitles[location.pathname] ?? 'nav.training'

  const handleLangToggle = () => {
    const newLang = i18n.language === 'ru' ? 'en' : 'ru'
    i18n.changeLanguage(newLang)
    toggleLanguage()
  }

  return (
    <header className="flex h-12 items-center justify-between border-b border-border px-6 app-drag-region">
      <h1 className="text-sm font-semibold">{t(titleKey)}</h1>
      <div className="flex items-center gap-2 no-drag">
        <Button variant="ghost" size="icon" onClick={handleLangToggle} title="Toggle language">
          <Globe className="h-4 w-4" />
        </Button>
      </div>
    </header>
  )
}
