import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface SettingsState {
  language: 'en' | 'ru'
  theme: 'dark' | 'light'
  apiUrl: string
  toggleLanguage: () => void
  setTheme: (theme: 'dark' | 'light') => void
  setApiUrl: (url: string) => void
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      language: 'en',
      theme: 'dark',
      apiUrl: 'http://localhost:8000',

      toggleLanguage: () =>
        set((s) => ({ language: s.language === 'en' ? 'ru' : 'en' })),

      setTheme: (theme) => {
        document.documentElement.classList.toggle('dark', theme === 'dark')
        set({ theme })
      },

      setApiUrl: (apiUrl) => set({ apiUrl }),
    }),
    { name: 'mah-settings' },
  ),
)
