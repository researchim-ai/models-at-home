import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import en from './en.json'
import ru from './ru.json'

let language = 'en'
try {
  const saved = JSON.parse(localStorage.getItem('mah-settings') ?? 'null')
  if (saved?.state?.language === 'ru') language = 'ru'
  document.documentElement.classList.toggle('dark', saved?.state?.theme !== 'light')
} catch { /* Ignore corrupt persisted preferences. */ }

i18n.use(initReactI18next).init({
  resources: {
    en: { translation: en },
    ru: { translation: ru },
  },
  lng: language,
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
})

export default i18n
