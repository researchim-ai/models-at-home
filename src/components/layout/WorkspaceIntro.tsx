import { useTranslation } from 'react-i18next'
import { GraduationCap, Eye, Bot, Orbit } from 'lucide-react'
const content = {
  training: { icon: GraduationCap, eyebrow: 'Language models', title: ['Teach a model something new.', 'Научите модель новому.'], description: ['From your first dataset to a model of your own. Configure an experiment, check your setup, and follow every step.', 'От первого датасета до собственной модели. Настройте эксперимент, проверьте готовность и следите за каждым шагом обучения.'] },
  vlm: { icon: Eye, eyebrow: 'Vision + language', title: ['Give your model a new perspective.', 'Новый взгляд для вашей модели.'], description: ['Train on images and instructions. Fine-tune with SFT or explore our native GRPO algorithms, with Transformers or Unsloth.', 'Обучайте на изображениях и инструкциях: SFT или собственные алгоритмы GRPO, с Transformers и Unsloth.'] },
  vla: { icon: Bot, eyebrow: 'Vision + language + action', title: ['VLA Studio', 'VLA Studio'], description: ['Turn demonstrations into action policies. Work with continuous, discrete, and hybrid actions for robotics, simulations, and desktop tasks.', 'Превращайте демонстрации в действия: непрерывные, дискретные и смешанные action-модели для роботов, симуляторов и задач на компьютере.'] },
  world: { icon: Orbit, eyebrow: 'Learn the dynamics', title: ['World Models Studio', 'World Models Studio'], description: ['Explore local pretrained worlds or teach a model to predict what happens next. Check resources, train on episodes, and inspect imagined trajectories.', 'Исследуйте готовые миры или учите модель предсказывать следующий шаг. Проверьте ресурсы, обучите на эпизодах и изучите воображаемые траектории.'] },
}
export function WorkspaceIntro({ space }: { space: keyof typeof content }) {
  const { i18n } = useTranslation()
  const item = content[space]
  const language = i18n.language.startsWith('ru') ? 1 : 0
  return <section className="workspace-intro" aria-labelledby={`intro-${space}`}>
    <div><p className="workspace-eyebrow">{item.eyebrow}</p><h1 id={`intro-${space}`} className="workspace-title">{item.title[language]}</h1><p className="workspace-description">{item.description[language]}</p></div>
    <div className="workspace-symbol" aria-hidden="true"><item.icon className="h-8 w-8" strokeWidth={1.4} /></div>
  </section>
}
