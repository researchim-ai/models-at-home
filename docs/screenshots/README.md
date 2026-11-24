# Скриншоты студии

Это реальные кадры из собранного Linux-приложения с работающим Python backend,
в окне 1440 × 900. Изображения не содержат установщиков, весов моделей или кешей.

Для обновления сначала соберите приложение, затем запустите desktop smoke-проверку:

```bash
MAH_PYTHON=/path/to/python \
MAH_DESKTOP_SCREENSHOT_DIR=/tmp/mah-screenshots \
node scripts/smoke-desktop.cjs /path/to/linux-unpacked/models-at-home-studio
```

Нужен графический сеанс (или Xvfb). Проверка использует отдельное временное рабочее
пространство. Она снимает разделы в тёмной теме и проверяет светлую тему, ширину
1024 px и сохранение настроек. В README используются `training-studio.png`,
`world-models.png` и `training-studio-light.png`; остальные кадры нужны для проверки UI.
Копируйте в этот каталог только выбранные изображения, а не всю папку сборки.

Для скриншота конструктора с общим recurrent core добавьте `MAH_SMOKE_BUILDER=1`.
Файл `model-builder-looped.png` используется в руководстве Model Builder.

Для Research Lab включите `MAH_SMOKE_RESEARCH=1`: выполняется реальный CPU probe,
проверяются ссылки на исследования, граф, baseline и передача в VLM/VLA Studio.
`research-lab.png` — выбранный светлый кадр `research-lab-light.png` из этого сценария.

`MAH_SMOKE_ACT_TRAIN=1` с настроенным `MAH_LEROBOT_PYTHON` обучает настоящий ACT
в собранном приложении и снимает `vla-act-training.png` с метриками и экспортом.
