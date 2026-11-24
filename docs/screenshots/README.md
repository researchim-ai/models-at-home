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
