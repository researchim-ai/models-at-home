# Models at Home Studio

Десктопное приложение для тренировки ML-моделей на своём железе.
Билдится под **Linux**, **Windows**, **macOS**.

Electron-окно с React UI внутри. Бэкенд на Python (FastAPI) запускается
автоматически при старте приложения: через Docker в режиме `Auto`, либо
нативно через локальный Python как fallback/ручной режим.


## World Models Studio

Отдельная `/world-models` страница: собственные Latent GRU и Gaussian RSSM,
обучение динамики по эпизодам с continuous/discrete/hybrid действиями,
optional states/rewards/dones, preflight, validation, checkpoints/resume и
предсказанные кадры в Rollout. Есть встроенное синтетическое демо.
Во вкладке «Готовые модели» встроены DIAMOND CS:GO и Atari (Breakout/Pong):
собственные локальные model workers, pretrained weights, игровой экран,
WASD/камера, step/pause/reset. Архитектуры и sampler включены в репозиторий.
Есть CPU-проба и GPU-режим; после скачивания весов inference работает офлайн.
Cosmos 3 оставлен дополнительным локальным pipeline.
Инструкция: [docs/WORLD_FOUNDATION_MODELS.md](docs/WORLD_FOUNDATION_MODELS.md).
Подробности: [docs/WORLD_MODELS_STUDIO.md](docs/WORLD_MODELS_STUDIO.md).

## VLA Studio

Отдельная страница **VLA Studio** обучает action-модели для робототехники,
игр/симуляторов и mouse/keyboard-сценариев. Есть непрерывные, дискретные
и смешанные action chunks, Native VLA с нуля и VLM + Action Head с
Full/LoRA/QLoRA и optional Unsloth. Встроены синтетические примеры,
preflight, метрики/графики, validation, checkpoints/resume и action preview.

Контракты JSONL, установка и проверенные границы — [docs/VLA_STUDIO.md](docs/VLA_STUDIO.md).

## Проверка и исправления (23 ноября 2026)

Актуальные изменения и границы проверки описаны в [docs/FIXES.md](docs/FIXES.md).
Первоначальная карта находится в [docs/PROJECT_MAP.md](docs/PROJECT_MAP.md).
Реальные VLM-прогоны, исправления, окружение и приоритеты: [docs/VLM_VALIDATION.md](docs/VLM_VALIDATION.md).
В VLM Studio перед запуском нажми **Проверить эксперимент**: будет показан пример
картинки и масок loss, проверены окружение и выборка данных без загрузки весов.

Актуальные VLM-пресеты: **Qwen3-VL, Qwen3.5 и Gemma 3** (Gemma пока без локальной GPU-проверки).
Для VLM подключены **наши GRPO / Dr.GRPO / DAPO / SDPO** из `homellm.training.rl`:
собственные losses, advantages и ReplayBuffer с мультимодальным policy/reference/teacher.
Выбор **Transformers или Unsloth** относится к модели и ускорению; TRL trainer
для GRPO не используется. Обычный профиль не требует TRL, optional Unsloth
имеет собственную зависимость от него. Новые VLM используют отдельную `.venv-vlm`
или `MAH_VLM_PYTHON`; основной API и текстовые модели сохраняют прежний Python.
Установка, текущие GPU-проверки и границы — в [руководстве VLM](docs/VLM_VALIDATION.md).

- ML-исходники могут лежать в `./models-at-home`, соседнем `../models-at-home`
  или в каталоге `MAH_RUNTIME_ROOT`. Сборка проверяет наличие runtime и включает его код.
- Native использует `MAH_PYTHON`, затем проектное `.venv`, затем Python из PATH.
  Проверяются импорты FastAPI, uvicorn, psutil, Accelerate и Transformers.
  Для другого интерпретатора: `MAH_PYTHON=/path/to/python npm run dev`.
- В установленной программе данные находятся в `userData/workspace`, а не в ресурсах
  приложения. `MODELS_AT_HOME_ROOT` позволяет выбрать другой workspace.
- Каждый новый запуск получает отдельный `output_dir/<run_id>`. Продолжение
  восстанавливает последний checkpoint; для увеличения законченного плана API принимает
  `POST /api/training/runs/{id}/continue` с `{"max_steps": <новый предел>}`.
  Legacy GRPO runtime не поддерживает resume — API возвращает явную ошибку.
- Закрытие или перезапуск backend останавливает принадлежащие ему дочерние задачи.
  Старые живые PID из неподтверждённых файлов не завершаются автоматически.
- Electron передаёт случайный токен сессии HTTP/WS-клиенту. Jupyter использует
  случайный токен; кнопка Start Jupyter получает его вместе с URL.
- Для standalone Compose сначала собери базовый ML-образ:
  `docker build -t models-at-home-base:latest ../models-at-home`, затем
  `docker compose up --build`. Профиль `notebooks` публикует отдельный Jupyter на 8889;
  его случайный токен показывается в логах. Управляемый через API Jupyter использует 8888.

Проверки: `npm run typecheck`, `npm test`,
`.venv/bin/python -m pytest backend/tests -q`, `npx vite build`.
Дополнительные интеграционные сценарии: `scripts/smoke-desktop.cjs` и
`scripts/smoke-training.py` (последнему нужен ML runtime на PYTHONPATH и GPU).
Он поддерживает `--multi-gpu` (GPU 0 и 1) и `--architecture home|gpt2|home_moe|blueprint`;
по умолчанию использует GPU 1. Проверяет pretrain, resume, inference и SFT chat/instruct
на временных локальных данных без скачиваний.

В Model Builder собери схему и нажми **Обучить эту модель**. Training Studio получит
её снимок; выбери tokenizer и данные перед запуском. **Edit JSON → Apply JSON**
позволяет редактировать связи графа с проверкой структуры. Для Blueprint проверены
pretrain, resume, inference и SFT; GRPO, Unsloth и QLoRA отклоняются явно.


---

## Быстрый старт

### Требования

- **Node.js** >= 20
- **Python** >= 3.10 + pip
- **Docker** + **NVIDIA Container Toolkit** — для GPU-тренировки (опционально)

### 1. Установка

```bash
# JS-зависимости
npm install

# Отдельный Python с ML- и API-зависимостями
python3 -m venv .venv
source .venv/bin/activate
pip install -e "../models-at-home[full,rl]"
pip install -r backend/requirements-dev.txt
```

На Windows активируй `.venv\Scripts\activate` вместо `source`. Если ML-runtime
находится в другом месте, используй его путь при установке и `MAH_RUNTIME_ROOT`.

### 2. Запуск (dev mode)

```bash
npm run dev
```

Это **одна команда**. Она:
1. Запускает Vite dev server (hot-reload React)
2. Компилирует Electron `main`/`preload`
3. Открывает **настоящее Electron-окно** (десктоп-прила, не браузер!)
4. Electron пробует поднять Docker backend (`models-at-home-studio:latest`)
5. Если Docker недоступен или контейнер не поднялся — fallback на локальный Python FastAPI
6. Порт подбирается автоматически (обычно `:8000`, при конфликте `:8001+`)
7. Пока бэкенд грузится — в окне показан splash-экран с прогрессом
8. Когда `/api/system/health` отвечает 200 — UI подменяется на рабочий

В sidebar внизу всегда видно статус бэкенда — `Online` / `Offline`.

**Single instance**: второй запуск `npm run dev` поднимет на передний план уже открытое окно.

**Где смотреть логи бэкенда:**
- **Меню приложения → Backend → Show Backend Logs**
- Или напрямую: `~/.config/Models at Home Studio/logs/backend.log` (Linux)
  - macOS: `~/Library/Logs/Models at Home Studio/backend.log`
  - Windows: `%APPDATA%\Models at Home Studio\logs\backend.log`

### 3. Остановка

Закрой окно Electron — бэкенд остановится автоматически.

---

## Сборка приложения

Собирает готовые инсталлеры для OS:

```bash
# Linux — AppImage + .deb (x64)
npm run build:linux

# Windows — .exe (NSIS installer) + portable .exe (x64)
npm run build:win

# macOS — .dmg + .zip (x64 + arm64)
npm run build:mac

# Текущая платформа
npm run build
```

Готовые файлы окажутся в `release/` с именем вида
`Models at Home Studio-0.1.0-linux-x64.AppImage`.

**Что попадает в сборку:**
- `dist/` — собранный React (renderer)
- `dist-electron/` — скомпилированный main/preload
- `resources/backend/` — Python FastAPI API
- `resources/models-at-home/` — ML-код (без `models/`, `datasets/`, `out/`, `logs/`, `.venv/`)
- `build/icons/` — иконки под все платформы

**Иконки:** `build/icon.png` (Linux/общая), `build/icon.ico` (Windows), `build/icon.icns` (macOS).

---

## Backend modes

В приложении есть три режима backend, выбрать можно в **Settings → Backend Mode**:

| Режим | Поведение |
|-------|-----------|
| `Auto` | Сначала Docker, если не вышло — локальный Python |
| `Docker` | Только Docker. Если Docker daemon/образ/контейнер не поднялись — будет ошибка на splash |
| `Native` | Только локальный Python (`python3 -m uvicorn backend.api:app`) |

Docker-образ состоит из двух слоёв:

- `models-at-home-base:latest` — тяжёлый ML runtime (CUDA, PyTorch, vLLM, Deepspeed и ML-код)
- `models-at-home-studio:latest` — финальный backend app image с FastAPI API (`backend/`)

Electron автоматически использует `models-at-home-studio:latest`. Если образа нет и
`dockerAutoBuild=true`, приложение соберёт его само. Первый build может занять долго
и занять десятки гигабайт.

В Docker backend переменная `MODELS_AT_HOME_ROOT=/app`, поэтому API, WebSocket метрик,
модели, датасеты, `.runs`, blueprints, configs и notebooks смотрят в один и тот же корень.

JupyterLab запускается из страницы **Notebooks**. В Docker-режиме Electron выбирает
свободный host-port начиная с `8888` и пробрасывает его в контейнер как
`JUPYTER_PUBLIC_URL`, поэтому backend не падает, если стандартный порт уже занят.

---

## Как это устроено

```
┌─────────────────────────────────────────┐
│         Electron Desktop App            │
│                                         │
│  ┌─────────────┐  ┌──────────────────┐  │
│  │ Main Process│  │ Renderer (React) │  │
│  │             │  │                  │  │
│  │ Стартует    │  │ 7 страниц:       │  │
│  │ Python      │  │  - Training      │  │
│  │ бэкенд     │  │  - VLM Studio    │  │
│  │ автоматом   │  │  - Model Builder │  │
│  │             │  │  - Study Center  │  │
│  │ Управляет   │  │  - Notebooks     │  │
│  │ Docker      │  │  - Agent Studio  │  │
│  │ (optional)  │  │  - Settings      │  │
│  └──────┬──────┘  └────────┬─────────┘  │
│         │                  │             │
└─────────┼──────────────────┼─────────────┘
          │                  │
          │        HTTP + WebSocket
          │                  │
  ┌───────▼──────────────────▼──────────┐
  │  FastAPI Backend (Python, :8000)    │
  │                                     │
  │  /api/training/*  — обучение        │
  │  /api/models/*    — модели          │
  │  /api/datasets/*  — датасеты        │
  │  /api/chat/*      — чат с моделью   │
  │  /api/agent/*     — AI-агент        │
  │  /ws/metrics/*    — метрики (WS)    │
  │                                     │
  │  subprocess → trainer_worker,       │
  │               train_rl, vllm, etc.  │
  └─────────────────────────────────────┘
```

При нажатии «Start Training» → FastAPI запускает `trainer_worker` как subprocess →
метрики пишутся в `.runs/{id}/metrics.json` → FastAPI читает и шлёт по WebSocket →
React рисует графики в реальном времени.

Backend закрепляет один канонический config contract перед запуском legacy worker:

- `dataset_path` из React превращается в `data_path` для старого ML-кода
- `num_epochs` синхронизируется с `epochs`
- `model_path` для SFT/continual/GRPO синхронизируется с `base_model_path`
- `lr_scheduler` синхронизируется с worker-полем `lr_schedule`
- `flash_attention`/`liger_kernel` синхронизируются с `use_flash_attention`/`use_liger`
- GRPO получает worker-поля `grpo_learning_rate`, `grpo_max_optim_steps`, `lora_r`, `grad_checkpoint`
- VLM получает `model_name_or_path`, `lora_r`, `tuning_method`, `lr_schedule`, `num_epochs`
- архитектурные поля (`hidden_size`, `num_layers`, `n_heads`, `seq_len`) сохраняются в `config.json`
- если обязательных данных нет, API возвращает `400`, а не запускает падающий subprocess
- если subprocess умер сразу после старта, API возвращает ошибку с хвостом `stderr.log`

Метрики старого worker (`current_step`, `current_loss`, `current_lr`) нормализуются
в WebSocket до полей React UI (`step`, `loss`, `learning_rate`).

---

## Проверенный training path

Проверено end-to-end:

1. **Мини pretrain 2 шага**:
   - HomeModel с нуля
   - tiny text dataset
   - `max_steps=2`
   - worker дошёл до `current_step=2`, loss и GPU stats записались

2. **Multi-GPU DDP pretrain 200 шагов**:
   - `parallel_mode=multi_gpu`
   - `accelerate_multi_gpu.yaml`
   - 2 GPU
   - HomeModel примерно 252M параметров
   - `hidden_size=1024`, `num_layers=12`, `seq_len=512`
   - результат: `status=completed`, `step=200/200`, `final_model` сохранён

Пример фактической нагрузки GPU из теста:

```text
10s: GPU0 4649 MiB, 87%  | GPU1 4283 MiB, 85%
20s: GPU0 4644 MiB, 80%  | GPU1 4283 MiB, 100%
25s: GPU0 4644 MiB, 100% | GPU1 4283 MiB, 99%
35s: GPU0 4655 MiB, 100% | GPU1 4283 MiB, 98%
```

Известный нюанс: в DDP-прогоне debug warning показал duplicate samples между
процессами на первых шагах. Это не ломает запуск/GPU usage, но для качества
долгих multi-GPU тренировок стоит отдельно улучшить шардирование датасета.

---

## Agent Studio и Notebooks

**Agent Studio** запускает `llama-server` через backend, скачивает выбранный GGUF
из Hugging Face в `models-at-home/models/`, проверяет `/health` на порту `8787` и
использует реальный `homellm.app.agent_runtime.run_agent_turn`. Tool calls идут из
trace рантайма и стримятся в UI как SSE-события. Сессии хранятся в
`models-at-home/.runs/agent_sessions/`.

**Notebooks** теперь не требуют ручного запуска. Кнопка **Start Jupyter** вызывает
backend route, который стартует `python -m jupyterlab` с root directory
`models-at-home/notebooks/` и токеном `mah-local` (или `JUPYTER_TOKEN`). В Docker
папка `notebooks/` примонтирована как volume, а внешний URL приходит из
`JUPYTER_PUBLIC_URL`.

Выполнение ячеек происходит внутри JupyterLab. Studio управляет запуском, статусом,
открытием notebook'ов и embed/openExternal, но не заменяет сам Jupyter kernel.

---

## Структура проекта

```
models-at-home-studio/
├── electron/               # Electron main process
│   ├── main.ts             #   Окно, бэкенд lifecycle, IPC
│   ├── preload.ts          #   contextBridge (безопасный API)
│   └── docker.ts           #   Docker management (dockerode)
├── src/                    # React frontend (renderer)
│   ├── pages/              #   9 страниц
│   │   ├── TrainingStudio  #   Обучение (7 вкладок)
│   │   ├── VLMStudio       #   Visual Language Models
│   │   ├── WorldModelsStudio #   Обучение динамики и imagined rollouts
│   │   ├── VLAStudio       #   Vision-Language-Action policies
│   │   ├── ModelBuilder    #   Визуальный конструктор
│   │   ├── StudyCenter     #   Учебные материалы
│   │   ├── Notebooks       #   JupyterLab
│   │   ├── AgentStudio     #   AI-агент (llama.cpp)
│   │   └── Settings        #   Настройки
│   ├── components/ui/      #   Компоненты (shadcn-стиль)
│   ├── api/                #   REST/WS клиент + React Query
│   ├── stores/             #   Zustand (training, settings, docker)
│   └── i18n/               #   Переводы en/ru
├── backend/                # FastAPI Python бэкенд
│   ├── api.py              #   Точка входа
│   ├── ws.py               #   WebSocket метрик
│   └── routes/             #   training, models, datasets, chat, agent, vlm, system, study
├── models-at-home/         # Исходный ML-код (не трогаем)
│   ├── homellm/            #   Модели, обучение, inference
│   ├── Dockerfile          #   CUDA + PyTorch
│   └── ...
├── docker-compose.yml      # Docker-оркестрация (опционально)
├── package.json
└── vite.config.ts
```

---

## npm-скрипты

| Команда            | Что делает                                     |
|--------------------|------------------------------------------------|
| `npm run dev`      | Запускает Electron-прилу в dev mode            |
| `npm run build`    | Production build + electron-builder            |
| `npm run build:linux` | Сборка под Linux (AppImage + deb)           |
| `npm run build:win`   | Сборка под Windows (NSIS .exe)              |
| `npm run build:mac`   | Сборка под macOS (.dmg)                     |
| `npm run typecheck`   | Проверка типов TypeScript                   |

---

## FAQ

**Q: Нужен ли Docker?**
В режиме `Auto` приложение сначала пробует Docker, потому что это самый
надёжный путь для CUDA/GPU окружения. Если Docker недоступен, Electron
fallback'нется на локальный Python backend. Для полноценной GPU-тренировки
рекомендуется Docker + NVIDIA Container Toolkit.

**Q: Где ML-код?**
В `models-at-home/homellm/`. Он не менялся — FastAPI просто вызывает те же
функции, что раньше вызывал Streamlit.

**Q: Как добавить свою страницу?**
1. Создай `src/pages/MyPage.tsx`
2. Добавь `lazy(() => import(...))` роут в `src/App.tsx`
3. Добавь навигацию в `src/components/layout/Sidebar.tsx`

**Q: Бэкенд не стартует, что делать?**
- Проверь установленные пакеты: `pip install fastapi "uvicorn[standard]" websockets pydantic`
- Проверь Docker: `docker info`
- Проверь GPU runtime: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
- Открой логи: меню **Backend → Show Backend Logs**
- Splash-экран в приложении сам покажет ошибку через 20 секунд и предложит перезапуск
- Убедись, что `python3 --version` работает в PATH (на Windows: `python --version`)

**Q: Порт 8000 занят?**
Электрон автоматически подберёт следующий свободный (8001, 8002, ...). React видит порт через IPC и общается с правильным.

**Q: Где данные пользователя?**
- Тренировочные ранчи, чекпоинты, датасеты — в `models-at-home/` (как в оригинале)
- Логи бэкенда — в стандартной директории ОС для логов приложения (см. выше)

---

## Стек

| Слой       | Технология                              |
|------------|-----------------------------------------|
| Desktop    | Electron 35                             |
| Frontend   | React 19, TypeScript, Vite 6            |
| Стили      | Tailwind CSS 4, shadcn/ui              |
| Графики    | Recharts                                |
| Стейт      | Zustand, React Query                    |
| i18n       | i18next (en + ru)                       |
| Backend    | FastAPI, uvicorn, WebSocket             |
| ML         | PyTorch, Transformers, Accelerate       |
| Inference  | vLLM, llama.cpp                         |
| Сборка     | electron-builder (Linux/Win/Mac)        |

---

## Лицензия

Apache 2.0
