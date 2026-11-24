# Карта и разбор Models at Home Studio

**Это исходный снимок до исправлений. Текущее состояние и результаты повторной проверки: [FIXES.md](FIXES.md).**

Дата: 18 ноября 2026. Основание: исходники текущей рабочей копии и локальные проверки.

## Вывод

Это настольная студия для локальных экспериментов с LLM/VLM: подготовка данных, настройка обучения, мониторинг, inference, конструирование архитектур, Jupyter и агент с инструментами. По состоянию кода — функционально широкий прототип, которому прежде всего нужна стабилизация выполнения и хранения экспериментов.

Главный приоритет — добиться воспроизводимого цикла «подготовить → запустить → сохранить → продолжить → проверить модель». Сейчас между этими шагами есть подтверждённые разрывы. Расширять список алгоритмов до их устранения будет менее полезно.

## Границы исследования

- Изучены все прикладные модули `src/`, `electron/`, `backend/`, конфигурации сборки, Docker, тесты, локализация и README. Сгенерированные дистрибутивы, бинарные иконки и сторонние `node_modules` не рассматриваются как авторский исходный код.
- В этих трёх каталогах — **59 файлов, 11 581 строка**, включая CSS, JSON переводов и тесты. Это размер исходников, а не оценка сложности или покрытия тестами.
- React содержит **7 страниц**; backend — **10 HTTP-групп, 56 HTTP-обработчиков и 1 WebSocket**.
- `models-at-home/` внутри репозитория отсутствует. Соседний `/home/user/workspace/models-at-home` существует и доступен текущему Python как пакет `homellm`. Выборочно проверены контракты worker: output directory и resume. Полный аудит соседнего ML-проекта в этот раз не выполнялся.
- README описывает прошлые GPU-прогоны. Их результаты не считаются проверкой текущей копии: GPU-обучение, загрузка моделей и установщики в этом исследовании не запускались.
- Изменены только аналитические материалы. Исправления прикладного кода не выполнялись.

## Архитектура и зависимости

```text
Пользователь
  ├─ React renderer: src/main.tsx → App.tsx → BackendBoot → Layout
  │    ├─ страницы → src/api/hooks.ts → React Query
  │    ├─ src/stores/* → Zustand (форма, выбранный run, настройки, backend)
  │    ├─ src/api/client.ts → HTTP /api/*, SSE chat/agent, WS metrics
  │    └─ window.electronAPI → preload.ts → IPC
  │
  └─ Electron main: electron/main.ts
       ├─ окно, меню, выбор файлов, открытие внешних ссылок
       ├─ config.ts → config.json в userData
       └─ запуск backend
            ├─ Auto: Docker, затем Native при неудаче
            ├─ Docker: docker.ts → dockerode → контейнер FastAPI
            └─ Native: Python → uvicorn backend.api:app

FastAPI: backend/api.py
  ├─ routes/training.py → homellm.app.trainer_worker / RL worker
  ├─ routes/vlm.py      → homellm.training.vlm_* / Transformers
  ├─ routes/chat.py     → Transformers / vLLM / llama.cpp
  ├─ routes/agent.py    → llama-server :8787 → homellm agent runtime/tools
  ├─ routes/models.py + datasets.py → Hugging Face / локальные файлы
  ├─ routes/notebooks.py → отдельный JupyterLab-процесс
  ├─ routes/blueprints.py → JSON + homellm registry/validation
  ├─ routes/study.py + system.py → Markdown, GPU, конфиги, health
  └─ ws.py → чтение .runs/<id>/metrics.json раз в секунду

ML workspace (по MODELS_AT_HOME_ROOT либо встроенному пути):
  datasets/ · models/ · out/ · .runs/ · blueprints/ · configs/
  notebooks/ · study_materials/
```

React 19 + TypeScript + Vite; Tailwind 4; Zustand; TanStack Query; Recharts; i18next. Electron управляет окружением, FastAPI адаптирует запросы, `homellm` выполняет основную ML-работу. Отдельной БД или устойчивой очереди задач в Studio нет: состояние сохраняется в JSON и PID-файлах.

### Точки входа и зоны ответственности

- [src/main.tsx](/home/user/workspace/models-at-home-studio/src/main.tsx) — React, QueryClient, BrowserRouter.
- [src/App.tsx](/home/user/workspace/models-at-home-studio/src/App.tsx) — lazy-маршруты, оболочка загрузки backend.
- [electron/main.ts](/home/user/workspace/models-at-home-studio/electron/main.ts) — процесс приложения, запуск/остановка backend, IPC.
- [electron/preload.ts](/home/user/workspace/models-at-home-studio/electron/preload.ts) — ограниченный bridge в renderer; Node integration отключён, context isolation включён.
- [electron/docker.ts](/home/user/workspace/models-at-home-studio/electron/docker.ts) — два слоя образа, контейнер, volumes, GPU, health.
- [backend/api.py](/home/user/workspace/models-at-home-studio/backend/api.py) — регистрация API и CORS.
- [backend/paths.py](/home/user/workspace/models-at-home-studio/backend/paths.py) — разрешение ML workspace.
- [src/api/client.ts](/home/user/workspace/models-at-home-studio/src/api/client.ts) — общий транспорт; [types.ts](/home/user/workspace/models-at-home-studio/src/api/types.ts) — вручную поддерживаемые контракты.

## Карта пользовательских модулей

### 1. Training Studio — `/`

Файл: [TrainingStudio.tsx](/home/user/workspace/models-at-home-studio/src/pages/TrainingStudio.tsx), **2357 строк**.

Семь вкладок: Launch, Monitoring, Chat, History, Data, Models, Docs. Поддерживает настройки pretrain, continual pretrain, SFT, GRPO; full/LoRA/QLoRA; выбор распределённого режима; оценку VRAM; загрузку данных и моделей; предпросмотр данных; шаблоны чата; просмотр логов и метрик.

Зависимости: `trainingStore`, training/model/dataset hooks, `/api/training`, `/api/models`, `/api/datasets`, `/api/chat`, `/api/system`, `/ws/metrics/{run_id}`.

Страница объединяет сразу несколько самостоятельных областей. Это основная точка риска изменений и лучший кандидат на разделение по функциям.

### 2. VLM Studio — `/vlm`

Файл: [VLMStudio.tsx](/home/user/workspace/models-at-home-studio/src/pages/VLMStudio.tsx).

Запуск VLM pretrain/SFT/GRPO, история и inference «изображение + текст». Конфигурация и флаг running живут локально в React; после переключения вкладок или ухода со страницы UI теряет часть состояния, хотя worker может продолжать работу. Нужен тот же механизм run lifecycle, что у текстового обучения.

### 3. Model Builder — `/builder`

Файл: [ModelBuilder.tsx](/home/user/workspace/models-at-home-studio/src/pages/ModelBuilder.tsx).

Палитра блоков, упорядоченный список экземпляров, параметры, JSON preview, сохранение/загрузка blueprint. Сейчас это редактор списка, а не полноценный редактор графа. Связи `inputs` существуют в JSON, но UI не позволяет явно их редактировать. Прямой передачи выбранного blueprint в Training Studio нет.

### 4. Study Center — `/study`

Файл: [StudyCenter.tsx](/home/user/workspace/models-at-home-studio/src/pages/StudyCenter.tsx).

Локальный индекс `study_materials/<language>/**/*.md`, fallback на английский, отображение через ReactMarkdown. Это отдельный источник учебного контента от встроенной вкладки Docs.

### 5. Notebooks — `/notebooks`

Файл: [Notebooks.tsx](/home/user/workspace/models-at-home-studio/src/pages/Notebooks.tsx).

Управляет отдельным JupyterLab, проверяет доступность, показывает существующие `.ipynb` и доступные шаблоны, открывает их во внешнем браузере или iframe. Ячейки исполняет Jupyter, Studio лишь управляет подключением.

### 6. Agent Studio — `/agent`

Файл: [AgentStudio.tsx](/home/user/workspace/models-at-home-studio/src/pages/AgentStudio.tsx).

Настройка GGUF/llama-server; вкладки Chat, Runs, Tools, Capabilities. Сессии хранятся в `.runs/agent_sessions`. Backend вызывает `run_agent_turn`, адаптирует JSON-ответы модели, после завершения отдаёт trace и текст через SSE. Реального постепенного получения tool-событий во время выполнения здесь пока нет.

### 7. Settings — `/settings`

Файл: [Settings.tsx](/home/user/workspace/models-at-home-studio/src/pages/Settings.tsx).

Выбор backend mode, управление Docker, язык и тема. Настройки разделены между Electron `config.json` и persisted Zustand в localStorage. Указанный в settingsStore `apiUrl` транспортом не используется.

## Главные потоки данных

### Запуск приложения

Electron создаёт окно и backend manager → renderer показывает BackendBoot → main пробует Docker/Native → сообщает BootPhase по IPC → после online открываются страницы. Порт передаётся через bridge. Дальше sidebar периодически проверяет health.

### Обучение

Форма Zustand → JSON POST `/api/training/start` → Pydantic → нормализация legacy-полей → `.runs/<id>/config.json` + начальные метрики → subprocess worker → PID и логи → worker пишет metrics → WebSocket отправляет снимки → React хранит последние 500 точек и рисует графики.

История перечитывает каталоги каждые 5 секунд. Мониторинг перечитывает run каждые 3 секунды. Выбранный run и история графика не восстанавливаются из серверного active_run при перезапуске интерфейса.

### Chat и Agent

Chat: клиент отправляет историю → backend загружает/использует модель → SSE → обновление последнего сообщения. Agent: сессия → синхронный agent turn → trace/text → SSE → сохранение JSON-сессии. Отмена транспорта на клиенте не связана с надёжной остановкой вычислений на backend.

## Что уже сделано хорошо

- Ясное разделение desktop, renderer и API; собственно ML находится за адаптером.
- Lazy loading страниц, ErrorBoundary, smoke-тесты рендера.
- React Query централизует server state; Zustand отделяет форму и пользовательские настройки.
- BootPhase и backend-логи помогают диагностировать первый запуск.
- Есть нормализация legacy-метрик и форматов сессий, раннее обнаружение падения worker.
- Есть основные части полного экспериментального цикла: данные, модели, запуск, мониторинг, история, inference.

## Найденные проблемы и рекомендуемые исправления

Обозначения: **P1** — исправить до использования на важных данных или выпуска; **P2** — следующая очередь. «Воспроизведено» означает локальную проверку с синтетическими данными без запуска обучения. «По коду» означает установленное поведение реализации, но не полный end-to-end прогон.

### F01 · P1 · Продолжение обучения падает и не задаёт checkpoint

**Воспроизведено.** [TrainingRequest.betas](/home/user/workspace/models-at-home-studio/backend/routes/training.py:50) имеет тип `str`, но нормализация сохраняет его как список float. [continue_training](/home/user/workspace/models-at-home-studio/backend/routes/training.py:329) читает сохранённый config и снова строит TrainingRequest: обычная конфигурация с `[0.9, 0.95]` вызывает ValidationError.

Кроме того, endpoint не выбирает checkpoint и не передаёт `resume_from_checkpoint`. В соседнем worker это поле используется для восстановления optimizer/scheduler/data position. Оно отсутствует в схеме Studio. После исправления betas всё ещё нельзя считать этот endpoint настоящим resume.

**Исправить:** единая каноническая схема сохранения; отдельный ResumeRequest; явный путь checkpoint; проверка совместимости; тест «сохранённый config → resume → продолжение global step».

### F02 · P1 · Результаты запусков не изолированы и история ищет их по другой схеме

**По коду Studio и контракту соседнего worker.** По умолчанию все запросы получают `output_dir='out'`; start не добавляет отдельный каталог эксперимента. Worker пишет непосредственно в `config['output_dir']`, например `out/final_model`. История ищет `out/<experiment_name>/checkpoint*`, а кнопка Chat конструирует `output_dir/<experiment_name или run_id>/final_model`.

Это создаёт риск перезаписи результатов и неверных ссылок на checkpoint/model. Ссылки: [start](/home/user/workspace/models-at-home-studio/backend/routes/training.py:116), [поиск checkpoints](/home/user/workspace/models-at-home-studio/backend/routes/training.py:244), [путь Chat](/home/user/workspace/models-at-home-studio/src/pages/TrainingStudio.tsx:1651), [worker output](/home/user/workspace/models-at-home/homellm/app/trainer_worker.py:1190).

**Исправить:** backend назначает абсолютный `artifact_dir` для каждого run; сохраняет его в run record и возвращает готовые пути/ID артефактов. UI ничего не достраивает самостоятельно.

### F03 · P1 · Перезапуск Native может остановить новый backend

**По коду.** [stopBackend](/home/user/workspace/models-at-home-studio/electron/main.ts:483) назначает отложенный SIGKILL через общую переменную `backendProcess`, затем обнуляет её. Если до срабатывания таймера стартует новый процесс, таймер обращается уже к нему. Exit-handler старого процесса также без проверки обнуляет общую ссылку.

**Исправить:** захватывать конкретный ChildProcess; ждать завершения; очищать таймер; менять общую ссылку только при совпадении экземпляра. Проверять рестарт с задержанным завершением старого процесса.

### F04 · P1 · Клиент продолжает обращаться к старому порту

**Воспроизведено.** [client.ts](/home/user/workspace/models-at-home-studio/src/api/client.ts:1) кеширует порт и `basePromise` при импорте. При смене mock-порта с 8000 на 8001 второй health всё равно обращался к 8000. Bootstrap и пользовательские рестарты могут менять порт после этой инициализации.

**Исправить:** общий объект подключения, обновляемый по ready-событию; инвалидировать соединения/queries при смене backend generation; таймаут и AbortSignal для запросов.

### F05 · P1 · SSE теряет события и скрывает ошибки Chat

**Воспроизведено.** В [Chat parser](/home/user/workspace/models-at-home-studio/src/api/client.ts:252) и Agent parser каждый сетевой chunk разбирается отдельно. Разделённый на два chunk JSON `{"text":"hello"}` не дошёл до onChunk. Событие `{"error":"OOM"}` Chat parser игнорирует: onError не вызывается.

**Исправить:** один SSE decoder с остаточным буфером, обработкой разделителей событий, error/done, корректным завершением и отменой. Проверить разбиение каждого события в произвольной позиции, Unicode, EOF и отмену.

### F06 · P1 · Неодинаковая защита файловых путей и открытая сетевая поверхность

**Чтение вне корня воспроизведено на синтетических файлах.** [preview_dataset](/home/user/workspace/models-at-home-studio/backend/routes/datasets.py:119) не проверяет принадлежность пути DATASET_DIR. [Study document](/home/user/workspace/models-at-home-studio/backend/routes/study.py:44) сравнивает строковый префикс: каталог `study_materials-extra` проходит проверку для `study_materials`.

Имена сохранения в model/dataset download также соединяются с корнем без containment check. В delete для models/datasets такая проверка уже есть, но допускается сам корневой каталог. Некоторые run/session IDs используются как пути без общего валидатора.

У API нет аутентификации; CORS разрешает все origins. Docker port bindings не задают HostIp, Compose публикует порты без ограничения loopback. Jupyter имеет известный токен `mah-local` по умолчанию. Фактическая внешняя доступность зависит от Docker и firewall; сетевой pentest не проводился.

**Исправить:** общий safe child path (`resolve` + проверка относительного положения и запрет самого корня), строгие ID/имена, ограничения размеров и limit; bind host ports к 127.0.0.1; случайные токены на сессию; проверка HTTP/WS origin. В Electron ограничить схемы внешних URL и проверить возможность включения sandbox.

### F07 · P1 · Длительные операции блокируют API

**По коду.** `async def` напрямую вызывает синхронные загрузки Hugging Face, model load/generate, urllib, `time.sleep`. [Agent turn](/home/user/workspace/models-at-home-studio/backend/routes/agent.py:505) выполняется целиком до отправки первого содержательного события. [Chat streamer](/home/user/workspace/models-at-home-studio/backend/routes/chat.py:167) синхронно ждёт очередь токенов внутри async generator.

Пока операция выполняется, event loop этого backend может задерживать health, stop и другие запросы. Простое увеличение числа uvicorn workers разнесёт глобальное состояние моделей и сессий между процессами и не решит архитектурную проблему.

**Исправить:** для файловых/сетевых операций thread offload; для ML отдельный управляемый worker; API возвращает job ID; очередь событий сообщает прогресс и поддерживает отмену. Начать с небольшого JobManager без добавления внешней инфраструктуры.

### F08 · P1 · Упакованный запуск не подтверждён и имеет разрывы

**По коду.** [Electron loadFile](/home/user/workspace/models-at-home-studio/electron/main.ts:142) загружает `file://.../dist/index.html`, а [BrowserRouter](/home/user/workspace/models-at-home-studio/src/main.tsx:22) использует pathname без basename/адаптации. Под file:// pathname не равен `/`, значит маршруты страниц не соответствуют пути документа. Это нужно проверить на packaged smoke и заменить на HashRouter либо выделенный app protocol.

`package.json.extraResources`, DockerManager и Compose ожидают `./models-at-home`, которого в репозитории нет. Текущая доступность пакета через соседний проект маскирует неполноту установки. Данные создаются рядом с ресурсами приложения, что для установленной программы может оказаться недоступным для записи.

**Исправить:** явная установка/версия ML runtime; отдельный writable workspace в userData или выбранном каталоге; packaged smoke на чистом окружении. Vite build прошёл, включая относительные asset paths; полный installer не проверялся.

### F09 · P2 · Нет единого владельца жизненного цикла задач

Run хранится как config/metrics/PID, без supervisor. Нет проверки принадлежности PID и process group при stop; возможен повторно использованный PID. Delete может удалить каталог работающего run. При остановке API нет централизованного cleanup обучения/Jupyter/llama-server. Один `active_run.json` перезаписывается при новых запусках. При неудачном Docker health перед fallback контейнер не останавливается.

**Исправить:** RunManager с блокировками, process groups, проверкой identity процесса и явными переходами статуса; запрет удаления активных run; политика поведения при закрытии приложения; восстановление состояния после сбоя. JSON сначала можно сохранить, но записывать атомарно.

### F10 · P2 · Метрики и состояние интерфейса расходятся

[ws.py](/home/user/workspace/models-at-home-studio/backend/ws.py:35) отправляет данные только при изменении step: status/error/reward на том же шаге теряются. mtime фиксируется до успешного чтения JSON. WS reconnect отсутствует. График очищается при монтировании Monitoring и хранит только 500 точек в памяти.

`activeRunId` означает и выбранный исторический run, и «обучение запущено»: просмотр завершённого run приводит к кнопке Stop в Launch. VLM использует локальный running, не привязанный к серверному статусу. Текстовая история перебирает и каталог agent_sessions, поскольку не фильтрует каталоги без config.

**Исправить:** разделить selectedRunId и выполняющиеся задачи; серверный статус как источник истины; события со sequence/version; отправка изменения всего snapshot, reconnect и загрузка истории с диска.

### F11 · P2 · Дублирующиеся конфиги дают неоднозначные значения

**Воспроизведено.** TrainingRequest с `n_heads=16` после нормализации содержит `num_heads=8`; `flash_attention=false` соседствует с `use_flash_attention=true`; `lora_rank=64` — с `lora_r=32`. Причина — обе версии полей имеют defaults, поэтому fallback не отличает явно заданное значение от default.

Также `seq_len` по умолчанию 2048 перекрывает изменённый `sft_max_seq_length` в нормализации. Pydantic не ограничивает этапы enum-типом и большинство чисел допустимыми диапазонами. При отсутствии accelerate config распределённый режим молча запускается без accelerate.

**Исправить:** одна каноническая схема по stage, legacy aliases только на входе; cross-field validation (hidden_size % heads, warmup/steps, LoRA, precision/GPU); `extra='forbid'` после переходного периода; TS-типы из OpenAPI; явная ошибка при неподдерживаемом режиме.

### F12 · P2 · Builder меняет порядок списка, но не граф

[moveBlock](/home/user/workspace/models-at-home-studio/src/pages/ModelBuilder.tsx:149) переставляет элементы, не меняя inputs. Созданная цепочка A→B→C после перестановки визуально выглядит другой, но JSON-связи остаются прежними. Удаление просто удаляет ссылку, не переподключая цепь. Поля boolean редактируются как строки; оценка params не учитывает ряд реальных параметров блоков; изменение hidden size не обновляет уже созданные параметры embedding.

**Исправить:** выбрать модель редактора: последовательность с автоматическим построением inputs либо настоящий DAG. Добавить типизированные поля, проверку связности/циклов/размерностей, backend validation и действие «использовать blueprint в обучении».

### F13 · P2 · Проверки качества и первый запуск не покрывают ключевые сценарии

Есть только один файл с 8 smoke-тестами на mock fetch. Проверка базовых страниц ожидает текст sidebar и ждёт фиксированное время, а не обязательно готовность нужной lazy-страницы. Ошибки пойманные ErrorBoundary не во всех формах гарантированно попадут в текущий сборщик console.error. В глубоком Agent-тесте legacy session может не быть выбрана, и это не проваливает тест.

Backend-тестов и CI workflow в репозитории нет. `npm run typecheck` проверяет только src; Electron пришлось проверить отдельной командой. Python requirements заданы диапазонами, часть зависимостей ставится Compose-командой на старте с подавлением stderr. Docker и Compose расходятся по сборке и параметрам запуска.

**Исправить:** API contract tests с временным workspace, регрессии F01/F04/F05/F06; UI action tests; Electron packaged smoke; минимальный training smoke с явно подготовленным runtime; CI typecheck renderer+main, pytest, vitest, Vite build. Развести liveness API и readiness ML runtime.

## Что улучшить в продукте после стабилизации

1. **Мастер первого запуска.** Выбор каталога данных, обнаружение runtime/Python/GPU, понятный список доступных функций. Сейчас health возвращает ok независимо от готовности ML; пользователь может увидеть Online, но получить ошибку при первой операции.
2. **Мастер эксперимента.** Сценарий → модель → данные → проверка формата/VRAM → запуск. Редкие параметры вынести в Advanced, сохранив доступ для опытных пользователей.
3. **Реестр экспериментов.** У каждого run: immutable config, seed, версии runtime/dependencies, dataset/model revision, artifact_dir, checkpoint provenance, итоговые метрики. Сравнение run, экспорт конфигурации и клонирование эксперимента.
4. **Отчёт о данных до запуска.** Схема колонок, число примеров, длины, пропуски, train/validation split, корректность chat template. Preview должен поддерживать заявленные Arrow/Parquet и сохранённые datasets, а не только TXT/JSONL.
5. **Управляемые фоновые задачи.** Прогресс и отмена загрузки/индексации/обучения; оценка свободного места; очередь GPU-задач. Обучение, Chat, VLM и Agent сейчас могут конкурировать за VRAM без общего распределителя.
6. **Честные статусы и ошибки.** Показать причину ошибки API, а не generic toast; сохранить черновики; не сбрасывать ongoing stream при переключении страницы; Stop generation; восстановление после перезапуска окна.
7. **Согласованный интерфейс.** Полная локализация en/ru, настоящая light theme (сейчас CSS-токены всегда тёмные), клавиатурная доступность Tabs/Tooltip, подписи icon buttons и полей, обработка длинных путей.
8. **Упорядочить документацию.** README говорит Apache 2.0, package.json — MIT, Settings — Apache 2.0; определить лицензию отдельно для Studio и ML ядра. Добавить реальный bootstrap и матрицу поддержанных OS/GPU режимов. Не обещать работающий backend только по наличию Python.

## Предлагаемый порядок работ

### Этап 1 — надёжность основного цикла

F01/F02: канонический config, уникальные artifact paths, resume из checkpoint. Затем F03/F04/F05: безопасный restart, актуальное подключение, SSE decoder. Параллельную разработку для этого плана не требуется: зависимости между изменениями лучше сохранить явными.

**Критерий готовности:** два маленьких эксперимента не перезаписывают друг друга; interrupted run продолжает step/optimizer state; после рестарта UI соединяется с новым портом; поток не теряет текст и показывает ошибки.

### Этап 2 — границы доступа и управляемое выполнение

F06/F07/F09: safe paths, loopback/token, worker manager и события. Устранить расхождения состояния F10/F11.

**Критерий готовности:** чтение/запись вне workspace отклоняется; во время долгой задачи отвечают health и stop; после сбоя/перезапуска задачи имеют достоверный статус.

### Этап 3 — поставка

F08/F13: закреплённый ML runtime, отдельный writable workspace, единый Docker путь, packaged routing, CI.

**Критерий готовности:** установка на чистую поддержанную систему открывает страницы, явно проверяет runtime и проходит один короткий train → checkpoint → resume → inference сценарий.

### Этап 4 — удобство экспериментов

Разбить TrainingStudio по `features/training`, `features/datasets`, `features/models`, `features/chat`; вынести повторяемые backend операции из routes в `services` (сейчас каталог почти пуст); наладить Builder, подготовку данных и сравнение run. Добавлять новые ML-возможности после этого.

## Выполненные проверки

- `npm run typecheck` — успешно.
- `tsc -p tsconfig.node.json --noEmit --tsBuildInfoFile /tmp/mah-review-node.tsbuildinfo` — успешно; отдельно покрывает Electron/Vite.
- `npm test` — **8/8**, один test file; все с mock backend.
- `vite build --outDir /tmp/mah-review-renderer` — успешно, также собраны main/preload. Это не сборка установщика.
- FastAPI ASGI-проверка с временным MODELS_AT_HOME_ROOT: health, local models, agent capabilities — 200. Capabilities использует доступный в текущем окружении соседний homellm, поэтому это не доказательство автономности поставки.
- Временный config с нормализованным betas → continue: ValidationError. Worker не запускался.
- Нормализация legacy aliases: подтверждены противоречивые значения.
- Настоящий client.ts транспилирован и выполнен с mock IPC/fetch: после смены порта адрес остался старым; split SSE потерял hello; SSE error не вызвал onError.
- Синтетические файлы во временном каталоге: datasets preview прочитал файл вне datasets; Study прочитал файл из соседнего каталога с совпадающим префиксом.

Не выполнялись: установка готового приложения, UI-проверка реального Electron-окна, GPU-тренировка, внешние загрузки, multi-GPU и сетевой pentest. Утверждения о рисках lifecycle/packaging основаны на коде и отдельно помечены выше.
