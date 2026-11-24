# Исправления и повторная проверка

Обновлено 23 ноября 2026. Исходная карта и находки до изменений сохранены в PROJECT_MAP.md.

## Встроенная локальная проба DIAMOND

Удалены облачный provider/SDK dependency и внешний Matrix CLI из активного World Models контура. В `backend/vendor/diamond` включены лицензированные MIT architectures, denoisers и EDM samplers Atari/CS:GO с сохранением upstream checkpoints и source revisions. Добавлен собственный persistent local worker, игровой экран Studio, WASD/камера, step/pause/reset, offline inference и освобождение GPU при уходе со страницы/idle/shutdown. Подготовлен небольшой атрибутированный стартовый пример Breakout; CS:GO использует настоящие history/action spawn assets авторов.

Реальные проверки: DIAMOND CS:GO + upsampler (381.6M) на RTX 3090 Ti; DIAMOND Breakout (4.4M) на CPU. Сетевые соединения заблокированы во время model operations: 0 обращений; разные действия меняют prediction; reset/replay совпадают побитово. Полный собранный Electron: загрузка pretrained DIAMOND на GPU, новые декодированные кадры, WASD/autoplay/pause/reset/release, CPU Breakout со встроенным screenshot; девять страниц, native world training/rollout, Builder handoff и reload прошли. 209 backend/ML-тестов в обоих Python, 35 клиентских, TypeScript/Vite build. Не выдаём опубликованные видео за generation. [Текущая инструкция](WORLD_FOUNDATION_MODELS.md), [фактический отчёт](world-local-validation-2026-11-23.json).

## История: прежний API/external-runtime подход, заменён встроенной локальной пробой

Добавлена отдельная вкладка готовых моделей с Cosmos 3 Edge/Nano/Super через официальный Diffusers pipeline, Matrix-Game 4 через отдельный upstream runtime и Genie 4 через Gemini Interactions API. Есть управляемые download/generation jobs, проверка model/task/action contracts, auth-protected MP4, previews, логи и остановка. Дообучение foundation weights явно не предлагается; native GRU/RSSM training сохранён отдельно.

Исправлены dereference symlink у Python venv (иначе запускался базовый интерпретатор), транзитивная зависимость atomic_json от FastAPI и обработка завершения платной Genie session: complete с acknowledgement, fallback cancel с terminal status, cooperative stop и bounded recording. API key не сохраняется в конфигурации.

Проверка: 206 backend/ML-тестов в обоих Python, 35 клиентских, TypeScript/Vite build и собранный Electron на всех девяти страницах. Настоящий pretrained Cosmos 3 Edge transformer (3,369,657,024 параметров) проверен на RTX 3090 Ti: finite forward и влияние действий на vision prediction; это не full guarded video generation. Она заблокирована HTTP 401 на обязательном gated guardrail без HF access/token. Genie live API без ключа и GPU inference отдельного Matrix runtime не запускались.

[Инструкция и scope](WORLD_FOUNDATION_MODELS.md), [фактический отчёт](world-foundation-validation-2026-11-23.json).

## World Models Studio

Добавлены собственные Latent GRU и Gaussian RSSM, отдельная страница `/world-models`, API и изолированный worker. Поддержаны три контракта действий, эпизодные окна без перехода через terminal, train-only normalization, optional state/reward/done targets, свободные rollout без будущих наблюдений, checkpoints с полным optimizer/scaler/RNG и fingerprint содержимого кадров. Исправлен FP16 overflow handling: batch повторяется после уменьшения scale, не засчитывая пропущенное обновление.

Проверка: 191 backend/ML-тест, 34 клиентских теста, TypeScript/Vite build, три настоящих CUDA-прогона (GRU continuous FP32, RSSM discrete BF16, RSSM hybrid FP16) с checkpoint extension и повторной загрузкой для rollout. Собранный Electron проверен на всех девяти страницах, включая preflight → training → export → image upload → rollout нового экрана. Фактический scope и проверки: [WORLD_MODELS_STUDIO.md](WORLD_MODELS_STUDIO.md).

## VLA Studio — обучение action policies

Добавлена отдельная `/vla` страница и API для трёх доменов: робототехника, игры/симуляторы и desktop. Собственная Native VLA и VLM + Action Head обучаются по демонстрациям; поддержаны continuous/discrete/hybrid chunks, train-only normalization, terminal masks, validation MAE/accuracy, полные checkpoints/resume и prediction из экспорта. Встроенные демо явно синтетические. Воркеры используют изолированный ML Python, process supervision и общий транспорт метрик/логов. Это behavior cloning, не фиктивный online RL без среды.

Проверено: 12 обучающих CPU/GPU-прогонов, 183 backend/ML-теста, 32 клиентских теста, TypeScript/build и полный desktop flow от демо до action prediction.

Подробности: [VLA_STUDIO.md](VLA_STUDIO.md).

## VLM поверх собственного RL-ядра — актуальная реализация

По уточнённому требованию TRL GRPOTrainer убран из пути VLM RL. Добавлена мультимодальная обвязка вокруг собственных `homellm.training.rl` GRPOLoss/SDPOLoss, compute_advantages, Experience/ReplayBuffer, BinaryReward и overlong penalty. DAPO получает native dynamic sampling с добором, SDPO — teacher reprompt с исходными изображениями. Unsloth отвечает за модель/ускорение; objective остаётся нашим. BNPO/GSPO из предыдущей сторонней интеграции убраны из выбора.

Native checkpoint version=3 хранит optimizer/scheduler/scaler/RNG/data cursor и успешные SDPO траектории; чужие TRL/legacy optimizer states отклоняются. В проверке одинакового плана native resume дал побитово одинаковые конечные адаптеры относительно непрерывного обучения. У обычного профиля больше нет требования TRL; optional Unsloth зависимости вынесены отдельно.

Native-проверка: 155 backend/ML-тестов в обоих окружениях, 29 клиентских тестов, TypeScript/Vite build и smoke готового Electron-пакета. 22 успешных GPU-прогона, включая pretrained Qwen3.5-0.8B GRPO/SDPO с обоими загрузчиками, заблокированный импорт TRL и FP16 Full. Метрики и фактические loss-классы — в [native-отчёте](vlm-native-validation-2026-11-23.json).

Актуальные проверки и ограничения: [VLM_VALIDATION.md](VLM_VALIDATION.md). Следующий раздел — история заменённого TRL пути; его GPU-результаты не переносятся на native реализацию.

## История: предыдущая TRL интеграция (заменена)


LLaVA/Qwen2 заменены в текущих пресетах и smoke на Qwen3-VL/Qwen3.5; Gemma 3 добавлена как ещё не проверенный локально пресет. Настоящий TRL GRPO заменил прежний reward-filtered SFT. Проверены GRPO, Dr.GRPO, BNPO, GSPO и DAPO loss с обычным загрузчиком и Unsloth, включая Full/LoRA/QLoRA на tiny Qwen3-VL. SFT и GRPO LoRA официальной pretrained Qwen3.5-0.8B проверены через оба backend с checkpoint/optimizer resume 2→4, изменением адаптеров и image inference. Эти короткие прогоны не доказывают улучшение качества на независимом наборе.

Создано отдельное окружение `.venv-vlm`: обновление Transformers основного API ломало legacy Home `tie_weights`, поэтому VLM preflight/training и постоянный inference вынесены в отдельные процессы. Поддержан `MAH_VLM_PYTHON`; старое окружение не обновлялось. Исправлены генерация image placeholders без пикселей, тихий пропуск невалидных JSONL строк, память на PIL всего GRPO-датасета и нестабильный experiment_name в React. Добавлены rewards/KL/grad norms, ответы rollout и предупреждение об одинаковых rewards. SDPO VLM отклонён с явной причиной; DAPO отмечен как loss без dynamic filtering.

Проверки: 141 backend/ML-тест и 29 клиентских; TypeScript и Vite build; готовый Electron-пакет; прежний Home pretrain/resume/inference/SFT. Краткий JSON по 23 успешным экспериментам — [vlm-validation-2026-11-23.json](vlm-validation-2026-11-23.json).

Весь актуальный протокол, окружение, команды и ограничения: [VLM_VALIDATION.md](VLM_VALIDATION.md). Следующие разделы описывают историю исправлений до 20 ноября; прежние утверждения об отключённом VLM GRPO и отсутствии pretrained/Unsloth проверок заменены этим разделом.


## Дополнительные исправления 20 ноября

Дополнено реальной VLM-проверкой: Full/LoRA/4-bit QLoRA для маленьких LLaVA и Qwen2-VL, включая checkpoint/optimizer resume и inference из экспорта; отдельно LLaVA pretrain с проверкой freeze. Исправлены пути картинок, dtype, assistant-only masking, EOS=PAD, processor tensors, LoRA export/loading и VLM-мониторинг. VLM GRPO отключён: legacy реализация оказалась reward-filtered SFT. Подробности, воспроизводимые команды и ограничения — в [VLM_VALIDATION.md](VLM_VALIDATION.md).

Добавлен processor-only preflight: отдельный процесс выбранной GPU, без загрузки весов; проверка окружения, локальных weight shards, выборки train/validation, изображений, длины и целей loss. UI показывает картинку и маски токенов, блокирует запуск до успешной проверки и сбрасывает результат при изменении настроек. Добавлены лимиты seq_len/pixels и precision в форме, API каталога версионируемых рецептов и предупреждения о непроверенных режимах/версиях. Границы проверки указаны явно; прямой API start сохранён для автоматизации.

- Model Builder передаёт сохранённую схему и её снимок в Training Studio. Гиперпараметры и данные выбираются перед запуском; кнопка Builder сама не запускает обучение. Run сохраняет immutable-снимок схемы, поэтому редактирование исходного JSON не меняет resume.
- Схемы проверяются без Torch и выполнения пользовательских блоков: уникальные допустимые ID, порядок зависимостей, циклы/пропущенные ссылки, неиспользуемые ветки, основные размерности, heads, residual Add, параметры экспертов и контекста. Текущий runtime поддерживает несколько входов только у Add; остальные такие графы отклоняются явно. Проверка структуры не заменяет проверку поведения произвольного custom_op/inline_code.
- Добавлен редактор JSON с проверкой перед применением. Невалидный текст не попадает в модель; вложенные параметры сохраняются, до применения черновика Save/Train отключены. Размеры стандартных существующих embedding/MLP/linear-блоков синхронизируются при изменении общих размеров; пользовательские ширины сохраняются.
- Blueprint-модели проходят через слой совместимости HF: сериализация конфигурации, gradient checkpointing, загрузка экспорта, inference и SFT. Размер словаря синхронизируется с tokenizer в корне схемы и явных token_embedding.
- Исправлены два тихих сбоя блоков: positional_embedding в последовательности добавляет позиции к токенным признакам; linear с равной входной/выходной шириной остаётся обучаемым слоем и учитывает bias. Новая семантика помечена studio_graph_version=1; старые экспорты без маркера сохраняют прежнюю семантику при загрузке.
- У MoE параметр выбора экспертов передаётся как num_select; старый top_k преобразуется и проверяется. Убраны неработающие поля палитры и исправлены defaults позиционных/liger-блоков.
- Исправлен выбор «локальный checkpoint» в UI: значение checkpoint нормализуется на входе API для fine-tuning.
- Оценка VRAM готовой локальной модели берёт размеры из её config.json, включая GPT-2 aliases и text_config. Значения прежнего Home-пресета больше не заменяют реальную архитектуру. Устаревший асинхронный результат не показывается после изменения конфигурации. Для произвольных blueprint-графов оценка явно недоступна.
- При ошибке записи process.json или pid supervisor завершает новый worker, очищает регистрацию и по возможности сохраняет error, вместо фонового неучтённого процесса.

Проверен короткий offline GPU-цикл Blueprint на одной и двух GPU: API save → train 2 шага → resume до 4 → export/inference → SFT chat с packing → SFT instruct с assistant-only. Отдельно проверено, что позиционный блок сохраняет зависимость выхода от входных токенов, а одинаковая по ширине linear остаётся nn.Linear. Полные GRPO/Unsloth/QLoRA для Blueprint явно не поддерживаются; общий статус непроверенных режимов ниже.

## Дополнительные исправления 19 ноября

- Supervisor сохраняет PID, время создания и случайный маркер процесса. После аварии backend восстанавливает управление только при совпадении всех признаков. Живой неподтверждённый PID блокирует остановку и удаление run; чужому процессу сигнал не отправляется. Добавлена зависимость psutil, включая Docker и проверку Native Python.
- Jupyter сохраняет токен между рестартами backend; параллельные Start сериализованы, неподтверждённый сервер не выдаётся за управляемый. Изменение заданного токена перезапускает только принадлежащий Studio сервер.
- API отдаёт сохранённые истории метрик; UI загружает их после подключения и перезагрузки. Старый HTTP-ответ не перезаписывает более новые WS-точки или другой выбранный run. Validation loss привязывается к собственным шагам оценки; NaN/Infinity превращаются в null и не ломают JSON/график.
- Добавлен text_worker — слой совместимости Studio с установленным ML-движком. Home, GPT-2 и MoE выбираются явно, независимо от model ID токенизатора. Сохраняются intermediate_size, параметры экспертов и token IDs. Для дообучения тип определяется по сохранённой модели.
- SFT получает выбранный формат chat/instruct. Для стандартного runtime работают all-token/assistant-only loss и packing; упаковка сохраняет маски, удаляет padding и дополняет последний блок. Длинные примеры по-прежнему ограничены seq_len до упаковки.
- Проверка контекста останавливает SFT/continual и custom-model inference до CUDA-сбоя при превышении вместимости модели. Усилены проверки чисел, precision, optimizer, scheduler, экспертов и GPU IDs.
- Inference умеет загружать экспортированные GPT-2 и MoE. Для этих двух legacy-реализаций отключён несовместимый DynamicCache: это сохраняет корректность, но генерация может работать медленнее.
- Distributed launch передаёт фактические precision и число выбранных GPU. Для multi_gpu нельзя выбрать единственный GPU; один GPU в FSDP/DeepSpeed не запрещён.
- У VLM GRPO интерфейс задаёт эпохи: текущий движок игнорирует max_steps, поэтому API отклоняет такой параметр и неподдерживаемый resume. VLM tuning_method синхронизирован с use_lora.
- Выбранный Unsloth SFT не подменяется другим backend при отсутствии пакета. Неподдерживаемые текущим Unsloth resume, packing и assistant-only отклоняются явно. Сам Unsloth на этой машине не проверялся.

## Исправлено

### Обучение и результаты

- Исправлена загрузка сохранённого `betas`: API принимает строку и пару чисел.
- Legacy aliases синхронизируются до применения defaults; SFT sequence length и chat template доходят до worker.
- Добавлены проверки этапов, распределённых режимов, чисел, архитектуры и обязательных путей. При отсутствии accelerate-конфига запуск отклоняется.
- Каждый run получает отдельный абсолютный output directory. История читает фактический каталог; Chat получает готовый путь final_model от API.
- Resume выбирает checkpoint по числовому шагу, передаёт его worker и сохраняет происхождение запуска. Закончившийся план требует явного увеличения max_steps.
- Исправлены CLI-контракты: GRPO запускается через адаптер к `--config_json`, VLM получает обязательный `--metrics`. Dr.GRPO преобразуется в enum legacy runtime, а group size проверяется до запуска.
- HF pretrain использует model ID и инициализацию с нуля; путь базовой модели больше не включает случайную загрузку готовых весов. SFT/continual требуют скачанную локальную модель, как ожидает текущий worker.
- Реальное состояние процесса возвращается отдельно от текстового статуса метрик. Удалять или повторно запускать активный run нельзя.

### Процессы и подключение

- Native restart ждёт завершения конкретного старого процесса. Таймер и exit-handler не могут остановить или обнулить новый backend.
- Запуск Native подтверждается HTTP health, а не строкой лога. Рестарты сериализуются; неготовый Docker-контейнер останавливается перед fallback.
- Python выбирается через MAH_PYTHON, локальное .venv или PATH, с проверкой API/ML-импортов.
- Добавлено владение worker-процессами, остановка process group, ожидание выхода и обновление конечного статуса при падении. При завершении API выполняется cleanup.
- Длительные синхронные операции перенесены из async event loop в обработчики/итераторы thread pool. Состояние загруженных моделей защищено от одновременного изменения.
- HTTP и WS получают текущий порт, без кеша при импорте. После переподключения обновляются queries.
- SSE decoder сохраняет границы событий, UTF-8 и остаток между пакетами; ошибки Chat и Agent больше не проглатываются. Abort завершает reader; уход со страницы отменяет клиентский stream.
- WS передаёт изменения на том же шаге, включая конечный статус; повторяет чтение частично записанного JSON. История графика сохраняется при переключении вкладок одного run. Мониторинг переподключается и обновляет последнюю точку шага без дубликата.

### Inference

- Transformers-модель и tokenizer повторно используются между запросами.
- Ошибка генерации в отдельном потоке завершает streamer и возвращается клиенту вместо бесконечного ожидания.
- Экспортированный Home model загружается через HomeForCausalLM. Для моделей без chat template применяется текстовое форматирование.
- Tokenizer не передаёт неподдерживаемый Home-моделью token_type_ids; generation token IDs берутся из локального tokenizer.
- В режиме без sampling не передаются параметры temperature/top_p/top_k.

### Файлы и доступ

- Общие safe_child/safe_id закрывают выход из каталогов через абсолютные пути, `..`, symlink и подмену ID. Сам корневой каталог нельзя удалить как dataset/model.
- Сохранения blueprint, run config и session выполняются атомарно.
- HTTP и WebSocket проверяют origin и токен desktop-сессии. Docker host ports ограничены loopback. DockerManager удаляет/останавливает только контейнер с точным именем и собственной owner-меткой; чужой или старый немаркированный контейнер не удаляется автоматически.
- Jupyter получает случайный токен; readiness проверяет именно этот токен и настроенный порт. Порты двух вариантов запуска Jupyter в Compose больше не конфликтуют.
- Внешние URL из renderer ограничены HTTP, HTTPS и mailto; Electron sandbox включён.

### Интерфейс и поставка

- HashRouter работает при загрузке file:// и полной перезагрузке установленного приложения.
- Model Builder переподключает последовательность при перемещении/удалении блока. Непоследовательный граф не переставляется молча. Boolean-параметры редактируются checkbox, ID блоков уникальны.
- VLM running определяется сервером, а просмотр завершённого run не делает его активным обучением.
- Light theme меняет реальные цветовые токены; настройки темы восстанавливаются. Повреждённый localStorage не останавливает i18n.
- Исправлены preview CSV/Arrow/Parquet и listing GGUF-файлов. Ограниченная загрузка датасета больше не собирает все строки в список.
- Долгая сборка Docker не объявляется ошибкой только из-за истечения 30 секунд.
- Сборка явно находит ML runtime, исключает тяжёлые пользовательские данные и использует writable workspace отдельно от ресурсов приложения.
- typecheck покрывает renderer, Electron и Vite; добавлены backend regression tests, усилены page smoke tests и добавлен CI workflow.

## Что проверено

- 121 backend/ML-тест с доступным ML runtime на PYTHONPATH: VLM masks/tensors, preflight без загрузки весов, ошибки данных/CUDA/shards, лимиты выборки, рецепты и таймауты. Четырнадцать ML-тестов пропускаются в API-only окружении.
- 29 клиентских тестов: прежние страницы/transport/Builder, VLM loss/errors/GRPO guard и блокировка запуска с защитой от устаревшего preflight.
- TypeScript renderer + Electron/Vite, Python compileall, git diff --check и docker compose config.
- Production Vite build и сборка Linux directory package с локальным Electron.
- Реальное Electron-приложение с настоящими preload и Python API: все семь страниц, полная перезагрузка file:// и штатное закрытие; тестовые данные изолированы в /tmp.
- Короткие offline GPU-прогоны: Home на двух GPU (RTX 3090 Ti + RTX 3090), GPT-2 и MoE на одной GPU. Для каждого: pretrain до шага 2, checkpoint resume до шага 4, разные output directories, inference из экспорта, SFT chat с packing/all-token loss и SFT instruct с assistant-only loss. Tokenizer и данные синтетические; скачиваний нет.

## Окружение и оставшиеся границы

Базовый Python этой машины содержит несовместимые NumPy/scikit-learn: ML-импорт падал до начала обучения. Общие пакеты не изменялись. Создан локальный `.venv` проекта с доступом к уже установленным ML-пакетам окружения `modelsathome`; API/test-зависимости установлены только в `.venv`. Для отдельной поставки нужно указать совместимый Python через MAH_PYTHON или установить её зависимости.

Большие pretrained VLM, GRPO, Unsloth, FSDP и DeepSpeed тренировки, длительные multi-GPU прогоны, обучение больших моделей, скачивания моделей из Hugging Face, macOS/Windows и установщики AppImage/deb/NSIS/DMG не сертифицированы этими прогонами. Доказана механика коротких текстовых и VLM циклов; это не гарантия качества модели или поддержки каждой комбинации параметров и сторонних ML-пакетов.

Старые GRPO/VLM GRPO и Unsloth SFT runtime не реализуют полноценное восстановление optimizer/data state; Studio явно отклоняет такой resume. Для старых живых PID без подтверждённого владения API отказывается отправлять сигнал; они могут потребовать ручной проверки. Новые процессы имеют проверяемую запись для восстановления после аварии backend.

Рефакторинг большой TrainingStudio, визуальный редактор DAG со связями мышью, GPU-очередь и durable event history остаются отдельными улучшениями архитектуры. Связи графа уже можно редактировать через проверяемый JSON. Неоднозначность MIT в package.json и Apache 2.0 в старом README не менялась: выбор лицензии требует решения владельца.
