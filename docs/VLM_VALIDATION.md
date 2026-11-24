# VLM: собственный RL-движок

Обновлено 23 ноября 2026. Текущий GRPO для VLM использует **наше ядро `homellm.training.rl`**, а не TRL GRPOTrainer. Исходники соседнего `models-at-home` не изменялись; мультимодальная интеграция находится в Studio.

## Что переиспользуется

Из существующего RL-модуля напрямую импортируются:

- `GRPOLoss`, `SDPOLoss`, `create_loss_function` — наши objective, clipping, KL, token/sample aggregation и distillation;
- `compute_advantages` — собственная групповая нормализация GRPO/Dr.GRPO;
- `Experience`, `ReplayBuffer.append_group` — собственный буфер и фильтрация постоянных reward-групп;
- `compute_overlong_penalty` — собственный shaping для DAPO;
- `BinaryReward` и интерфейс пакетных наград;
- конфигурация, которую использует наш существующий trainer (`legacy_config.GRPOConfig`, совместимая с его loss API).

Новый `backend/services/vlm_native_trainer.py` добавляет транспорт изображений, генерацию, мультимодальные policy/reference/teacher forward, optimizer lifecycle и checkpoint. Математика перечисленных алгоритмов не переписана через сторонний trainer. `vlm_grpo.py` содержит адаптеры данных, конфигурации и выбора log probabilities.

## Алгоритмы

- **GRPO:** собственный GRPOLoss, std-нормировка advantages, sample-level aggregation.
- **Dr.GRPO:** тот же native loss с mean-centering без std и фиксированным знаменателем.
- **DAPO:** собственные token-level loss и Clip Higher; фильтрация нулевых reward-групп через ReplayBuffer, ограниченный добор новых групп, собственный overlong penalty. Если лимит добора исчерпан, процесс сообщает ошибку, а не выдаёт пустое обновление за успешное обучение. При buffer=0 helper сохраняет исходную семантику ядра; настраиваемый buffer задаёт мягкую штрафную зону.
- **SDPO:** собственный SDPOLoss (GRPO + selected-token self-distillation). Успешная траектория становится демонстрацией в teacher reprompt. В teacher-контекст повторно передаются **те же изображения**, completion actions выравниваются отдельно от другого prompt prefix. Проверяются ненулевой distillation loss и вызовы teacher.

BNPO/GSPO, добавленные в предыдущем TRL-пути, удалены из текущего списка: они не являются исходными вариантами нашего RL-ядра. Настройки full-logit/top-k distillation, EMA teacher и Liger fused objective в текущую VLM-обвязку не перенесены. Baseline SDPO использует detached student как teacher и сохранённые успешные решения.

## Обычный backend и Unsloth

В обоих случаях алгоритм — **наш native RL**. Обычный путь использует Transformers/PEFT для модели, Unsloth — FastVisionModel для её загрузки/адаптеров и ускорения. Unsloth переключается между generation/training; policy loss остаётся нашим. Не используются TRL GRPOTrainer, TRL loss или его replay loop.

Для обычного пути TRL не требуется. Профили разделены:

```bash
.venv/bin/python -m venv --system-site-packages .venv-vlm
.venv-vlm/bin/python -m pip install -r backend/requirements-dev.txt -r backend/requirements-vlm-modern.txt
# Необязательный ускоритель:
.venv-vlm/bin/python -m pip install -r backend/requirements-vlm-unsloth.txt
```

Unsloth сам зависит от TRL и импортирует его внутри библиотеки; соответствующий pin сохранён только в optional Unsloth-профиле. Наличие этой зависимости не означает использование её GRPO алгоритма.

Новые модели требуют отдельного HF окружения: `.venv-vlm` автоматически используется для VLM workers, либо задаётся `MAH_VLM_PYTHON`. Основной API/text runtime остаётся на прежнем Python (`MAH_PYTHON`). На проверке Transformers 5 ломал legacy Home tie_weights, поэтому глобальное обновление не выполнялось.

Проверенный профиль: Torch 2.9.1+cu128, torchvision 0.24.1, Transformers 5.17.0, PEFT 0.22.0, bitsandbytes 0.50.2, Accelerate 1.12.0, Pillow 12.0.0. Optional Unsloth/Unsloth Zoo 2026.11.9 совместимы с TRL 1.15.0; torchao 0.14.1 сохранён как совместимый с Torch 2.9 профиль, его квантование не используется. CUDA Torch/torchvision устанавливаются по [официальной матрице](https://pytorch.org/get-started/previous-versions/).

## Передача изображений и обучение

Последняя эталонная assistant-реплика удаляется из генеративного prompt и остаётся только reward-эталоном. Поля processor (`pixel_values`, grid и прочие) сохраняются при генерации и каждом policy/reference/SDPO teacher forward. Веса reference для LoRA — та же замороженная база с отключёнными адаптерами; Full использует отдельную замороженную базовую модель.

Вероятности считаются только для completion actions. Первый EOS сохраняется, padding после EOS исключается, в том числе при PAD=EOS. Генерация image/video placeholders без пикселей запрещена. Expanded prompt + completion budget проверяется без обрезания image tokens; teacher reprompt также должен помещаться в seq_len.

`batch_size` в native RL — число prompt-групп, `gradient_accumulation` — множитель этих групп на optimizer update. Forward/backward выполняется по одному completion, чтобы ограничить VRAM; суммарная нормировка сохраняет native sample-level или token-level objective для всего накопленного update. Это отличается от SFT microbatch. Dropout отключён для сравнимости rollout и policy probabilities.

Метрики содержат `rl_engine=homellm_native`, полное имя фактического native loss-класса, rewards/std, KL, grad norms, отфильтрованные/сгенерированные группы, ответы rollout, teacher forwards и distillation loss. При постоянных rewards GRPO показывает предупреждение; обновление только из-за KL не считается reward-driven обучением.

## Checkpoint и resume

Native checkpoint сохраняет модель/адаптер, processor, optimizer, scheduler, scaler, RNG Python/Torch/CUDA, позицию данных, успешные SDPO решения и служебные счётчики. Нет незавершённого replay batch: сохранение происходит после optimizer update. Отпечаток проверяет настройки модели, точности, objective и optimizer, содержимое JSONL и локальный model config. Числа нормализованы, поэтому `-1` и `-1.0` не ломают совместимость.

`vlm_worker_version=3`, формат состояния `homellm_native_v1`, файлы `native_state.pt` и `native_manifest.json`. Старые reward-filtered SFT и TRL optimizer checkpoints не возобновляются под новым движком. Они остаются в истории; для нового native run нужна база/совместимые веса. Общий continue выбирает только полные native checkpoints.

Проверка `--verify-resume` сравнивает непрерывное обучение с resume из checkpoint-2 при одинаковом плане. На обычном native GRPO получены **побитово одинаковые** конечные адаптеры.

## Модели и проверки

Текущие пресеты — Qwen3-VL 2B/4B Instruct, Qwen3.5 0.8B/2B/4B и Gemma 3 4B. Gemma — ещё не проверенный локально gated-пресет. Каталог recipes подтверждает только фактически проверенный engine/method/backend/objective; старые TRL-результаты не переносятся автоматически на native RL.

GPU smoke запускает настоящую tiny архитектуру, processor и изображения, проверяет native loss class, ненулевые rewards/std/градиенты, изменение весов, optimizer resume 2→4, экспорт и image inference. SDPO дополнительно проверяет teacher и ненулевой distillation loss. Отдельные прогоны используют официальные pretrained веса Qwen3.5-0.8B, revision `2fc06364715b967f1860aea9cf38778875588b17`.

```bash
PYTHONPATH=/path/to/models-at-home .venv-vlm/bin/python scripts/smoke-vlm.py --stage vlm_grpo --method lora --algorithm grpo --verify-resume
PYTHONPATH=/path/to/models-at-home .venv-vlm/bin/python scripts/smoke-vlm.py --stage vlm_grpo --method lora --algorithm dapo
PYTHONPATH=/path/to/models-at-home .venv-vlm/bin/python scripts/smoke-vlm.py --stage vlm_grpo --method lora --algorithm sdpo --backend unsloth
PYTHONPATH=/path/to/models-at-home .venv-vlm/bin/python scripts/smoke-vlm.py --family qwen3_5 --stage vlm_grpo --method lora --backend unsloth --reward-fn exact_match --model-path /path/to/Qwen3.5-0.8B --result /tmp/native-proof.json
```

SFT, современные пресеты, preflight и изолированный persistent inference сохранены. Preflight не загружает веса; проверяет первые 200 JSONL строк и три URL-примера, actual processor masks/lengths и native RL imports. При обучении JSONL разбирается строго целиком. Изменённые файлы требуют повторной проверки; содержимое изображений и самих базовых весов не входит в checkpoint fingerprint.

Исторический [TRL-отчёт](vlm-validation-2026-11-23.json) относится к заменённому пути и **не является свидетельством работы native RL**. Актуальный [native-отчёт по 22 успешным GPU-прогонам](vlm-native-validation-2026-11-23.json) содержит фактические классы loss и метрики. Проверены GRPO/Dr.GRPO/DAPO/SDPO через оба загрузчика на tiny Qwen3-VL; GRPO Full/LoRA/QLoRA через оба загрузчика; GRPO/SDPO LoRA на tiny и официальной pretrained Qwen3.5-0.8B через оба загрузчика. Отдельно прошли обычный GRPO с заблокированным импортом TRL и FP16 Full с scaler resume. Native resume по одинаковому плану побитово совпал с непрерывным обучением.

Итоговые проверки: **155 backend/ML-тестов** прошли в обоих Python-окружениях, **29 клиентских тестов**, TypeScript, Vite build и smoke готового Electron-пакета (native GRPO/SDPO, Unsloth, preflight, страницы, reload/backend health).

## Границы

Короткие синтетические GPU-прогоны подтверждают механику обновления, а не улучшение качества VQA/OCR. Не сертифицированы все размеры pretrained моделей, Gemma 3, MoE, видео, multi-GPU VLM/FSDP/DeepSpeed, длительные запуски, Windows/macOS. Сейчас выбирается одна GPU. Следующий качественный эксперимент — фиксированный независимый holdout с ответами до/после.
