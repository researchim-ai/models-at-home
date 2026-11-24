# Встроенные локальные world models

В **World Models → Готовые модели** есть локальная проба pretrained world models. Основной путь — DIAMOND с нейросетевыми блоками, denoiser и diffusion sampler **в репозитории Studio**, без внешнего inference-приложения. Облачная интеграция удалена.

## Что можно попробовать

**DIAMOND CS:GO**: 381,638,406 параметров вместе с upsampler. Стартовые истории наблюдений/действий из семи опубликованных авторами spawn-сцен; будущее рассчитывается рекуррентно по вашим действиям. Динамика генерирует 30×56, отдельная neural upsampling model — 150×280. Клавиши WASD, Space, камера стрелками, экранные кнопки. Веса около 1.53 GB; GPU рекомендуется, CPU поддержан, но медленнее.

**DIAMOND Atari**: pretrained Breakout и Pong, denoiser примерно 4.4M параметров, кадры 64×64, около 54 MB checkpoint на игру. Подходит для быстрой CPU-пробы. Для Breakout включён начальный screenshot; для Pong загрузите собственный. Это world-model inference: Atari ROM и эмулятор не запускаются. Первый screenshot повторяется в четыре context frames; полноценная исходная история движения при таком warm start неизвестна. Поэтому длинная траектория может отличаться от исходной игры.

Архитектуры и samplers перенесены из [официального DIAMOND](https://github.com/eloialonso/diamond), MIT Copyright (c) 2024 Eloi Alonso. Source revisions и изменения импортов записаны в [backend/vendor/diamond/NOTICE.md](../backend/vendor/diamond/NOTICE.md); оригинальная лицензия сохранена. Формулы denoising, pretrained key names и sampling сохранены. Нет зависимости от upstream pygame, Hydra launcher, внешнего repo или установленной игры.

Pretrained weights и CS:GO spawn assets: [eloialonso/diamond](https://huggingface.co/eloialonso/diamond), pinned revision `5d4abca9af6ab1b3ab1c6fc228cacb7281130f7f`. Веса загружаются по кнопке без обязательного API-ключа. После загрузки inference не обращается к сети.

Atari screenshot — атрибутированный CC BY-SA crop опубликованной DIAMOND-демонстрации. Подробности и лицензия производного изображения в [examples/world-local/ATTRIBUTION.md](../examples/world-local/ATTRIBUTION.md).

## Быстрый старт

1. Откройте «Готовые модели», выберите DIAMOND CS:GO или Atari.
2. Нажмите «Скачать локальные веса». Скачивается выбранная модель, не все checkpoints семейства.
3. Выберите CPU/CUDA, diffusion steps; для CS:GO — сцену и upsampler.
4. Нажмите «Загрузить модель». Начальный кадр отображается внутри Studio.
5. Нажмите «Играть», затем кликните по сцене для клавиатурного управления. Или используйте «Один шаг» и экранные кнопки.
6. «Пауза» останавливает генерацию, «Сбросить сцену» восстанавливает исходное состояние, «Освободить модель» завершает worker.

Ни одна клавиша не отправляется в ОС, внешнюю игру или робота. Действия — только conditioning tensors нейросети. Каждый шаг использует предыдущие предсказанные кадры и текущее действие; видео не воспроизводится вместо генерации. В CS:GO у всех mouse axes всегда есть ровно одна выбранная bin-категория, включая нейтральное положение; противоположные WASD исключаются.

Потеря фокуса окна приостанавливает autoplay и сбрасывает нажатия. Запросы шагов сериализованы: максимум один in-flight переход, без накопления очереди действий. Reset ожидает уже отправленный шаг, после чего очищает историю. Уход со страницы закрывает игровую сессию; idle timeout пять минут освобождает забытый worker. Закрытие API завершает модель. На Linux worker дополнительно получает parent-death signal.

Показываемый model FPS измеряет локальный model step и кодирование кадра; итоговая интерактивная частота также зависит от UI/HTTP. Это не обещание фиксированного realtime FPS на любом устройстве. Модель может менять детали, терять объекты, ошибаться в коллизиях и правилах игры.

## Зависимости и Python

DIAMOND требует Torch, NumPy, Pillow; Hugging Face Hub нужен для получения/cached lookup весов. Для отдельного Python с подходящим CPU/CUDA Torch:

```bash
/path/to/world-python -m pip install -r requirements-world-local.txt
MAH_WORLD_PYTHON=/path/to/world-python npm run dev
```

При пустом MAH_WORLD_PYTHON используется `.venv-world` либо `models/.world-venv` в backend workspace, затем существующий VLM Python и Python API. Symlinks Python venv сохраняются: интерпретатор не подменяется базовым Conda/system Python.

Папку с заранее скачанным snapshot можно указать в форме. Нужен snapshot layout `csgo/model/csgo.pt`, `csgo/spawn/...` либо `atari_100k/models/Breakout.pt` / `Pong.pt`. При пустом пути используется Hub cache **с local_files_only=True**. Отсутствие весов не вызывает скрытое обращение к inference API; сначала скачайте модель явной кнопкой.

В Docker подготовьте отдельный runtime с Torch в persistently mounted models directory:

```bash
docker exec mah-studio-backend python -m venv --system-site-packages /app/models/.world-venv
docker exec mah-studio-backend /app/models/.world-venv/bin/python -m pip install -r /app/requirements-world-local.txt
```

Его путь Studio обнаруживает автоматически. Дополнительный container path можно задать через MAH_DOCKER_WORLD_PYTHON. Полная пересборка тяжёлого Docker image/GPU-прогон контейнера здесь не выполнялась; сквозные проверки использовали native backend.

## Дополнительные Cosmos 3

Cosmos 3 Edge/Nano/Super сохранены как **локальные** pretrained pipelines для text/image/action-conditioned video. Они используют официальный `Cosmos3OmniPipeline` из Diffusers внутри ML worker. Для них отдельно устанавливаются `requirements-world-foundation.txt`, нужен BF16-capable CUDA GPU, память для выбранного размера и доступ Hugging Face к обязательным guardrail weights. Это дополнительный путь: DIAMOND-проба не зависит от Cosmos и этих gated весов.

Источники: [NVIDIA Cosmos](https://github.com/NVIDIA/Cosmos), [официальная Edge model card](https://huggingface.co/nvidia/Cosmos3-Edge). Для action modes используйте официальный embodiment-specific raw pose contract, а не произвольные нормализованные VLA vectors. T кратно 4, на выходе T+1 кадров. Контракты размеров проверяются до запуска.

## API и scope

Игровые сессии: `/api/world-models/play/start`, `/step`, `/reset`, `/stop`, `/status`. Сессия имеет случайный ID, запросы синхронизированы, stale ID не может остановить новую сессию. Нейросеть живёт в отдельном persistent Python process, без отдельного HTTP model server и без shell launchers.

Download / batch video / Cosmos: `/api/world-models/foundation/models`, `/preflight`, `/download`, `/generate`, `/jobs`, `/stop/{id}`, `/jobs/{id}/result`, `/jobs/{id}/video`.

Продуктовый scope этой версии — **локальная проба pretrained моделей**, download и autoregressive inference. Дообучение DIAMOND/Cosmos через Studio пока не подключено; training methods upstream сохранены в vendored denoiser, но это не замена готовому finetuning flow. Обучение наших Native GRU/RSSM и их checkpoints остаётся отдельной вкладкой.

## Проверка

```bash
.venv-vlm/bin/python scripts/smoke-local-world.py --model diamond_csgo --snapshot /path/to/snapshot --device cuda --upsample --result /tmp/csgo.json
.venv-vlm/bin/python scripts/smoke-local-world.py --model diamond_atari --snapshot /path/to/snapshot --device cpu --result /tmp/atari.json
```

Скрипт использует **настоящие pretrained weights**, блокирует outbound socket connections на время model operations, проверяет разные будущие кадры по действиям, точный reset/replay и сохраняет фактическое время шагов. [Отчёт](world-local-validation-2026-11-23.json).

Для собранного приложения `MAH_SMOKE_LOCAL_WORLD=1 MAH_SMOKE_DIAMOND_PATH=/path/to/snapshot` включает в desktop smoke реальную GPU загрузку, декодированный кадр, step/reset и закрытие сессии внутри Electron. Эти проверки подтверждают работоспособность локального контура, а не точность физики или игровой benchmark score.
