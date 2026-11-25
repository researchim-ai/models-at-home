# Разборы статей в Learning Lab

Это второй уровень учебной библиотеки Models at Home: после коротких вводных уроков
можно разобрать механизм из статьи, написать его на PyTorch и проверить конкретные
свойства. Откройте **Ноутбуки → Подборка → Разборы статей · с экспериментами**.
Язык материалов следует языку интерфейса. Поиск работает по названию урока, оригинальному
названию статьи и авторам; фильтр темы применяется вместе с фильтром подборки.

Все десять разборов доступны на русском и английском, работают на CPU, не обращаются
к API и не скачивают веса/датасеты. Нужны PyTorch и Matplotlib в Python-окружении backend.
Для чтения первоисточника потребуется интернет. Год в карточке — год первой arXiv версии,
он может отличаться от года конференции или последующей версии статьи.

## Как устроен разбор

Сначала сформулированы задача, цель и необходимые знания. Затем — формула в читаемой
текстовой нотации, значения переменных, формы тензоров и последовательная реализация.
Проверки показывают не только изменение loss: например, сохранение функции при merge,
отсутствие future leakage или неизменность результата при перестановке индексов вместе
с labels. После эксперимента есть интерпретация результата, задания и пояснения к ответам.
В конце предложено записать гипотезу, условия, измерения и ограничения вывода.

В каждом ноутбуке указана **граница воспроизведения**. Код написан специально для уроков,
а не скопирован из авторских репозиториев. Это реализация отдельных механизмов по статье,
не полный reproduction её архитектуры, датасета и benchmark-результатов. Изменённые
conventions, например выбор std в GRPO или recency weighting в ensemble, обозначены явно.

## Архитектуры и языковые модели

1. **Attention Is All You Need — Vaswani et al., 2017.**
   Q/K/V, scaling, causal mask, attention heatmap, градиенты проекций и проверка
   неизменности ранних outputs при изменении будущего.
   [Статья](https://arxiv.org/abs/1706.03762) ·
   [RU](../models-at-home/notebooks/papers/ru/01-attention.ipynb) ·
   [EN](../models-at-home/notebooks/papers/en/01-attention.ipynb).
2. **RoFormer — Su et al., 2021.**
   RoPE как поворот пар координат, частоты, сохранение нормы, relative-offset identity
   и отличие математической эквивалентности от способности к context extrapolation.
   [Статья](https://arxiv.org/abs/2104.09864) ·
   [RU](../models-at-home/notebooks/papers/ru/02-rope.ipynb) ·
   [EN](../models-at-home/notebooks/papers/en/02-rope.ipynb).
3. **LoRA — Hu et al., 2021.**
   Frozen base, матрицы A/B, асимметрия первых градиентов, controlled low-rank target,
   проверка merge и ловушка нулевой инициализации обеих матриц.
   [Статья](https://arxiv.org/abs/2106.09685) ·
   [RU](../models-at-home/notebooks/papers/ru/03-lora.ipynb) ·
   [EN](../models-at-home/notebooks/papers/en/03-lora.ipynb).
4. **Universal Transformers — Dehghani et al., 2018.**
   Общее causal ядро и depth signal; одинаковый forward при разных full/truncated
   gradients; оценка чувствительности к inference depth. Adaptive halting не реализован.
   [Статья](https://arxiv.org/abs/1807.03819) ·
   [RU](../models-at-home/notebooks/papers/ru/04-universal.ipynb) ·
   [EN](../models-at-home/notebooks/papers/en/04-universal.ipynb).

Рекомендуемый порядок: основы обучения → Attention → RoPE → LoRA → Universal Transformers.
После этого можно менять похожие блоки в Model Builder и сравнивать варианты в Research Lab.

## Мультимодальность и обучение по предпочтениям

5. **CLIP — Radford et al., 2021.**
   Два encoder, нормализация embeddings, symmetric contrastive loss, retrieval и
   перестановка пар. Используются синтетические признаки вместо image/text towers.
   [Статья](https://arxiv.org/abs/2103.00020) ·
   [RU](../models-at-home/notebooks/papers/ru/05-clip.ipynb) ·
   [EN](../models-at-home/notebooks/papers/en/05-clip.ipynb).
6. **DPO — Rafailov et al., 2023.**
   Sequence log-probs, response masks, фиксированная reference, initial loss=log(2),
   проверка знаков градиентов и влияние нормировки по длине.
   [Статья](https://arxiv.org/abs/2305.18290) ·
   [RU](../models-at-home/notebooks/papers/ru/06-dpo.ipynb) ·
   [EN](../models-at-home/notebooks/papers/en/06-dpo.ipynb).
7. **DeepSeekMath / GRPO — Shao et al., 2024.**
   Group normalization, constant rewards, token masks, old policy и reference как
   разные роли, KL estimator и clipping для advantages разных знаков.
   Это unit experiment objective на log-probabilities, не полноценный rollout trainer.
   [Статья](https://arxiv.org/abs/2402.03300) ·
   [RU](../models-at-home/notebooks/papers/ru/07-grpo.ipynb) ·
   [EN](../models-at-home/notebooks/papers/en/07-grpo.ipynb).

CLIP логично проходить после вводного VLM. DPO и GRPO — после causal LM, масок
и вводных уроков по preference/RL. Доступность этих механизмов в конкретном training
разделе зависит от выбранной модели; наличие учебного ноутбука её не расширяет.

## Генеративные модели и политики действий

8. **DDPM — Ho, Jain & Abbeel, 2020.**
   Forward noise schedule, oracle reconstruction, epsilon prediction и reverse chain.
   Обучается одномерная смесь Gaussians; проверяются finite samples и PNG-гистограммы.
   Это не воспроизведение DIAMOND или image generation benchmark.
   [Статья](https://arxiv.org/abs/2006.11239) ·
   [RU](../models-at-home/notebooks/papers/ru/08-ddpm.ipynb) ·
   [EN](../models-at-home/notebooks/papers/en/08-ddpm.ipynb).
9. **Flow Matching — Lipman et al., 2022.**
   Linear conditional path, target velocity, endpoint/finite-difference checks,
   обучение поля и Euler sampling с разным числом шагов. Независимый coupling,
   без глобального optimal-transport решения и без VLA conditioning.
   [Статья](https://arxiv.org/abs/2210.02747) ·
   [RU](../models-at-home/notebooks/papers/ru/09-flow.ipynb) ·
   [EN](../models-at-home/notebooks/papers/en/09-flow.ipynb).
10. **ACT — Zhao et al., 2023.**
    State-only CVAE, reparameterization, masked L1+KL, z=0 inference и temporal ensemble
    предсказаний одного абсолютного времени. Отдельно показано, почему action=0 не
    является признаком отсутствия данных. Это не официальный visual ACT Transformer.
    [Статья](https://arxiv.org/abs/2304.13705) ·
    [RU](../models-at-home/notebooks/papers/ru/10-act.ipynb) ·
    [EN](../models-at-home/notebooks/papers/en/10-act.ipynb).

После базовых world-model/VLA уроков пройдите DDPM → Flow Matching → ACT. Затем можно
перейти в World Models/VLA Studio и сопоставить учебный механизм с настоящим pipeline,
проверив normalization, формат данных, loss и режим inference выбранной политики.

## Как проводить эксперименты

Сохраните копию ноутбука. Сформулируйте прогноз до изменения кода. Меняйте один фактор,
сбрасывайте ядро и запускайте ячейки по порядку: повторный запуск ячейки обучения на уже
обученной модели меняет исходные условия. Записывайте seed и compute budget. Отдельно
различайте корректность механизма, memorization train batch и качество на новых данных.

Встроенные assertions предназначены для проверки конкретных свойств. Для свободных
заданий с изменённой постановкой некоторые assertions закономерно перестанут выполняться:
объясните причину, вместо того чтобы просто удалить проверку.

## Проверка и обновление / Validation and maintenance

Content source: `scripts/build_paper_notebooks.py`. Regenerate with
`python scripts/build_paper_notebooks.py`; only generated paper files are replaced.
User copies should live outside the generated `papers/{ru,en}` filenames.

Run `python -m pytest backend/tests/test_paper_notebooks.py -q` to validate every RU/EN
notebook, compare executable-cell parity, check catalog metadata, and execute every
unique paper study in a fresh local kernel, including all assertions and PNG outputs.
No pretrained checkpoint or remote service is needed for these checks.
