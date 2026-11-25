# Ноутбуки внутри студии / Native learning notebooks

Откройте **Ноутбуки** в боковом меню. Выберите урок, прочитайте объяснение и нажмите
**Запустить всё** либо кнопку рядом с ячейкой. Shift + Enter выполняет текущую ячейку
Python. Код, Markdown, ошибки и графики отображаются в приложении; браузер с JupyterLab
для этого не нужен.

## Что входит

16 тем, каждая на русском и английском (32 `.ipynb`): основы обучения; данные и маски;
LLM pretraining/SFT; LoRA; GRPO; предпочтения/DPO и distillation; exploration/NGU;
VLM; VLA; world models; блоки архитектуры; looped transformers; MoE; Research Lab;
память и время; inference и checkpoints. Язык библиотеки следует языку интерфейса.

В каждом уроке есть объяснение, настоящий запускаемый CPU-пример на PyTorch,
самостоятельное задание и ссылки на первоисточники. Примеры синтетические и не
скачивают веса. Они объясняют механизмы: это не оценка качества больших моделей,
не полная реализация NGU, не обучение ACT/π0 и не замена production trainer.
Кнопка над уроком открывает соответствующий раздел студии.

## Разборы статей

Дополнительно доступны 10 подробных разборов (20 `.ipynb` на двух языках): Attention,
RoPE, LoRA, Universal Transformers, CLIP, DPO, DeepSeekMath/GRPO, DDPM, Flow Matching
и ACT. Вместе с вводными уроками это 52 учебных ноутбука. Выберите подборку
**Разборы статей · с экспериментами**, затем тему или найдите оригинальное название
статьи/автора. В карточке показаны авторы и год первого arXiv submission.

Каждый разбор связывает задачу и формулу с формами тензоров, содержит несколько
последовательных ячеек, asserts, график или диагностический эксперимент, задания,
пояснения к ответам и точную границу воспроизведения. Это собственные учебные
реализации механизмов; benchmark-результаты статей не воспроизводятся.
[Программа и порядок изучения](PAPER_STUDIES.md).

## Редактирование и файлы

- Изменяйте код и Markdown, добавляйте и удаляйте ячейки.
- **Сохранить** записывает код и текущий вывод; **Копия** сохраняет отдельный `.ipynb`.
- Можно импортировать `.ipynb` с компьютера и экспортировать текущий ноутбук.
  Импортированный файл сначала нужно сохранить в рабочую папку, затем запускать.
- Библиотека читает `notebooks/` в ML workspace рекурсивно, включая существующие
  проектные ноутбуки. Новые уроки находятся в `notebooks/lessons/{ru,en}/`.
- Сохранение проверяет ревизию: внешние изменения файла нельзя молча затереть.
- Запуск не сохраняет файл автоматически. Несохранённые изменения отмечены точкой.
  Сохраните их перед переходом в другой раздел.

## Ядро и окружение

У каждого открытого ноутбука собственное локальное Python-ядро с переменными между
ячейками. Используется **тот же Python, что у backend**, а не случайный глобальный
kernelspec. Рабочая директория — ML workspace; модули `backend` и `homellm` доступны
через Python path. Docker-backend исполняет код внутри своего контейнера.

Зависимости выполнения включены в `backend/requirements.txt`. Для ручной установки
в нужное окружение:

```bash
python -m pip install jupyter-client ipykernel matplotlib
```

Для встроенных учебных примеров также нужен PyTorch из ML-окружения. При отсутствии
ядра можно читать и редактировать ноутбуки; интерфейс показывает команду установки
и путь к Python. Ошибки импортов библиотек отображаются в выводе ячейки.

**Стоп** прерывает вычисление; **Сбросить ядро** освобождает процесс и удаляет переменные,
сохраняя код и показанный вывод. Следующий запуск создаёт новое ядро. При смене
ноутбука и закрытии раздела ядро завершается. При закрытии backend завершаются все
его ядра. Неиспользуемые ядра завершаются через час; одновременно допускается до
четырёх. Лимит ячейки — 10 минут, кода — 100 KB, вывода — 2 MB; файл — до 10 MB.
После аварии ядра или таймаута сбросьте его и повторите ячейки с начала.

Ячейки исполняют обычный Python с правами backend: имеют доступ к файлам и ресурсам.
Это не sandbox. Не запускайте неизвестный код. Renderer показывает текст, JSON,
PNG/JPEG; HTML/JavaScript, widgets, интерактивные HTML-графики и stdin не исполняются.
Markdown тоже не исполняет HTML. Для сложных виджетов можно отдельно использовать
Jupyter, но встроенным урокам это не требуется.

## Разработка / Development

The UI is `src/pages/Notebooks.tsx`; APIs live in `backend/routes/notebooks.py`;
thread-owned Jupyter clients live in `backend/services/notebook_runtime.py`.
There is no iframe, separate JupyterLab server, or frontend kernel token.
The implementation follows the [Jupyter client protocol](https://jupyter-client.readthedocs.io/en/stable/api/jupyter_client.html).

Rebuild shipped lessons with `python scripts/build_notebook_lessons.py`. It replaces
only the generated lessons, so keep personal experiments as copies. Packaging already
ships runtime notebooks and copies missing files to the user workspace without
replacing existing files.

Run `python -m pytest backend/tests/test_native_notebooks.py -q` to validate files,
exercise kernel persistence, errors, interruption, isolation, file conflicts and
execute every unique lesson with actual PyTorch backward/PNG plot capture. RU/EN
notebooks share executable code, while each file is validated separately.

Rebuild paper studies with `python scripts/build_paper_notebooks.py`; generated files
are in `notebooks/papers/{ru,en}/`. The generator does not touch personal notebook copies.
Run `python -m pytest backend/tests/test_paper_notebooks.py -q`: every unique study runs
in a fresh Python kernel, every code cell must finish successfully, all expected plots
must return PNG, and both language versions must have identical executable code.
