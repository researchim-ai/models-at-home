export const repo = "https://github.com/researchim-ai/models-at-home"
export const content = {
  "en": {
    "title": "Models at Home — a studio for understanding and building models",
    "description": "Learn how neural networks work, build your own architectures and train models on your data. Models at Home brings model development and training analysis into one local studio.",
    "skip": "Skip to content",
    "nav": [
      "The studio",
      "Research Lab",
      "How it works",
      "Get started"
    ],
    "source": "Source code",
    "menu": "Open navigation",
    "language": "Choose language",
    "eyebrow": "A STUDIO FOR MODEL DEVELOPMENT AND LEARNING",
    "hero": [
      "Understand models.",
      "Build your own."
    ],
    "intro": "Models at Home is a studio for learning how neural networks work and building models of your own. Edit an architecture, train it on your data, and inspect what happens during training.",
    "explore": "Explore the tools",
    "start": "Run it locally",
    "heroNote": "Open source · Runs locally · Training and analysis",
    "sketch": {
      "label": "EXAMPLE: A MODEL WITH A SHARED CORE",
      "input": "your data",
      "core": "a shared core",
      "repeat": "4 passes",
      "output": "your model",
      "note": "One core is applied four times.",
      "tag": "DATA → TRAINING",
      "shared": "shared weights"
    },
    "strip": [
      "Language",
      "Vision",
      "Action",
      "World models",
      "Architecture"
    ],
    "studio": {
      "eyebrow": "WHAT THE STUDIO INCLUDES",
      "title": "Tools for different types of models.",
      "intro": "The studio has sections for language, vision, actions, world models and custom architectures. Select a tab to see the tools and application screenshots.",
      "image": "Actual application screenshot",
      "expand": "Open full-size screenshot",
      "tabLabel": "Explore studio sections"
    },
    "workspaces": [
      {
        "name": "Language",
        "label": "LLM STUDIO",
        "title": "Train language models",
        "description": "Train a small language model from scratch, fine-tune a pretrained model on instructions, or use the supported reward-based training methods. Configure data and training, then follow the run in the interface.",
        "tags": [
          "Pretraining",
          "SFT",
          "Native GRPO / Dr.GRPO / DAPO / SDPO"
        ],
        "image": "training-studio-light",
        "alt": "Light-theme Training Studio with model, dataset, training configuration and readiness checks",
        "link": "README.md",
        "detail": "Full, LoRA and QLoRA availability depends on the model and backend."
      },
      {
        "name": "Vision",
        "label": "VLM STUDIO",
        "title": "Work with images and text",
        "description": "Train a model on images paired with questions, answers or instructions. Use a supported Hugging Face VLM, or train a compatible architecture from Research Lab with the studio’s image adapter.",
        "tags": [
          "Image + text",
          "SFT",
          "Native GRPO for supported HF models"
        ],
        "image": "vlm-studio",
        "alt": "VLM Studio configuration, vision-language training stages and preflight controls",
        "link": "docs/VLM_VALIDATION.md",
        "detail": "Transformers and optional Unsloth paths have model-specific dependencies. Blueprint adapters currently use full-weight supervised training."
      },
      {
        "name": "Action",
        "label": "VLA STUDIO",
        "title": "Train models to predict actions",
        "description": "Train on demonstrations that pair an observation with an action or a sequence of actions. Choose a native policy, a VLM with an action head, or a supported standalone robotics policy.",
        "tags": [
          "Behavior cloning",
          "Action chunks",
          "OpenVLA · SmolVLA · ACT · π0"
        ],
        "image": "vla-act-training",
        "alt": "A completed real ACT CPU training run with training and validation curves, action MAE and policy export",
        "link": "docs/VLA_POLICIES.md",
        "detail": "Standalone policies support continuous actions through their own supervised objectives. Online VLA GRPO is not implemented."
      },
      {
        "name": "Worlds",
        "label": "WORLD MODELS",
        "title": "Train models of environment dynamics",
        "description": "Train Latent GRU and Gaussian RSSM models on recorded episodes and inspect the trajectories they predict. A separate local inference path lets you try pretrained DIAMOND.",
        "tags": [
          "Episode datasets",
          "GRU / RSSM training",
          "Local DIAMOND inference"
        ],
        "image": "world-models",
        "alt": "World Models Studio with local architecture selection, resource estimates and rollout settings",
        "link": "docs/WORLD_MODELS_STUDIO.md",
        "detail": "DIAMOND is an inference path. The trainable world models are the studio’s GRU/RSSM implementations."
      },
      {
        "name": "Architecture",
        "label": "MODEL BUILDER",
        "title": "Build your own architecture",
        "description": "Assemble a model from blocks and edit their parameters and connections. Try attention, normalization, MoE and shared loops. A compatible architecture can then be sent to LLM, VLM or VLA training.",
        "tags": [
          "Looped Transformers",
          "GQA / RoPE / MoE",
          "LLM · VLM · VLA handoff"
        ],
        "image": "model-builder-looped",
        "alt": "Model Builder showing an editable shared recurrent core, nested Transformer blocks and architecture analysis",
        "link": "docs/MODEL_BUILDER.md",
        "detail": "The builder creates your own architectures. It does not transplant arbitrary pretrained weights into a new graph."
      },
      {
        "name": "Research Lab",
        "label": "RESEARCH LAB",
        "title": "Inspect the model and its training",
        "description": "See how tensors pass through the architecture, inspect activations and gradients, and run a short CPU check with real parameter updates. Save variants and compare their measured signals and estimated compute costs.",
        "tags": [
          "Block knowledge cards",
          "Real CPU probes",
          "Baselines + research notes"
        ],
        "image": "research-lab",
        "alt": "Research Lab architecture graph and loop knowledge card with original research papers",
        "link": "docs/RESEARCH_LAB.md",
        "detail": "Synthetic probes test mechanisms, not model quality. Analytical memory estimates are separate from measured signals."
      }
    ],
    "readGuide": "Read the guide",
    "manifesto": {
      "mark": "model structure\ntraining signals",
      "eyebrow": "LEARN BY WORKING WITH THE MODEL",
      "title": "Understand what your changes do.",
      "text": "Build a small architecture, change a block and compare the results. The studio helps you inspect the model’s structure, follow training and try the trained model.",
      "items": [
        [
          "Use your own hardware",
          "Training runs in local Python processes or a configured Docker environment. You choose the data, model and compute available to the run."
        ],
        [
          "Work through the training process",
          "Prepare a dataset, check the configuration, train, save checkpoints and load an exported model for inference."
        ],
        [
          "Read, inspect and extend",
          "Block cards link to research papers. Study Center provides learning materials, and JupyterLab lets you continue working in notebooks. Agent Studio includes a local llama-server assistant."
        ]
      ]
    },
    "research": {
      "eyebrow": "MODEL STRUCTURE AND TRAINING ANALYSIS",
      "title": [
        "Study the architecture.",
        "Check your changes."
      ],
      "text": "Learn what each supported block does, read its formula and follow the link to the research behind it. Short probes help you check tensor shapes, activations and gradients before a longer training run.",
      "guide": "Explore the Research Lab",
      "demoLabel": "Learn about the model’s building blocks",
      "demoNote": "Select a block to see an example of the explanations available in the desktop studio.",
      "source": "Original research",
      "ideas": "What to compare",
      "blocks": [
        {
          "name": "Shared loop",
          "type": "loop",
          "formula": "hₖ₊₁ = core(hₖ, x; θ)",
          "description": "Reuse one set of weights at several depths. More iterations increase computation without creating independent copies of the core.",
          "experiment": "Keep parameter count fixed. Compare depth, gradient flow and validation quality.",
          "paper": "Universal Transformers",
          "authors": "Dehghani et al. · 2018",
          "url": "https://arxiv.org/abs/1807.03819",
          "relationship": "Background for recurrent depth. The studio loop is its own finite-depth implementation, not a full Universal Transformer."
        },
        {
          "name": "RMS normalization",
          "type": "rmsnorm",
          "formula": "y = x / RMS(x) · γ",
          "description": "Rescale the hidden state by its root mean square and a learned scale. The card shows the formula and links to the paper that describes the method.",
          "experiment": "Compare activation scale and gradients against LayerNorm under the same training budget.",
          "paper": "Root Mean Square Layer Normalization",
          "authors": "Biao Zhang & Rico Sennrich · 2019",
          "url": "https://arxiv.org/abs/1910.07467",
          "relationship": "The original RMSNorm paper. Fused implementations are a separate implementation choice."
        },
        {
          "name": "Grouped-query attention",
          "type": "GQA",
          "formula": "Hq ≥ Hkv, Hq mod Hkv = 0",
          "description": "Let several query heads share key/value heads. Explore the balance between attention capacity and key/value projection cost.",
          "experiment": "Change KV heads while keeping hidden width and query heads fixed. Measure compute and validation quality.",
          "paper": "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints",
          "authors": "Ainslie et al. · 2023",
          "url": "https://arxiv.org/abs/2305.13245",
          "relationship": "The GQA paper. Adding GQA to a new model is separate from the paper’s pretrained-checkpoint conversion recipe."
        }
      ],
      "features": [
        [
          "Structure",
          "Inspect connections, tensor shapes and parameter counts. See estimated compute costs for training and inference."
        ],
        [
          "Training signals",
          "Run short synthetic training checks and inspect activations, gradients and causal behaviour."
        ],
        [
          "Compare variants",
          "Save an architecture, record what you want to test, then compare variants with the same measurement settings."
        ]
      ],
      "note": "Synthetic probes check how the model works. Evaluate quality separately on your own held-out data. Compute and memory estimates are labelled separately from measurements."
    },
    "policies": {
      "eyebrow": "STANDALONE VLA ARCHITECTURES",
      "title": "Policies for learning from demonstrations.",
      "text": "VLA Studio includes OpenVLA, SmolVLA, ACT and π0, alongside the studio’s native policies, VLM action heads and compatible Research Lab architectures.",
      "labels": [
        "Policy",
        "Training objective",
        "What to know"
      ],
      "items": [
        [
          "OpenVLA",
          "Action-token cross entropy",
          "Full or LoRA. Original policy: one image + instruction → one 7D action. No state input."
        ],
        [
          "SmolVLA",
          "Flow matching",
          "Pretrained vision-language backbone + action expert. Train on your own demonstrations."
        ],
        [
          "ACT",
          "L1 + CVAE KL",
          "An image-and-state policy. Train from scratch or load an ACT checkpoint; no language conditioning."
        ],
        [
          "π0",
          "Flow matching + ODE sampling",
          "The LeRobot/OpenPI port. The current adapter requires full action chunks."
        ]
      ],
      "note": "ACT has real CPU end-to-end checks; the other mechanisms have reduced-model checks. Full-size pretrained GPU runs still need separate validation. These standalone adapters currently use supervised training, without GRPO or Unsloth.",
      "link": "Policy setup & verified modes"
    },
    "workflow": {
      "eyebrow": "FROM DATA TO A TRAINED MODEL",
      "title": "How training works in the studio.",
      "items": [
        [
          "Prepare your data",
          "Use texts, image conversations, action demonstrations or episodes. Data can come from local files or a supported Hugging Face source."
        ],
        [
          "Choose or build a model",
          "Select a pretrained base or assemble your own architecture. Match the tokenizer, data format and action dimensions to the task."
        ],
        [
          "Configure and train",
          "Run readiness or preflight checks, then start training. Follow loss, supported reward metrics, validation results and logs. Save checkpoints along the way."
        ],
        [
          "Evaluate the result",
          "Try a text conversation, inspect predicted actions or view a rollout. Export the model and compare its results on held-out data."
        ]
      ]
    },
    "tools": {
      "title": "Other tools in the studio.",
      "items": [
        "Run history & logs",
        "Compatible checkpoint resume",
        "Model inference",
        "JupyterLab",
        "Study Center",
        "Local agent assistant"
      ]
    },
    "setup": {
      "eyebrow": "INSTALLATION",
      "title": "Run the studio on your computer.",
      "text": "Clone the repository, install the dependencies and open the desktop app. For a first run, use a small model or one of the CPU demos.",
      "requirements": [
        "Node.js 22.12+",
        "Python 3.12",
        "Git"
      ],
      "copy": "Copy commands",
      "copied": "Copied",
      "copyFailed": "Select the commands to copy them.",
      "osLabel": "Choose installation platform",
      "platforms": [
        "Linux / macOS",
        "Windows · PowerShell"
      ],
      "note": "For serious training, install PyTorch/CUDA for your device. Modern VLMs and standalone VLA policies use separate environments. Check the guides before downloading large weights.",
      "guide": "Full installation guide",
      "vla": "Standalone VLA setup",
      "backend": "In Settings, choose Native for local Python. Docker is available for the workflows supported by your container environment."
    },
    "faq": {
      "eyebrow": "QUESTIONS ABOUT THE STUDIO",
      "title": "Before the first run.",
      "items": [
        [
          "Who is the studio for?",
          "For people who want to understand how models work, build their own architectures and run training experiments. The interface brings common tools together, while block explanations, research links and training signals help you study the model as you work."
        ],
        [
          "Do I need a powerful GPU?",
          "Small Native VLA, world-model demos and bounded Research Lab probes can run on CPU. Large language and vision models usually need a suitable GPU. Memory estimates help you plan; they do not guarantee a model will fit."
        ],
        [
          "Does the studio send my data to a cloud model?",
          "Core training runs locally. A network connection is needed when downloading models or datasets from Hugging Face. Your files, cached weights and run outputs stay in your local workspace."
        ],
        [
          "Can I train an architecture I build myself?",
          "Yes. A compatible causal blueprint can go to LLM training, or use the studio’s trainable image/action adapters in VLM and VLA Studio. Those adapters currently train from scratch with full-weight supervised objectives."
        ],
        [
          "Does every model support every training method?",
          "No. Full, LoRA, QLoRA, Unsloth and reward-based methods depend on the specific architecture and runtime. The interface and guides describe the supported combinations. Standalone VLA policies currently use supervised learning."
        ],
        [
          "Is this a robot-control or deployment system?",
          "VLA Studio trains on demonstrations and previews predicted action chunks. It does not connect to a robot, game or desktop controller. Actual deployment needs a separate environment adapter and evaluation."
        ]
      ]
    },
    "closing": [
      "Build a small model.",
      "Follow its training."
    ],
    "closingMark": "architecture\ntraining · analysis",
    "closingText": "Start with a built-in recipe, train on a small dataset, then change the architecture and compare the results.",
    "footer": "A studio for understanding, building and training neural networks.",
    "license": "Runtime: Apache-2.0",
    "back": "Back to top",
    "close": "Close screenshot",
    "prev": "Previous screenshot",
    "next": "Next screenshot",
    "gallery": "Application screenshot gallery",
    "footerDocs": "Documentation",
    "footerLab": "Research guide"
  },
  "ru": {
    "title": "Models at Home — студия для изучения и создания моделей",
    "description": "Разбирайтесь в устройстве нейросетей, создавайте свои архитектуры и обучайте модели на своих данных. Models at Home объединяет разработку моделей и анализ обучения в одном локальном приложении.",
    "skip": "Перейти к содержимому",
    "nav": [
      "Студия",
      "Research Lab",
      "Как это работает",
      "Начать"
    ],
    "source": "Исходный код",
    "menu": "Открыть навигацию",
    "language": "Выбрать язык",
    "eyebrow": "СТУДИЯ ДЛЯ ИЗУЧЕНИЯ И ОБУЧЕНИЯ МОДЕЛЕЙ",
    "hero": [
      "Изучайте модели.",
      "Создавайте свои."
    ],
    "intro": "Models at Home — студия, в которой можно разобраться, как устроены нейросети, собрать свою архитектуру и обучить модель на своих данных. Меняйте блоки, наблюдайте за обучением и проверяйте результат в одном интерфейсе.",
    "explore": "Посмотреть инструменты",
    "start": "Запустить локально",
    "heroNote": "Открытый код · Локальный запуск · Обучение и анализ",
    "sketch": {
      "label": "ПРИМЕР МОДЕЛИ С ОБЩИМ ЯДРОМ",
      "input": "ваши данные",
      "core": "общее ядро",
      "repeat": "4 прохода",
      "output": "ваша модель",
      "note": "Одно ядро применяется четыре раза.",
      "tag": "ДАННЫЕ → ОБУЧЕНИЕ",
      "shared": "общие веса"
    },
    "strip": [
      "Язык",
      "Зрение",
      "Действия",
      "World models",
      "Архитектура"
    ],
    "studio": {
      "eyebrow": "ВОЗМОЖНОСТИ СТУДИИ",
      "title": "Инструменты для разных типов моделей.",
      "intro": "В студии есть разделы для языковых, визуальных и action-моделей, world models и собственных архитектур. Переключайте вкладки, чтобы посмотреть инструменты и скриншоты приложения.",
      "image": "Реальный скриншот приложения",
      "expand": "Открыть скриншот целиком",
      "tabLabel": "Разделы студии"
    },
    "workspaces": [
      {
        "name": "Язык",
        "label": "LLM STUDIO",
        "title": "Обучение языковых моделей",
        "description": "Обучайте небольшую языковую модель с нуля, дообучайте готовую модель на инструкциях или используйте поддерживаемые методы обучения с наградами. В интерфейсе можно выбрать данные, задать параметры и следить за запуском.",
        "tags": [
          "Pretraining",
          "SFT",
          "Свои GRPO / Dr.GRPO / DAPO / SDPO"
        ],
        "image": "training-studio-light",
        "alt": "Светлая Training Studio: выбор модели, датасета, настройка обучения и проверка готовности",
        "link": "README.md",
        "detail": "Доступность Full, LoRA и QLoRA зависит от модели и backend."
      },
      {
        "name": "Зрение",
        "label": "VLM STUDIO",
        "title": "Модели для изображений и текста",
        "description": "Обучайте модель на изображениях с вопросами, ответами или инструкциями. Можно выбрать поддерживаемую VLM с Hugging Face или обучить совместимую архитектуру из Research Lab с адаптером изображений.",
        "tags": [
          "Изображение + текст",
          "SFT",
          "Свой GRPO для совместимых HF-моделей"
        ],
        "image": "vlm-studio",
        "alt": "VLM Studio с настройками визуально-языковой модели, стадиями обучения и проверкой эксперимента",
        "link": "docs/VLM_VALIDATION.md",
        "detail": "Для Transformers и Unsloth нужны зависимости, совместимые с выбранной моделью. Архитектуры из конструктора пока обучаются с нуля, по подготовленным примерам и с обновлением всех весов."
      },
      {
        "name": "Действия",
        "label": "VLA STUDIO",
        "title": "Обучение моделей действий",
        "description": "Обучайте модель по демонстрациям: наблюдение связывается с действием или последовательностью действий. Доступны собственные политики студии, VLM с выходом действий и отдельные архитектуры для робототехники.",
        "tags": [
          "Behavior cloning",
          "Последовательности действий",
          "OpenVLA · SmolVLA · ACT · π0"
        ],
        "image": "vla-act-training",
        "alt": "Завершённое реальное обучение ACT на CPU: кривые обучения и validation, action MAE и экспорт политики",
        "link": "docs/VLA_POLICIES.md",
        "detail": "Отдельные VLA-политики работают с непрерывными действиями и обучаются по демонстрациям. Обучение VLA через взаимодействие со средой и GRPO пока не реализовано."
      },
      {
        "name": "Миры",
        "label": "WORLD MODELS",
        "title": "Модели динамики среды",
        "description": "Обучайте Latent GRU и Gaussian RSSM на записанных эпизодах и смотрите, какие траектории они предсказывают. В отдельном режиме можно локально попробовать готовый DIAMOND.",
        "tags": [
          "Данные по эпизодам",
          "Обучение GRU / RSSM",
          "Локальный инференс DIAMOND"
        ],
        "image": "world-models",
        "alt": "World Models Studio: выбор локальной архитектуры, оценка ресурсов и настройки rollout",
        "link": "docs/WORLD_MODELS_STUDIO.md",
        "detail": "DIAMOND можно запустить с готовыми весами. Для обучения собственных world models в студии предусмотрены GRU и RSSM."
      },
      {
        "name": "Архитектура",
        "label": "MODEL BUILDER",
        "title": "Конструктор собственных архитектур",
        "description": "Собирайте модель из блоков, меняйте их параметры и связи. Пробуйте attention, нормализацию, MoE и циклы с общими весами. Совместимую архитектуру можно передать в раздел обучения LLM, VLM или VLA.",
        "tags": [
          "Looped Transformers",
          "GQA / RoPE / MoE",
          "Передача в LLM · VLM · VLA"
        ],
        "image": "model-builder-looped",
        "alt": "Model Builder с редактируемым общим recurrent core, вложенными Transformer-блоками и анализом архитектуры",
        "link": "docs/MODEL_BUILDER.md",
        "detail": "Конструктор создаёт собственные архитектуры. Он не переносит произвольные готовые веса в новый граф автоматически."
      },
      {
        "name": "Research Lab",
        "label": "RESEARCH LAB",
        "title": "Анализ модели и её обучения",
        "description": "Смотрите, как тензоры проходят через архитектуру, изучайте активации и градиенты. Запускайте короткую CPU-проверку с обновлением весов, сохраняйте варианты и сравнивайте сигналы и вычислительные затраты.",
        "tags": [
          "Карточки блоков со статьями",
          "Настоящие CPU-пробы",
          "Baseline и исследовательские заметки"
        ],
        "image": "research-lab",
        "alt": "Research Lab: граф архитектуры и карточка loop со ссылками на первичные исследования",
        "link": "docs/RESEARCH_LAB.md",
        "detail": "Прогон на синтетических данных помогает проверить механизмы модели. Для оценки качества нужны ваши данные. Расчётные оценки памяти отделены от измеренных значений."
      }
    ],
    "readGuide": "Открыть руководство",
    "manifesto": {
      "mark": "устройство модели\nсигналы обучения",
      "eyebrow": "ИЗУЧЕНИЕ НА ПРАКТИКЕ",
      "title": "Разбирайтесь, на что влияют ваши изменения.",
      "text": "Соберите небольшую архитектуру, замените блок и сравните результаты. Студия помогает изучить устройство модели, проследить за обучением и проверить обученную модель.",
      "items": [
        [
          "Работа на своём оборудовании",
          "Обучение выполняется в локальных Python-процессах или настроенном Docker-окружении. Вы выбираете данные, модель и доступные вычислительные ресурсы."
        ],
        [
          "Все этапы обучения",
          "Подготовьте датасет, проверьте настройки, запустите обучение, сохраните checkpoints и загрузите экспортированную модель для проверки."
        ],
        [
          "Материалы и инструменты для изучения",
          "В карточках блоков есть ссылки на исследования, в Study Center — учебные материалы. Для работы в ноутбуках предусмотрен JupyterLab, а в Agent Studio — локальный помощник на llama-server."
        ]
      ]
    },
    "research": {
      "eyebrow": "УСТРОЙСТВО МОДЕЛИ И АНАЛИЗ ОБУЧЕНИЯ",
      "title": [
        "Изучайте архитектуру.",
        "Проверяйте изменения."
      ],
      "text": "Узнайте, какую задачу выполняет каждый поддерживаемый блок, посмотрите его формулу и прочитайте исходную статью. Короткие прогоны помогают проверить формы тензоров, активации и градиенты перед длительным обучением.",
      "guide": "Исследовать Research Lab",
      "demoLabel": "Из чего состоит модель",
      "demoNote": "Выберите блок: здесь показан пример объяснений, доступных в desktop-студии.",
      "source": "Исходная работа",
      "ideas": "Что можно сравнить",
      "blocks": [
        {
          "name": "Общий loop",
          "type": "loop",
          "formula": "hₖ₊₁ = core(hₖ, x; θ)",
          "description": "Применяйте одни и те же веса на нескольких глубинах. Больше итераций — больше вычислений, без независимых копий ядра.",
          "experiment": "Зафиксируйте число параметров. Сравните глубину, прохождение градиента и качество на validation.",
          "paper": "Universal Transformers",
          "authors": "Dehghani и др. · 2018",
          "url": "https://arxiv.org/abs/1807.03819",
          "relationship": "Связанная работа о recurrence по глубине. Loop в студии — собственная реализация с конечной глубиной, а не полный Universal Transformer."
        },
        {
          "name": "RMS-нормализация",
          "type": "rmsnorm",
          "formula": "y = x / RMS(x) · γ",
          "description": "Нормализует скрытое состояние через его среднеквадратичное значение и обучаемый коэффициент. В карточке можно посмотреть формулу и перейти к статье с описанием метода.",
          "experiment": "Сравните масштаб активаций и градиенты с LayerNorm при одинаковом бюджете обучения.",
          "paper": "Root Mean Square Layer Normalization",
          "authors": "Biao Zhang и Rico Sennrich · 2019",
          "url": "https://arxiv.org/abs/1910.07467",
          "relationship": "Оригинальная работа о RMSNorm. Fused-ядра — отдельный выбор реализации."
        },
        {
          "name": "Grouped-query attention",
          "type": "GQA",
          "formula": "Hq ≥ Hkv, Hq mod Hkv = 0",
          "description": "Несколько query-heads разделяют key/value-heads. Исследуйте баланс между возможностями attention и стоимостью KV-проекций.",
          "experiment": "Меняйте число KV-heads при фиксированной ширине и числе query-heads. Измеряйте вычисления и validation.",
          "paper": "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints",
          "authors": "Ainslie и др. · 2023",
          "url": "https://arxiv.org/abs/2305.13245",
          "relationship": "Работа о GQA. Добавление GQA в новую модель отличается от описанного авторами переноса готового checkpoint."
        }
      ],
      "features": [
        [
          "Структура",
          "Связи между блоками, формы тензоров и число параметров. Оценки вычислений для обучения и инференса."
        ],
        [
          "Сигналы обучения",
          "Короткие синтетические прогоны, активации, градиенты и проверка causal-поведения."
        ],
        [
          "Сравнение вариантов",
          "Сохраните архитектуру, запишите, что хотите проверить, и сравните варианты с одинаковыми настройками измерений."
        ]
      ],
      "note": "Синтетический прогон помогает проверить работу модели. Качество оценивайте отдельно на своих отложенных данных. Оценки памяти и вычислений помечены отдельно от результатов замеров."
    },
    "policies": {
      "eyebrow": "ОТДЕЛЬНЫЕ АРХИТЕКТУРЫ VLA",
      "title": "Политики для обучения по демонстрациям.",
      "text": "В VLA Studio доступны OpenVLA, SmolVLA, ACT и π0, а также собственные политики студии, VLM с выходом действий и совместимые архитектуры из Research Lab.",
      "labels": [
        "Политика",
        "Как обучается",
        "Что учитывать"
      ],
      "items": [
        [
          "OpenVLA",
          "Cross entropy по action tokens",
          "Full или LoRA. Оригинальная политика: изображение + инструкция → одно 7D-действие. Без state."
        ],
        [
          "SmolVLA",
          "Flow matching",
          "Готовый визуально-языковой backbone и action expert. Дообучается на ваших демонстрациях."
        ],
        [
          "ACT",
          "L1 + CVAE KL",
          "Изображение и state. Обучение с нуля или ACT checkpoint; текст инструкции не используется."
        ],
        [
          "π0",
          "Flow matching + ODE sampling",
          "Порт LeRobot/OpenPI. Текущему адаптеру нужны полные action chunks."
        ]
      ],
      "note": "ACT проверен полным CPU-циклом; остальные механизмы — на уменьшенных моделях. Полноразмерные pretrained GPU-прогоны требуют отдельной проверки. Эти адаптеры пока используют supervised обучение, без GRPO и Unsloth.",
      "link": "Установка политик и проверенные режимы"
    },
    "workflow": {
      "eyebrow": "ОТ ДАННЫХ ДО ОБУЧЕННОЙ МОДЕЛИ",
      "title": "Как проходит обучение в студии.",
      "items": [
        [
          "Подготовьте данные",
          "Тексты, диалоги с изображениями, демонстрации действий или эпизоды. Используйте локальные файлы или поддерживаемый источник Hugging Face."
        ],
        [
          "Выберите или соберите модель",
          "Возьмите готовую основу или создайте свою архитектуру. Согласуйте tokenizer, формат данных и размерности действий с задачей."
        ],
        [
          "Настройте и запустите обучение",
          "Пройдите проверку готовности и запустите обучение. Следите за loss, доступными метриками награды, validation и логами. Сохраняйте промежуточные checkpoints."
        ],
        [
          "Проверьте результат",
          "Попробуйте модель в чате, посмотрите предсказанные действия или rollout. Экспортируйте модель и сравните результаты на отложенных данных."
        ]
      ]
    },
    "tools": {
      "title": "Дополнительные инструменты студии.",
      "items": [
        "История запусков и логи",
        "Продолжение из совместимых checkpoints",
        "Проверка обученных моделей",
        "JupyterLab",
        "Study Center",
        "Локальный помощник"
      ]
    },
    "setup": {
      "eyebrow": "УСТАНОВКА И ЗАПУСК",
      "title": "Запустите студию на своём компьютере.",
      "text": "Склонируйте репозиторий, установите зависимости и откройте приложение. Для первого запуска выберите небольшую модель или одно из CPU-демо.",
      "requirements": [
        "Node.js 22.12+",
        "Python 3.12",
        "Git"
      ],
      "copy": "Скопировать команды",
      "copied": "Скопировано",
      "copyFailed": "Выделите команды и скопируйте их.",
      "osLabel": "Платформа установки",
      "platforms": [
        "Linux / macOS",
        "Windows · PowerShell"
      ],
      "note": "Для серьёзного обучения установите PyTorch/CUDA под своё устройство. Современным VLM и отдельным VLA-политикам нужны разные окружения. Посмотрите инструкции перед скачиванием больших весов.",
      "guide": "Полная инструкция установки",
      "vla": "Установка VLA-политик",
      "backend": "В Settings выберите Native для локального Python. Docker доступен для сценариев, поддерживаемых вашим контейнерным окружением."
    },
    "faq": {
      "eyebrow": "ВОПРОСЫ О СТУДИИ",
      "title": "Перед первым запуском.",
      "items": [
        [
          "Для кого эта студия?",
          "Для тех, кто хочет разобраться в устройстве моделей, создавать свои архитектуры и проводить эксперименты с обучением. Интерфейс объединяет рабочие инструменты, а объяснения блоков, ссылки на статьи и сигналы обучения помогают изучать модель в процессе работы."
        ],
        [
          "Мне нужна мощная видеокарта?",
          "Небольшие Native VLA и world-model демо, а также ограниченные пробы Research Lab работают на CPU. Большим языковым и визуальным моделям обычно нужен подходящий GPU. Оценки памяти помогают планировать, но не гарантируют, что модель поместится."
        ],
        [
          "Студия отправляет данные в облачную модель?",
          "Основное обучение выполняется локально. Сеть нужна для скачивания моделей и датасетов с Hugging Face. Ваши файлы, кеши весов и результаты запусков остаются в локальном workspace."
        ],
        [
          "Можно обучить архитектуру, которую я сам соберу?",
          "Да. Совместимый causal blueprint передаётся в LLM-обучение или получает обучаемые image/action адаптеры в VLM и VLA Studio. Эти адаптеры пока обучаются с нуля через full-weight supervised objectives."
        ],
        [
          "Любой метод обучения работает с любой моделью?",
          "Нет. Full, LoRA, QLoRA, Unsloth и методы с наградами зависят от архитектуры и runtime. Поддерживаемые сочетания описаны в интерфейсе и руководствах. Отдельные VLA-политики пока используют supervised обучение."
        ],
        [
          "Можно сразу подключить робота или управлять компьютером?",
          "VLA Studio обучает политику по демонстрациям и показывает предсказанные action chunks. Она не подключает робота, игру или desktop-контроллер. Для применения нужен отдельный environment adapter и оценка качества."
        ]
      ]
    },
    "closing": [
      "Соберите свою модель.",
      "Посмотрите, как она учится."
    ],
    "closingMark": "архитектура\nобучение · анализ",
    "closingText": "Начните с готового рецепта и небольшого датасета. Затем измените архитектуру и сравните результаты обучения.",
    "footer": "Студия для изучения, создания и обучения нейросетей.",
    "license": "Runtime: Apache-2.0",
    "back": "Наверх",
    "close": "Закрыть скриншот",
    "prev": "Предыдущий скриншот",
    "next": "Следующий скриншот",
    "gallery": "Галерея скриншотов приложения",
    "footerDocs": "Документация",
    "footerLab": "Руководство Research Lab"
  }
}

export const commands = {
  "unix": "git clone https://github.com/researchim-ai/models-at-home.git\ncd models-at-home\nnpm ci\n\npython3 -m venv .venv\nsource .venv/bin/activate\npython -m pip install --upgrade pip\npython -m pip install -r backend/requirements.txt\npython -m pip install -e ./models-at-home \"transformers==4.57.3\" datasets peft Pillow\n\nMAH_PYTHON=\"$PWD/.venv/bin/python\" npm run dev",
  "windows": "git clone https://github.com/researchim-ai/models-at-home.git\ncd models-at-home\nnpm ci\n\npy -3.12 -m venv .venv\n.\\.venv\\Scripts\\Activate.ps1\npython -m pip install --upgrade pip\npython -m pip install -r backend/requirements.txt\npython -m pip install -e ./models-at-home \"transformers==4.57.3\" datasets peft Pillow\n\n$env:MAH_PYTHON = \"$PWD\\.venv\\Scripts\\python.exe\"\nnpm run dev"
}
