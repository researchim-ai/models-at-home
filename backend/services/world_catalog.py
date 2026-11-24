"""Verified upstream foundation releases, distinct from our scratch-trained dynamics."""
COSMOS_EDGE_REVISION = '6afc1256321c3a8e623d89be3bc5117a1e90f96b'
DOMAINS = {'camera_pose': 9, 'av': 9, 'umi': 10, 'droid_lerobot': 10, 'robomind-franka': 10,
           'robomind-franka-dual': 20, 'robomind-ur': 10, 'agibotworld': 29, 'hand_pose': 57}
MODELS = [
    dict(id='diamond_csgo', name='DIAMOND · CS:GO', provider='diamond', repo_id='eloialonso/diamond', revision='5d4abca9af6ab1b3ab1c6fc228cacb7281130f7f',
         tasks=['local_play'], local_weights=True, training=False,
         description='Встроенная diffusion world model: кадры генерируются по вашим действиям. Готовые стартовые сцены, WASD и камера прямо в Studio. Архитектура и sampler находятся в репозитории.',
         requirements='≈1.53 GB весов; CUDA рекомендуется. Динамика 30×56 + optional neural upsampler 150×280. Скачивание без API-ключа, после него inference офлайн.',
         source='https://github.com/eloialonso/diamond/tree/csgo', license='Код MIT · Eloi Alonso; веса — официальный upstream release'),
    dict(id='diamond_atari', name='DIAMOND · Atari (Breakout / Pong)', provider='diamond', repo_id='eloialonso/diamond', revision='5d4abca9af6ab1b3ab1c6fc228cacb7281130f7f',
         tasks=['local_play'], local_weights=True, training=False,
         description='Компактная встроенная world model Atari. Pretrained Breakout и Pong, 64×64 кадры; управление дискретными действиями. Для Breakout включён стартовый пример; для Pong загрузите screenshot игры.',
         requirements='≈54 MB на игру. CPU или CUDA. ROM/emulator, облачный сервис и отдельный inference repo для world-model rollout не нужны.',
         source='https://github.com/eloialonso/diamond', license='Код MIT · Eloi Alonso; веса — официальный upstream release'),

    dict(id='cosmos3_edge', name='NVIDIA Cosmos 3 Edge · 4B', provider='cosmos3', repo_id='nvidia/Cosmos3-Edge', revision=COSMOS_EDGE_REVISION,
         tasks=['text2video', 'image2video', 'forward_dynamics'], local_weights=True, training=False,
         description='Открытая foundation model для видео и action-conditioned динамики. Embodiment-specific действия; наш произвольный VLA action vector автоматически не конвертируется.',
         requirements='CUDA, BF16; веса pipeline ≈8.2 GB плюс рабочая память и штатный Cosmos guardrail. CPU offload доступен.',
         source='https://huggingface.co/nvidia/Cosmos3-Edge', license='OpenMDW-1.1'),
    dict(id='cosmos3_nano', name='NVIDIA Cosmos 3 Nano · 16B', provider='cosmos3', repo_id='nvidia/Cosmos3-Nano', revision='45666784803b3a7ad577e0def405f9afcc3bddb2',
         tasks=['text2video', 'image2video', 'forward_dynamics'], local_weights=True, training=False,
         description='Более крупный вариант для физических сред. Выбирайте на GPU с достаточной памятью или с CPU offload.',
         requirements='Официальная целевая платформа: RTX Pro 6000 / H100 / B200; полный pipeline значительно больше Edge.',
         source='https://huggingface.co/nvidia/Cosmos3-Nano', license='OpenMDW-1.1'),
    dict(id='cosmos3_super', name='NVIDIA Cosmos 3 Super · 64B', provider='cosmos3', repo_id='nvidia/Cosmos3-Super', revision='73d5746645961a0ec9607123a003bd9c1f8a0282',
         tasks=['text2video', 'image2video', 'forward_dynamics'], local_weights=True, training=False,
         description='Крупная модель для генерации данных и teacher-задач. Локальный single-GPU адаптер Studio поддерживает CPU offload; distributed serving сюда не включён.',
         requirements='Официальная целевая платформа: H200 / B200 / GB200; ≈120 GB только веса, требуется большой CPU RAM для offload.',
         source='https://huggingface.co/nvidia/Cosmos3-Super', license='OpenMDW-1.1'),
]


def model_info(model_id):
    return next(row for row in MODELS if row['id'] == model_id).copy()
