"""Model presets and explicit evidence; a preset is not a certification."""
from importlib.metadata import PackageNotFoundError, version

PROFILE = {'torch': '2.9.1+cu128', 'torchvision': '0.24.1', 'transformers': '5.17.0',
           'accelerate': '1.12.0', 'peft': '0.22.0', 'bitsandbytes': '0.50.2', 'Pillow': '12.0.0',
           'trl': '1.15.0', 'unsloth': '2026.11.9', 'unsloth_zoo': '2026.11.9', 'torchao': '0.14.1'}
RECIPES = [
    {'id': 'qwen3-vl-native-v1', 'model_type': 'qwen3_vl', 'name': 'Qwen3-VL', 'schema_version': 3,
     'models': ['Qwen/Qwen3-VL-2B-Instruct', 'Qwen/Qwen3-VL-4B-Instruct'],
     'evidence': 'tiny-local-gpu-smoke', 'verified_on': '2026-11-23',
     'scenarios': [dict(stage=stage, tuning_method=method, training_backend=backend)
                   for stage in ('vlm_sft', 'vlm_grpo') for method in ('full', 'lora', 'qlora')
                   for backend in ('transformers', 'unsloth')],
     'rl_engine': 'homellm_native',
     'policy_algorithms': {'lora': ['grpo', 'dr_grpo', 'dapo', 'sdpo'],
                           'full': ['grpo'], 'qlora': ['grpo']}},
    {'id': 'qwen3.5-native-v1', 'model_type': 'qwen3_5', 'name': 'Qwen3.5', 'schema_version': 3,
     'models': ['Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3.5-2B', 'Qwen/Qwen3.5-4B'],
     'evidence': 'tiny-local-gpu-smoke', 'verified_on': '2026-11-23',
     'scenarios': [dict(stage=stage, tuning_method='lora', training_backend=backend)
                   for stage in ('vlm_sft', 'vlm_grpo') for backend in ('transformers', 'unsloth')],
     'rl_engine': 'homellm_native',
     'policy_algorithms': {'transformers': ['grpo', 'sdpo'], 'unsloth': ['grpo', 'sdpo']},
     'pretrained_evidence': {'model': 'Qwen/Qwen3.5-0.8B',
         'revision': '2fc06364715b967f1860aea9cf38778875588b17',
         'scope': '2 optimizer steps, resume to 4, synthetic red-image VQA; not a quality evaluation',
         'scenarios': [dict(stage='vlm_sft', tuning_method='lora', training_backend=backend)
                       for backend in ('transformers', 'unsloth')]
                      + [dict(stage='vlm_grpo', tuning_method='lora', training_backend=backend,
                              grpo_algorithm=algorithm, rl_engine='homellm_native')
                         for backend in ('transformers', 'unsloth') for algorithm in ('grpo', 'sdpo')]}},
    {'id': 'gemma3-v1', 'model_type': 'gemma3', 'name': 'Gemma 3', 'schema_version': 3,
     'models': ['google/gemma-3-4b-it'], 'evidence': 'upstream-supported-not-locally-tested',
     'scenarios': [], 'note': 'Требуется доступ к gated-репозиторию Hugging Face.'},
]


def catalog():
    installed = {}
    for name in PROFILE:
        try:
            installed[name] = version(name)
        except PackageNotFoundError:
            installed[name] = None
    return {'recipes': RECIPES, 'tested_profile': PROFILE, 'installed': installed,
            'scope': 'GRPO подтверждён для собственного homellm RL, а не TRL. Каждый сценарий имеет отдельное свидетельство. Tiny smoke подтверждает механику, не качество pretrained-модели.'}


def evidence(model_type, config):
    recipe = next((item for item in RECIPES if item['model_type'] == model_type), None)
    if recipe is None:
        return 'warning', 'Это семейство не входит в актуальную GPU-матрицу Studio. Нужен отдельный короткий прогон.'
    scenario = {key: config.get(key, 'transformers' if key == 'training_backend' else None)
                for key in ('stage', 'tuning_method', 'training_backend')}
    if scenario not in recipe['scenarios']:
        return 'warning', f"{recipe['name']}: именно этот этап/режим/backend пока не проверен. Рецепт {recipe['id']}."
    if config['stage'] == 'vlm_grpo':
        algorithms = recipe.get('policy_algorithms', {})
        tested = algorithms.get(config['tuning_method'], algorithms.get(scenario['training_backend'], []))
        if config.get('grpo_algorithm', 'grpo') not in tested:
            return 'warning', f"{recipe['name']}: этот policy objective/backend не входил в проверенную матрицу."
    installed = catalog()['installed']
    required = ['torch', 'transformers', 'accelerate', 'Pillow', 'torchvision']
    if config['tuning_method'] in ('lora', 'qlora'):
        required.append('peft')
    if config['tuning_method'] == 'qlora':
        required.append('bitsandbytes')
    if scenario['training_backend'] == 'unsloth':
        required += ['unsloth', 'unsloth_zoo', 'trl', 'torchao']
    differences = [f'{name}: {installed[name] or "нет"} (проверено {PROFILE[name]})' for name in required
                   if installed[name] != PROFILE[name]]
    if differences:
        return 'warning', f"Рецепт {recipe['id']} проверен в другом окружении: " + '; '.join(differences)
    return 'pass', f"{recipe['name']}, {scenario}; рецепт {recipe['id']}. Проверен короткий GPU-цикл, не качество pretrained-модели."
