"""
Алгоритмы RL для обучения LLM.

Этот модуль предоставляет разные варианты алгоритмов:
- GRPO: Group Relative Policy Optimization (базовый)
- DrGRPO: GRPO Done Right (без std нормализации)
- DAPO: Decoupled Clip + Dynamic Sampling

В будущем можно добавить:
- PPO: Proximal Policy Optimization
- REINFORCE: Базовый policy gradient
- DPO: Direct Preference Optimization
"""

# Алгоритмы реализованы через конфигурацию GRPOConfig
# и параметризацию GRPOLoss

__all__ = []
