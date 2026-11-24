# DIAMOND upstream attribution

MIT licensed code, Copyright (c) 2024 Eloi Alonso. See LICENSE.
Source: https://github.com/eloialonso/diamond
Atari revision: 5bcd1599755b4f2fae8e5e079e02f0728e174965
CS:GO revision: 851cefb497733d27f1b85c804104638765860fca

Included neural blocks, denoisers, inner models and diffusion samplers. Changes: package-relative imports; training-only Batch/LossAndLogs type dependencies decoupled from upstream trainer. Inference math and checkpoint key names preserved. Studio session/control/cache interfaces are our own. Upstream game/pygame, ROMs, Hydra launcher and RL trainer are not required for local inference.
