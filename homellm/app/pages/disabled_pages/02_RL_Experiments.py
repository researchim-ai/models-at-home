"""RL Experiments — DQN / PPO / AlphaZero (TicTacToe)"""

from __future__ import annotations

import base64
import json
import math
import os
import queue
import random
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ------------------------------------------------------------------------------
# Optional deps
# ------------------------------------------------------------------------------
try:
    # Headless rendering for Docker/Linux
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import gymnasium as gym  # type: ignore

    HAS_GYM = True
except Exception:
    HAS_GYM = False

try:
    import imageio.v2 as imageio  # type: ignore
    import imageio_ffmpeg  # noqa: F401

    HAS_VIDEO = True
except Exception:
    HAS_VIDEO = False

fragment = getattr(st, "fragment", getattr(st, "experimental_fragment", None))

# ------------------------------------------------------------------------------
# Streamlit setup
# ------------------------------------------------------------------------------
st.set_page_config(page_title="RL Experiments", page_icon="🤖", layout="wide")
st.title("🤖 RL Experiments")
st.caption("DQN / PPO (Gym) + AlphaZero (TicTacToe self-play + MCTS) — с проверками корректности")

if not HAS_GYM:
    st.error("Библиотека `gymnasium` не установлена. Установите: `pip install 'gymnasium[classic_control]'`")
    st.stop()

# ------------------------------------------------------------------------------
# Runs persistence (like LLM page, but under .runs/rl)
# ------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RL_RUNS_DIR = PROJECT_ROOT / ".runs" / "rl"
RL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
RL_ACTIVE_RUN_FILE = RL_RUNS_DIR / "active_run.json"


def rl_make_run_id() -> str:
    return datetime.now().strftime("rl_%Y%m%d_%H%M%S")


def rl_save_active_run(run_id: str, config: Dict[str, Any]) -> None:
    data = {"run_id": run_id, "started_at": datetime.now().isoformat(), "config": config}
    with open(RL_ACTIVE_RUN_FILE, "w") as f:
        json.dump(data, f, indent=2)


def rl_load_active_run() -> Optional[Dict[str, Any]]:
    if not RL_ACTIVE_RUN_FILE.exists():
        return None
    try:
        with open(RL_ACTIVE_RUN_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def rl_clear_active_run() -> None:
    if RL_ACTIVE_RUN_FILE.exists():
        RL_ACTIVE_RUN_FILE.unlink()


def jsonl_append(fp, obj: Dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def jsonl_tail(path: Path, max_lines: int = 5000) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    # Read tail efficiently
    dq: deque[str] = deque(maxlen=max_lines)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dq.append(line)
    out: List[Dict[str, Any]] = []
    for ln in dq:
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


# ------------------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------------------
if "rl_state" not in st.session_state:
    st.session_state.rl_state = {
        "running": False,
        "stop_event": threading.Event(),
        "metrics_queue": queue.Queue(),
        "episode_rewards": [],
        "logs": [],
        "latest_video_b64": None,
        "latest_video_episode": 0,
        "latest_video_version": 0,
        "video_lock": threading.Lock(),
        # last known algo stats (losses, eps, etc)
        "stats": {},
    }

if "rl_view_mode" not in st.session_state:
    st.session_state.rl_view_mode = False
if "rl_view_run_id" not in st.session_state:
    st.session_state.rl_view_run_id = None
if "rl_current_run_id" not in st.session_state:
    st.session_state.rl_current_run_id = None

_active = rl_load_active_run()
if _active and _active.get("run_id") and st.session_state.rl_current_run_id is None:
    # Restore last active run id for viewing convenience (thread liveness isn't persisted)
    st.session_state.rl_current_run_id = _active["run_id"]


def _log(state: Dict[str, Any], msg: str) -> None:
    state["logs"].append(msg)
    # keep logs bounded
    if len(state["logs"]) > 400:
        state["logs"] = state["logs"][-250:]


def resolve_device(device_choice: str) -> torch.device:
    if device_choice == "cpu":
        return torch.device("cpu")
    if device_choice == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------------------
# Networks
# ------------------------------------------------------------------------------
def activation_from_name(name: str) -> type[nn.Module]:
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    if name == "gelu":
        return nn.GELU
    if name == "silu":
        return nn.SiLU
    return nn.ReLU


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: List[int],
    activation: type[nn.Module],
    *,
    output_activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_size = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(in_size, h))
        layers.append(activation())
        in_size = h
    layers.append(nn.Linear(in_size, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class DQNNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int], activation: type[nn.Module]):
        super().__init__()
        self.net = build_mlp(obs_dim, act_dim, hidden_sizes, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int], activation: type[nn.Module]):
        super().__init__()
        self.actor = build_mlp(obs_dim, act_dim, hidden_sizes, activation)
        self.critic = build_mlp(obs_dim, 1, hidden_sizes, activation)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value


class AlphaZeroNet(nn.Module):
    """Policy logits + value in [-1, 1] for TicTacToe canonical board."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int], activation: type[nn.Module]):
        super().__init__()
        if len(hidden_sizes) < 1:
            hidden_sizes = [64]
        trunk_out = hidden_sizes[-1]
        trunk_hidden = hidden_sizes[:-1]
        self.trunk = build_mlp(obs_dim, trunk_out, trunk_hidden, activation)
        self.policy_head = nn.Linear(trunk_out, act_dim)  # logits
        self.value_head = nn.Linear(trunk_out, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h).squeeze(-1))
        return logits, value


# ------------------------------------------------------------------------------
# DQN
# ------------------------------------------------------------------------------
@dataclass
class ReplayItem:
    s: np.ndarray
    a: int
    r: float
    ns: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: deque[ReplayItem] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buf)

    def push(self, item: ReplayItem) -> None:
        self.buf.append(item)

    def sample(self, batch_size: int) -> List[ReplayItem]:
        return random.sample(list(self.buf), batch_size)


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: List[int],
        activation: type[nn.Module],
        lr: float,
        gamma: float,
        device: torch.device,
        *,
        grad_clip: float = 1.0,
    ):
        self.q = DQNNet(obs_dim, act_dim, hidden_sizes, activation).to(device)
        self.tq = DQNNet(obs_dim, act_dim, hidden_sizes, activation).to(device)
        self.tq.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.device = device
        self.grad_clip = grad_clip
        self.act_dim = act_dim

    @torch.no_grad()
    def act(self, obs: np.ndarray, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(self.act_dim)
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q(x)[0]
        return int(torch.argmax(q).item())

    def update(self, batch: List[ReplayItem]) -> float:
        s = torch.tensor(np.stack([b.s for b in batch]), dtype=torch.float32, device=self.device)
        a = torch.tensor([b.a for b in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor([b.r for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.tensor(np.stack([b.ns for b in batch]), dtype=torch.float32, device=self.device)
        done = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q(s).gather(1, a)
        with torch.no_grad():
            next_q = self.tq(ns).max(dim=1, keepdim=True).values
            target = r + (1.0 - done) * self.gamma * next_q

        loss = F.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.opt.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.tq.load_state_dict(self.q.state_dict())


# ------------------------------------------------------------------------------
# PPO (minimal but correct: clipped objective + GAE)
# ------------------------------------------------------------------------------
@dataclass
class PPOBatch:
    obs: torch.Tensor
    act: torch.Tensor
    logp: torch.Tensor
    adv: torch.Tensor
    ret: torch.Tensor


def gae_advantages(rewards: List[float], values: List[float], dones: List[bool], gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else 0.0
        next_nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    ret = adv + np.array(values, dtype=np.float32)
    return adv, ret


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: List[int],
        activation: type[nn.Module],
        lr: float,
        gamma: float,
        device: torch.device,
        *,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        grad_clip: float = 1.0,
        gae_lambda: float = 0.95,
    ):
        self.net = ActorCritic(obs_dim, act_dim, hidden_sizes, activation).to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.device = device
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.grad_clip = grad_clip
        self.gae_lambda = gae_lambda

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> Tuple[int, float, float]:
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(x)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return int(a.item()), float(logp.item()), float(value.item())

    def update(self, batch: PPOBatch, *, epochs: int, minibatch_size: int) -> Dict[str, float]:
        obs, act, old_logp, adv, ret = batch.obs, batch.act, batch.logp, batch.adv, batch.ret
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.shape[0]
        idx = np.arange(n)
        last = {}
        for _ in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, n, minibatch_size):
                mb = idx[start : start + minibatch_size]
                logits, value = self.net(obs[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act[mb])
                ratio = torch.exp(logp - old_logp[mb])

                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value, ret[mb])
                entropy = dist.entropy().mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                self.opt.step()

                last = {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                    "loss": float(loss.item()),
                }
        return last


# ------------------------------------------------------------------------------
# TicTacToe Game (correct for AlphaZero) + render
# ------------------------------------------------------------------------------
def ttt_winner(board: np.ndarray) -> int:
    """Returns 1 if X wins, -1 if O wins, 0 if none, 2 if draw."""
    lines = []
    lines.extend([board[i, :] for i in range(3)])
    lines.extend([board[:, i] for i in range(3)])
    lines.append(np.array([board[0, 0], board[1, 1], board[2, 2]]))
    lines.append(np.array([board[0, 2], board[1, 1], board[2, 0]]))
    for ln in lines:
        s = int(ln.sum())
        if s == 3:
            return 1
        if s == -3:
            return -1
    if np.all(board != 0):
        return 2
    return 0


def ttt_valid_moves(board: np.ndarray) -> np.ndarray:
    return (board.flatten() == 0).astype(np.float32)  # (9,)


def ttt_next(board: np.ndarray, player: int, action: int) -> Tuple[np.ndarray, int]:
    nb = board.copy()
    r, c = divmod(action, 3)
    if nb[r, c] != 0:
        raise ValueError("Invalid move")
    nb[r, c] = player
    return nb, -player


def ttt_canonical(board: np.ndarray, player: int) -> np.ndarray:
    return board * player


def ttt_render(board: np.ndarray) -> np.ndarray:
    # 288x288 (divisible by 16 for codecs)
    img = np.ones((288, 288, 3), dtype=np.uint8) * 255
    img[96:98, :, :] = 0
    img[192:194, :, :] = 0
    img[:, 96:98, :] = 0
    img[:, 192:194, :] = 0
    for r in range(3):
        for c in range(3):
            v = board[r, c]
            cx, cy = c * 96 + 48, r * 96 + 48
            if v == 1:
                for i in range(-30, 31):
                    x, y = cx + i, cy + i
                    if 0 <= x < 288 and 0 <= y < 288:
                        img[y, x] = [0, 0, 255]
                    x, y = cx + i, cy - i
                    if 0 <= x < 288 and 0 <= y < 288:
                        img[y, x] = [0, 0, 255]
            elif v == -1:
                for i in range(-30, 31):
                    for j in range(-30, 31):
                        if 25 * 25 < i * i + j * j < 30 * 30:
                            x, y = cx + i, cy + j
                            if 0 <= x < 288 and 0 <= y < 288:
                                img[y, x] = [255, 0, 0]
    return img


# ------------------------------------------------------------------------------
# AlphaZero MCTS (proper two-player self-play on canonical boards)
# ------------------------------------------------------------------------------
class AZMCTS:
    def __init__(
        self,
        net: AlphaZeroNet,
        device: torch.device,
        *,
        cpuct: float,
        num_sims: int,
        dirichlet_alpha: float,
        dirichlet_eps: float,
    ):
        self.net = net
        self.device = device
        self.cpuct = cpuct
        self.num_sims = num_sims
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

        self.P: Dict[bytes, np.ndarray] = {}
        self.N: Dict[bytes, np.ndarray] = {}
        self.W: Dict[bytes, np.ndarray] = {}
        self.valid: Dict[bytes, np.ndarray] = {}
        self.terminal: Dict[bytes, int] = {}
        # Root-only exploration (do NOT persist Dirichlet noise in self.P)
        self._root_key: Optional[bytes] = None
        self._root_prior_override: Optional[np.ndarray] = None

    def _key(self, canon_board: np.ndarray) -> bytes:
        return canon_board.astype(np.int8).tobytes()

    @torch.no_grad()
    def _eval(self, canon_board: np.ndarray) -> Tuple[np.ndarray, float]:
        x = torch.tensor(canon_board.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, v = self.net(x)
        p = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy().astype(np.float32)  # (9,)
        return p, float(v.item())

    def get_pi(self, board: np.ndarray, player: int, *, temp: float) -> np.ndarray:
        """Returns π(a|s) on the *real* board+player by running MCTS on canonical state."""
        canon = ttt_canonical(board, player)
        root_key = self._key(canon)

        # Ensure root is expanded exactly once (WITHOUT noise in stored priors).
        if root_key not in self.P:
            p, _v = self._eval(canon)
            valid_mask = ttt_valid_moves(board)
            p = p * valid_mask
            ps = float(p.sum())
            if ps <= 0:
                p = valid_mask / max(1e-8, float(valid_mask.sum()))
            else:
                p = p / ps

            self.P[root_key] = p.astype(np.float32)
            self.N[root_key] = np.zeros(9, dtype=np.float32)
            self.W[root_key] = np.zeros(9, dtype=np.float32)
            self.valid[root_key] = valid_mask.astype(np.float32)

        # Root-only Dirichlet noise (ephemeral for this move)
        self._root_key = root_key
        self._root_prior_override = None
        if self.dirichlet_eps > 0:
            # AlphaZero: add Dirichlet noise to root prior to encourage exploration
            p = self.P[root_key].copy()
            valid_mask = self.valid[root_key]
            noise = np.random.dirichlet([self.dirichlet_alpha] * 9).astype(np.float32)
            p = (1.0 - self.dirichlet_eps) * p + self.dirichlet_eps * noise
            p = p * valid_mask
            p = p / max(1e-8, float(p.sum()))
            self._root_prior_override = p.astype(np.float32)

        # Run sims
        for _ in range(self.num_sims):
            self._search(board, player)

        # Counts at root
        if root_key not in self.N:
            # extremely early edge-case
            mask = ttt_valid_moves(board)
            mask = mask / max(1e-8, mask.sum())
            return mask

        counts = self.N[root_key].copy()
        if temp <= 1e-8:
            pi = np.zeros_like(counts)
            pi[int(np.argmax(counts))] = 1.0
            return pi

        counts = np.power(counts, 1.0 / temp)
        s = float(counts.sum())
        if s <= 0:
            pi = np.ones_like(counts) / len(counts)
            return pi
        return counts / s

    def _search(self, board: np.ndarray, player: int) -> float:
        """
        Returns v from current player's perspective for state (board, player).
        Uses canonical representation internally.
        """
        w = ttt_winner(board)
        if w != 0:
            # terminal from current player's perspective
            if w == 2:
                return 0.0
            return 1.0 if w == player else -1.0

        canon = ttt_canonical(board, player)
        k = self._key(canon)

        if k not in self.P:
            p, v = self._eval(canon)
            valid_mask = ttt_valid_moves(board)
            p = p * valid_mask
            ps = float(p.sum())
            if ps <= 0:
                p = valid_mask / max(1e-8, float(valid_mask.sum()))
            else:
                p = p / ps

            self.P[k] = p.astype(np.float32)
            self.N[k] = np.zeros(9, dtype=np.float32)
            self.W[k] = np.zeros(9, dtype=np.float32)
            self.valid[k] = valid_mask.astype(np.float32)
            return v

        # select action via PUCT
        # Use root-only noisy prior only while searching from the current root.
        if self._root_key is not None and k == self._root_key and self._root_prior_override is not None:
            p = self._root_prior_override
        else:
            p = self.P[k]
        n = self.N[k]
        wsum = self.W[k]
        total_n = float(n.sum())
        best_a = -1
        best_u = -1e9
        for a in range(9):
            if self.valid[k][a] <= 0:
                continue
            q = 0.0 if n[a] <= 0 else wsum[a] / n[a]
            u = q + self.cpuct * p[a] * math.sqrt(total_n + 1e-8) / (1.0 + n[a])
            if u > best_u:
                best_u = u
                best_a = a

        if best_a < 0:
            return 0.0

        nb, np_player = ttt_next(board, player, best_a)
        v = -self._search(nb, np_player)  # switch perspective

        self.N[k][best_a] += 1.0
        self.W[k][best_a] += v
        return v


# ------------------------------------------------------------------------------
# Training workers
# ------------------------------------------------------------------------------
def encode_video(frames: List[np.ndarray], fps: int = 30) -> Optional[str]:
    if not HAS_VIDEO or not frames:
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            writer = imageio.get_writer(tmp.name, fps=fps, codec="libx264", quality=6, macro_block_size=16)
            try:
                for fr in frames:
                    writer.append_data(fr)
            finally:
                writer.close()
            tmp.seek(0)
            vid_bytes = tmp.read()
        return base64.b64encode(vid_bytes).decode()
    except Exception:
        return None


def worker_gym(config: Dict[str, Any], state: Dict[str, Any]) -> None:
    env = None
    run_dir = Path(config["run_dir"])
    try:
        set_all_seeds(int(config["seed"]))
        algo = config["algo"]
        env_name = config["env_name"]
        device = resolve_device(config["device"])

        env = gym.make(env_name, render_mode="rgb_array")
        obs_dim = int(env.observation_space.shape[0])
        act_dim = int(env.action_space.n)

        activation = activation_from_name(config["activation"])
        hidden_sizes = config["hidden_sizes"]
        lr = float(config["lr"])
        gamma = float(config["gamma"])

        record_every = int(config["record_every"])
        video_fps = int(config["video_fps"])

        _log(state, f"Gym worker: {algo} on {env_name} device={device}")
        (run_dir / "status.json").write_text(json.dumps({"status": "running", "started_at": datetime.now().isoformat()}, indent=2), encoding="utf-8")

        metrics_fp = open(run_dir / "metrics.jsonl", "a", encoding="utf-8")
        stats_fp = open(run_dir / "stats.jsonl", "a", encoding="utf-8")

        if algo == "DQN":
            agent = DQNAgent(
                obs_dim,
                act_dim,
                hidden_sizes,
                activation,
                lr,
                gamma,
                device,
                grad_clip=float(config["grad_clip"]),
            )
            buf = ReplayBuffer(int(config["replay_size"]))
            batch_size = int(config["batch_size"])
            warmup = int(config["warmup_steps"])
            target_every = int(config["target_sync_every"])
            train_every = int(config["train_every"])
            eps = float(config["eps_start"])
            eps_end = float(config["eps_end"])
            eps_decay = float(config["eps_decay"])

            global_step = 0
            episodes = 0
            while not state["stop_event"].is_set():
                obs, _ = env.reset(seed=int(config["seed"]) + episodes)
                done = False
                ep_reward = 0.0
                frames: List[np.ndarray] = []
                render_this = (episodes % record_every == 0)

                while not done and not state["stop_event"].is_set():
                    if render_this:
                        try:
                            frames.append(env.render())
                        except Exception:
                            pass

                    a = agent.act(obs, eps)
                    nobs, r, terminated, truncated, _ = env.step(a)
                    done = bool(terminated or truncated)
                    buf.push(ReplayItem(s=np.array(obs, dtype=np.float32), a=int(a), r=float(r), ns=np.array(nobs, dtype=np.float32), done=done))

                    obs = nobs
                    ep_reward += float(r)
                    global_step += 1

                    # train
                    loss = None
                    if len(buf) >= max(batch_size, warmup) and (global_step % train_every == 0):
                        loss = agent.update(buf.sample(batch_size))
                        state["stats"] = {**(state.get("stats") or {}), "loss": loss, "eps": eps, "buffer": len(buf)}

                    if global_step % target_every == 0:
                        agent.sync_target()

                    eps = max(eps_end, eps * eps_decay)

                # end episode
                episodes += 1
                state["metrics_queue"].put({"episode": episodes, "reward": ep_reward})
                jsonl_append(metrics_fp, {"episode": episodes, "reward": ep_reward, "ts": time.time()})
                if state.get("stats"):
                    jsonl_append(stats_fp, {"episode": episodes, "ts": time.time(), **(state.get("stats") or {})})
                if episodes % 50 == 0:
                    metrics_fp.flush()
                    stats_fp.flush()
                if render_this:
                    b64 = encode_video(frames, fps=video_fps)
                    if b64:
                        with state["video_lock"]:
                            state["latest_video_b64"] = b64
                            state["latest_video_episode"] = episodes
                            state["latest_video_version"] = int(state.get("latest_video_version", 0)) + 1
                        # also save latest replay to disk (optional)
                        (run_dir / "latest_replay.mp4.b64").write_text(b64, encoding="utf-8")

        else:  # PPO
            agent = PPOAgent(
                obs_dim,
                act_dim,
                hidden_sizes,
                activation,
                lr,
                gamma,
                device,
                clip_eps=float(config["ppo_clip"]),
                vf_coef=float(config["ppo_vf_coef"]),
                ent_coef=float(config["ppo_ent_coef"]),
                grad_clip=float(config["grad_clip"]),
                gae_lambda=float(config["ppo_gae_lambda"]),
            )
            update_every = int(config["ppo_update_every_steps"])
            ppo_epochs = int(config["ppo_epochs"])
            minibatch_size = int(config["ppo_minibatch"])

            episodes = 0
            steps_buf: List[np.ndarray] = []
            acts_buf: List[int] = []
            logp_buf: List[float] = []
            rews_buf: List[float] = []
            dones_buf: List[bool] = []
            vals_buf: List[float] = []

            frames: List[np.ndarray] = []

            while not state["stop_event"].is_set():
                obs, _ = env.reset(seed=int(config["seed"]) + episodes)
                done = False
                ep_reward = 0.0
                render_this = (episodes % record_every == 0)
                frames = []

                while not done and not state["stop_event"].is_set():
                    if render_this:
                        try:
                            frames.append(env.render())
                        except Exception:
                            pass

                    a, lp, v = agent.act(obs)
                    nobs, r, terminated, truncated, _ = env.step(a)
                    done = bool(terminated or truncated)

                    steps_buf.append(np.array(obs, dtype=np.float32))
                    acts_buf.append(int(a))
                    logp_buf.append(float(lp))
                    rews_buf.append(float(r))
                    dones_buf.append(done)
                    vals_buf.append(float(v))

                    obs = nobs
                    ep_reward += float(r)

                    if len(steps_buf) >= update_every:
                        adv, ret = gae_advantages(rews_buf, vals_buf, dones_buf, gamma, agent.gae_lambda)
                        batch = PPOBatch(
                            obs=torch.tensor(np.stack(steps_buf), dtype=torch.float32, device=device),
                            act=torch.tensor(acts_buf, dtype=torch.int64, device=device),
                            logp=torch.tensor(logp_buf, dtype=torch.float32, device=device),
                            adv=torch.tensor(adv, dtype=torch.float32, device=device),
                            ret=torch.tensor(ret, dtype=torch.float32, device=device),
                        )
                        stats = agent.update(batch, epochs=ppo_epochs, minibatch_size=minibatch_size)
                        state["stats"] = {**(state.get("stats") or {}), **stats}
                        steps_buf.clear()
                        acts_buf.clear()
                        logp_buf.clear()
                        rews_buf.clear()
                        dones_buf.clear()
                        vals_buf.clear()

                episodes += 1
                state["metrics_queue"].put({"episode": episodes, "reward": ep_reward})
                jsonl_append(metrics_fp, {"episode": episodes, "reward": ep_reward, "ts": time.time()})
                if state.get("stats"):
                    jsonl_append(stats_fp, {"episode": episodes, "ts": time.time(), **(state.get("stats") or {})})
                if episodes % 50 == 0:
                    metrics_fp.flush()
                    stats_fp.flush()
                if render_this:
                    b64 = encode_video(frames, fps=video_fps)
                    if b64:
                        with state["video_lock"]:
                            state["latest_video_b64"] = b64
                            state["latest_video_episode"] = episodes
                            state["latest_video_version"] = int(state.get("latest_video_version", 0)) + 1
                        (run_dir / "latest_replay.mp4.b64").write_text(b64, encoding="utf-8")

    except Exception as e:
        _log(state, f"CRITICAL: {e}")
        import traceback

        _log(state, traceback.format_exc())
    finally:
        try:
            metrics_fp.flush()
            stats_fp.flush()
            metrics_fp.close()
            stats_fp.close()
        except Exception:
            pass
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        state["running"] = False
        (run_dir / "status.json").write_text(json.dumps({"status": "stopped", "ended_at": datetime.now().isoformat()}, indent=2), encoding="utf-8")
        # clear active run file if it matches this run
        try:
            active = rl_load_active_run()
            if active and active.get("run_id") == config.get("run_id"):
                rl_clear_active_run()
        except Exception:
            pass


def worker_alphazero(config: Dict[str, Any], state: Dict[str, Any]) -> None:
    try:
        run_dir = Path(config["run_dir"])
        set_all_seeds(int(config["seed"]))
        device = resolve_device(config["device"])
        activation = activation_from_name(config["activation"])
        hidden_sizes = config["hidden_sizes"]
        lr = float(config["lr"])

        cpuct = float(config["az_cpuct"])
        num_sims = int(config["az_sims"])
        dir_a = float(config["az_dirichlet_alpha"])
        dir_eps = float(config["az_dirichlet_eps"])
        temp = float(config["az_temp"])
        eval_every = int(config.get("az_eval_every", 0))
        eval_games = int(config.get("az_eval_games", 0))

        train_batch = int(config["az_batch"])
        train_steps = int(config["az_train_steps"])
        replay_size = int(config["az_replay_size"])

        record_every = int(config["record_every"])
        video_fps = int(config["video_fps"])

        net = AlphaZeroNet(9, 9, hidden_sizes, activation).to(device)
        opt = optim.Adam(net.parameters(), lr=lr, weight_decay=float(config["az_l2"]))

        replay: deque[Tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=replay_size)  # (canon_flat, pi, z)
        last_results: deque[float] = deque(maxlen=200)  # +1 win, 0 draw, -1 loss for X

        _log(state, f"AlphaZero worker: TicTacToe self-play device={device} sims={num_sims} cpuct={cpuct}")
        (run_dir / "status.json").write_text(json.dumps({"status": "running", "started_at": datetime.now().isoformat()}, indent=2), encoding="utf-8")
        metrics_fp = open(run_dir / "metrics.jsonl", "a", encoding="utf-8")
        stats_fp = open(run_dir / "stats.jsonl", "a", encoding="utf-8")
        eval_fp = open(run_dir / "eval.jsonl", "a", encoding="utf-8")

        episodes = 0
        while not state["stop_event"].is_set():
            board = np.zeros((3, 3), dtype=np.int8)
            player = 1
            game_examples: List[Tuple[np.ndarray, np.ndarray, int]] = []  # (canon_flat, pi, player_at_state)
            frames: List[np.ndarray] = []
            render_this = (episodes % record_every == 0)

            mcts = AZMCTS(net, device, cpuct=cpuct, num_sims=num_sims, dirichlet_alpha=dir_a, dirichlet_eps=dir_eps)

            while True:
                if state["stop_event"].is_set():
                    break

                # get pi from MCTS
                pi = mcts.get_pi(board, player, temp=temp).astype(np.float32)
                valid = ttt_valid_moves(board)
                pi = pi * valid
                pi = pi / max(1e-8, float(pi.sum()))

                canon = ttt_canonical(board, player).astype(np.float32).flatten()
                game_examples.append((canon, pi, player))

                # sample action
                a = int(np.random.choice(9, p=pi))
                board, player = ttt_next(board, player, a)

                if render_this:
                    frames.append(ttt_render(board))

                w = ttt_winner(board)
                if w != 0:
                    # terminal
                    if w == 2:
                        z_final = 0.0
                    else:
                        z_final = float(w)  # winner is +1 or -1 in absolute players

                    # assign z from perspective of player at state: z = winner * player_at_state
                    for canon_flat, pi_s, p_at in game_examples:
                        z = z_final * float(p_at)
                        replay.append((canon_flat, pi_s, z))
                    break

            # train a bit
            loss_pi_v = None
            loss_v_v = None
            for _ in range(train_steps):
                if state["stop_event"].is_set():
                    break
                if len(replay) < max(64, train_batch):
                    break
                batch = random.sample(list(replay), train_batch)
                b_s, b_pi, b_z = zip(*batch)
                s = torch.tensor(np.stack(b_s), dtype=torch.float32, device=device)
                pi_t = torch.tensor(np.stack(b_pi), dtype=torch.float32, device=device)
                z_t = torch.tensor(np.array(b_z, dtype=np.float32), dtype=torch.float32, device=device)

                logits, v = net(s)
                logp = F.log_softmax(logits, dim=-1)
                loss_pi = -(pi_t * logp).sum(dim=-1).mean()
                loss_v = F.mse_loss(v, z_t)
                loss = loss_pi + loss_v

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), float(config["grad_clip"]))
                opt.step()

                loss_pi_v = float(loss_pi.item())
                loss_v_v = float(loss_v.item())

            # metrics
            episodes += 1
            # reward-like metric: +1 win, 0 draw, -1 loss for player 1 (X)
            w = ttt_winner(board)
            if w == 2:
                ep_reward = 0.0
            else:
                ep_reward = float(w)
            last_results.append(ep_reward)
            win_rate = float(np.mean([1.0 if r > 0 else 0.0 for r in last_results])) if last_results else 0.0
            draw_rate = float(np.mean([1.0 if r == 0 else 0.0 for r in last_results])) if last_results else 0.0
            loss_rate = float(np.mean([1.0 if r < 0 else 0.0 for r in last_results])) if last_results else 0.0

            state["stats"] = {
                **(state.get("stats") or {}),
                "loss_pi": loss_pi_v,
                "loss_v": loss_v_v,
                "replay": len(replay),
                "win_rate_200": round(win_rate, 3),
                "draw_rate_200": round(draw_rate, 3),
                "loss_rate_200": round(loss_rate, 3),
            }
            state["metrics_queue"].put({"episode": episodes, "reward": ep_reward})
            jsonl_append(metrics_fp, {"episode": episodes, "reward": ep_reward, "ts": time.time()})
            jsonl_append(stats_fp, {"episode": episodes, "ts": time.time(), **(state.get("stats") or {})})
            if episodes % 50 == 0:
                metrics_fp.flush()
                stats_fp.flush()
                eval_fp.flush()

            if render_this:
                b64 = encode_video(frames, fps=video_fps)
                if b64:
                    with state["video_lock"]:
                        state["latest_video_b64"] = b64
                        state["latest_video_episode"] = episodes
                        state["latest_video_version"] = int(state.get("latest_video_version", 0)) + 1
                    (run_dir / "latest_replay.mp4.b64").write_text(b64, encoding="utf-8")

            # Optional evaluation vs random (lightweight proof of learning)
            if eval_every > 0 and eval_games > 0 and (episodes % eval_every == 0) and not state["stop_event"].is_set():
                wins = draws = losses = 0
                for _ in range(eval_games):
                    b = np.zeros((3, 3), dtype=np.int8)
                    p = 1
                    # Use full search (or close to it) during eval. No Dirichlet.
                    m = AZMCTS(net, device, cpuct=cpuct, num_sims=num_sims, dirichlet_alpha=dir_a, dirichlet_eps=0.0)
                    while True:
                        ww = ttt_winner(b)
                        if ww != 0:
                            if ww == 2:
                                draws += 1
                            elif ww == 1:
                                wins += 1
                            else:
                                losses += 1
                            break
                        if p == 1:
                            pi_eval = m.get_pi(b, p, temp=0.0)
                            a = int(np.argmax(pi_eval))
                        else:
                            valid = np.where(ttt_valid_moves(b) > 0)[0]
                            a = int(np.random.choice(valid))
                        b, p = ttt_next(b, p, a)

                total = max(1, wins + draws + losses)
                state["stats"] = {
                    **(state.get("stats") or {}),
                    "eval_games": total,
                    "eval_win": wins,
                    "eval_draw": draws,
                    "eval_loss": losses,
                    "eval_win_rate": round(wins / total, 3),
                }
                jsonl_append(
                    eval_fp,
                    {
                        "episode": episodes,
                        "ts": time.time(),
                        "eval_games": total,
                        "eval_win": wins,
                        "eval_draw": draws,
                        "eval_loss": losses,
                        "eval_win_rate": round(wins / total, 3),
                    },
                )

    except Exception as e:
        _log(state, f"CRITICAL: {e}")
        import traceback

        _log(state, traceback.format_exc())
    finally:
        try:
            metrics_fp.flush()
            stats_fp.flush()
            eval_fp.flush()
            metrics_fp.close()
            stats_fp.close()
            eval_fp.close()
        except Exception:
            pass
        state["running"] = False
        (run_dir / "status.json").write_text(json.dumps({"status": "stopped", "ended_at": datetime.now().isoformat()}, indent=2), encoding="utf-8")
        try:
            active = rl_load_active_run()
            if active and active.get("run_id") == config.get("run_id"):
                rl_clear_active_run()
        except Exception:
            pass


# ------------------------------------------------------------------------------
# Sidebar UI (like LLM page)
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("🛠️ Configurator")

    algo = st.selectbox("Algorithm", ["DQN", "PPO", "AlphaZero"])

    if algo in ("DQN", "PPO"):
        env_name = st.selectbox("Environment (Gym)", ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"])
    else:
        env_name = "TicTacToe-v0 (self-play)"
        st.info("AlphaZero работает на TicTacToe self-play (честная двухигроковая игра для MCTS).")

    with st.expander("🧠 Neural Network", expanded=True):
        activation_name = st.selectbox("Activation", ["ReLU", "SiLU", "Tanh", "GELU"], index=0)
        num_layers = st.number_input("Hidden Layers", min_value=1, max_value=6, value=2)
        layer_size = st.number_input("Layer Size", min_value=16, max_value=1024, value=64, step=16)
        hidden_sizes = [int(layer_size)] * int(num_layers)

    with st.expander("⚙️ Common Training", expanded=True):
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)
        device_choice = st.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
        lr = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-1, value=1e-3, format="%.6f")
        gamma = st.slider("Gamma", min_value=0.8, max_value=0.999, value=0.99, step=0.001)
        grad_clip = st.number_input("Grad clip", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

    if algo == "DQN":
        with st.expander("🔁 DQN", expanded=False):
            replay_size = st.number_input("Replay size", 1000, 200000, 20000, step=1000)
            batch_size = st.number_input("Batch size", 16, 512, 64, step=16)
            warmup_steps = st.number_input("Warmup steps", 0, 20000, 1000, step=100)
            train_every = st.number_input("Train every (steps)", 1, 50, 1, step=1)
            target_sync_every = st.number_input("Target sync every (steps)", 10, 5000, 500, step=10)
            eps_start = st.slider("Epsilon start", 0.1, 1.0, 1.0, 0.05)
            eps_end = st.slider("Epsilon end", 0.0, 0.2, 0.05, 0.01)
            eps_decay = st.slider("Epsilon decay (mult per step)", 0.95, 0.9999, 0.995, 0.0001)
    elif algo == "PPO":
        with st.expander("🧠 PPO", expanded=False):
            ppo_clip = st.slider("Clip ε", 0.05, 0.3, 0.2, 0.01)
            ppo_gae_lambda = st.slider("GAE λ", 0.8, 0.99, 0.95, 0.01)
            ppo_epochs = st.number_input("Epochs per update", 1, 20, 4, step=1)
            ppo_minibatch = st.number_input("Minibatch size", 16, 2048, 128, step=16)
            ppo_update_every_steps = st.number_input("Update every (steps)", 64, 8192, 1024, step=64)
            ppo_vf_coef = st.slider("VF coef", 0.1, 1.0, 0.5, 0.05)
            ppo_ent_coef = st.slider("Entropy coef", 0.0, 0.05, 0.01, 0.001)
    else:
        with st.expander("🌲 AlphaZero (TicTacToe)", expanded=False):
            az_sims = st.number_input("MCTS sims", 5, 400, 50, step=5)
            az_cpuct = st.slider("c_puct", 0.5, 5.0, 1.5, 0.1)
            az_dirichlet_alpha = st.slider("Dirichlet α", 0.01, 1.0, 0.3, 0.01)
            az_dirichlet_eps = st.slider("Dirichlet ε", 0.0, 0.5, 0.25, 0.05)
            az_temp = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1)
            az_replay_size = st.number_input("Replay size", 256, 50000, 5000, step=256)
            az_batch = st.number_input("Batch size", 16, 2048, 128, step=16)
            az_train_steps = st.number_input("Train steps per game", 0, 50, 5, step=1)
            az_l2 = st.number_input("L2 weight decay", 0.0, 1e-2, 1e-4, format="%.6f")
            st.divider()
            st.caption("Оценка качества (пруф, что реально учится): net+MCTS vs random-O")
            az_eval_every = st.number_input("Eval every N games (0=off)", 0, 5000, 200, step=50)
            az_eval_games = st.number_input("Eval games", 0, 200, 20, step=5)

    with st.expander("🎬 Video", expanded=False):
        record_every = st.number_input("Record every N episodes", 1, 200, 10, step=1)
        video_fps = st.number_input("Video FPS", 5, 60, 30, step=1)
        show_replay_auto = st.checkbox("Показывать replay автоматически (может грузить браузер)", value=False)
        replay_autoplay = st.checkbox("Autoplay", value=False)
        replay_loop = st.checkbox("Loop", value=False)

    st.divider()
    is_running = bool(st.session_state.rl_state["running"])
    c1, c2 = st.columns(2)
    with c1:
        start = st.button("🚀 Start", type="primary", disabled=is_running, width="stretch")
    with c2:
        stop = st.button("🛑 Stop", disabled=not is_running, width="stretch")

    if start:
        st.session_state.rl_state["stop_event"].clear()
        st.session_state.rl_state["episode_rewards"] = []
        st.session_state.rl_state["stats"] = {}
        with st.session_state.rl_state["metrics_queue"].mutex:
            st.session_state.rl_state["metrics_queue"].queue.clear()

        run_id = rl_make_run_id()
        run_dir = RL_RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg: Dict[str, Any] = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "algo": algo,
            "env_name": env_name if algo != "AlphaZero" else "TicTacToe-v0",
            "activation": activation_name,
            "hidden_sizes": hidden_sizes,
            "seed": int(seed),
            "device": device_choice,
            "lr": float(lr),
            "gamma": float(gamma),
            "grad_clip": float(grad_clip),
            "record_every": int(record_every),
            "video_fps": int(video_fps),
        }

        if algo == "DQN":
            cfg.update(
                {
                    "replay_size": int(replay_size),
                    "batch_size": int(batch_size),
                    "warmup_steps": int(warmup_steps),
                    "train_every": int(train_every),
                    "target_sync_every": int(target_sync_every),
                    "eps_start": float(eps_start),
                    "eps_end": float(eps_end),
                    "eps_decay": float(eps_decay),
                }
            )
            target = worker_gym
        elif algo == "PPO":
            cfg.update(
                {
                    "ppo_clip": float(ppo_clip),
                    "ppo_gae_lambda": float(ppo_gae_lambda),
                    "ppo_epochs": int(ppo_epochs),
                    "ppo_minibatch": int(ppo_minibatch),
                    "ppo_update_every_steps": int(ppo_update_every_steps),
                    "ppo_vf_coef": float(ppo_vf_coef),
                    "ppo_ent_coef": float(ppo_ent_coef),
                }
            )
            target = worker_gym
        else:
            cfg.update(
                {
                    "az_sims": int(az_sims),
                    "az_cpuct": float(az_cpuct),
                    "az_dirichlet_alpha": float(az_dirichlet_alpha),
                    "az_dirichlet_eps": float(az_dirichlet_eps),
                    "az_temp": float(az_temp),
                    "az_replay_size": int(az_replay_size),
                    "az_batch": int(az_batch),
                    "az_train_steps": int(az_train_steps),
                    "az_l2": float(az_l2),
                    "az_eval_every": int(az_eval_every),
                    "az_eval_games": int(az_eval_games),
                }
            )
            target = worker_alphazero

        # Persist run config
        (run_dir / "config.json").write_text(
            json.dumps({"run_id": run_id, "started_at": datetime.now().isoformat(), "config": cfg}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        rl_save_active_run(run_id, cfg)
        st.session_state.rl_current_run_id = run_id

        st.session_state.rl_state["running"] = True
        t = threading.Thread(target=target, args=(cfg, st.session_state.rl_state), daemon=True)
        t.start()
        st.rerun()

    if stop:
        st.session_state.rl_state["stop_event"].set()
        st.rerun()


# ------------------------------------------------------------------------------
# Main UI (LLM-like sections)
# ------------------------------------------------------------------------------
def _selected_run_dir() -> Tuple[Optional[str], Optional[Path], bool]:
    """Returns (run_id, run_dir, is_live_mode)."""
    if st.session_state.rl_view_mode and st.session_state.rl_view_run_id:
        rid = st.session_state.rl_view_run_id
        return rid, (RL_RUNS_DIR / rid), False
    rid = st.session_state.get("rl_current_run_id")
    if rid:
        return rid, (RL_RUNS_DIR / rid), True
    return None, None, True


def _load_run_frames(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[Dict[str, Any]]]:
    df_metrics = pd.DataFrame(jsonl_tail(run_dir / "metrics.jsonl", max_lines=5000))
    df_stats = pd.DataFrame(jsonl_tail(run_dir / "stats.jsonl", max_lines=5000))
    df_eval = pd.DataFrame(jsonl_tail(run_dir / "eval.jsonl", max_lines=5000))
    cfg = None
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = None
    return df_metrics, df_stats, df_eval, cfg


def _render_monitoring(run_id: Optional[str], run_dir: Optional[Path]) -> None:
    if not run_id or not run_dir or not run_dir.exists():
        st.info("Запустите эксперимент (🚀 Запуск) или выберите run в 📜 История.")
        return

    df_metrics, df_stats, df_eval, cfg = _load_run_frames(run_dir)

    c_left, c_right = st.columns([1, 1])

    # --- Self-play / training (primary)
    with c_left:
        st.subheader("📊 Мониторинг — Self-play / Training")
        if cfg:
            algo = cfg.get("config", {}).get("algo")
        else:
            algo = None

        # Prefer AlphaZero self-play rates if available, otherwise reward curve
        if not df_stats.empty and "win_rate_200" in df_stats.columns:
            d = df_stats.sort_values("episode") if "episode" in df_stats.columns else df_stats
            tail = d.tail(2000)
            if len(tail) > 1200:
                step = max(1, len(tail) // 1200)
                tail = tail.iloc[::step]
            plot = tail.set_index("episode")[["win_rate_200", "draw_rate_200", "loss_rate_200"]]
            st.line_chart(plot, height=260)
            last = d.iloc[-1]
            m1, m2, m3 = st.columns(3)
            m1.metric("Win(200)", f"{float(last.get('win_rate_200', 0)):.3f}")
            m2.metric("Draw(200)", f"{float(last.get('draw_rate_200', 0)):.3f}")
            m3.metric("Loss(200)", f"{float(last.get('loss_rate_200', 0)):.3f}")
        elif not df_metrics.empty and "reward" in df_metrics.columns:
            d = df_metrics.sort_values("episode") if "episode" in df_metrics.columns else df_metrics
            d["Avg10"] = d["reward"].rolling(10, min_periods=1).mean()
            tail = d.tail(2000)
            if len(tail) > 1200:
                step = max(1, len(tail) // 1200)
                tail = tail.iloc[::step]
            st.line_chart(tail.set_index("episode")[["reward", "Avg10"]], height=260)
        else:
            st.info("Нет метрик self-play/training в run.")

        # Show last stats snapshot
        if not df_stats.empty:
            st.caption("Последние training stats")
            st.json(dict(df_stats.iloc[-1]), expanded=False)
        else:
            # live fallback
            if st.session_state.rl_state.get("stats"):
                st.caption("Последние training stats (live)")
                st.json(st.session_state.rl_state.get("stats") or {}, expanded=False)

    # --- Eval (side-by-side)
    with c_right:
        st.subheader("✅ Мониторинг — Eval")
        if df_eval.empty:
            st.info("Eval ещё не запускался. Включите в настройках AlphaZero: `Eval every N games` > 0.")
        else:
            d = df_eval.sort_values("episode") if "episode" in df_eval.columns else df_eval
            tail = d.tail(2000)
            st.line_chart(tail.set_index("episode")[["eval_win_rate"]], height=260)
            last = d.iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("Eval win-rate", f"{float(last.get('eval_win_rate', 0)):.3f}")
            c2.metric("Eval games", f"{int(last.get('eval_games', 0))}")
            c3.metric("Eval @episode", f"{int(last.get('episode', 0))}")
            st.caption("Последний eval snapshot")
            st.json(dict(last), expanded=False)


def _render_replay(run_id: Optional[str], run_dir: Optional[Path]) -> None:
    st.subheader("🎬 Replay")
    if not HAS_VIDEO:
        st.warning("Видео недоступно: установите `imageio-ffmpeg`.")
        return
    if not run_dir or not run_dir.exists():
        st.info("Нет run для реплея.")
        return

    vid_b64: Optional[str]
    ep = 0
    ver = 0

    if st.session_state.rl_view_mode and st.session_state.rl_view_run_id:
        vid_path = run_dir / "latest_replay.mp4.b64"
        vid_b64 = vid_path.read_text(encoding="utf-8") if vid_path.exists() else None
    else:
        with st.session_state.rl_state["video_lock"]:
            vid_b64 = st.session_state.rl_state.get("latest_video_b64")
            ep = int(st.session_state.rl_state.get("latest_video_episode") or 0)
            ver = int(st.session_state.rl_state.get("latest_video_version") or 0)

    if not vid_b64:
        st.markdown("*Пока нет реплея — он появится после N эпизодов (см. Video settings).*")
        return

    with st.container(border=True):
        if "rl_ui_last_video_version" not in st.session_state:
            st.session_state.rl_ui_last_video_version = -1

        show_now = False
        if show_replay_auto and ver != st.session_state.rl_ui_last_video_version:
            show_now = True
        if st.button("▶️ Показать replay сейчас", key=f"show_replay_{run_id}_{ver}"):
            show_now = True

        st.caption(f"Replay готов (episode {ep}). Версия: {ver}.")

        if show_now:
            st.session_state.rl_ui_last_video_version = ver
            autoplay_attr = "autoplay" if replay_autoplay else ""
            loop_attr = "loop" if replay_loop else ""
            st.markdown(
                f"""
<div>
  <video controls {autoplay_attr} {loop_attr} muted playsinline
    style="width:100%; max-height:520px; border:1px solid #444; border-radius:8px;">
    <source src="data:video/mp4;base64,{vid_b64}" type="video/mp4" />
  </video>
</div>
""",
                unsafe_allow_html=True,
            )


def _render_history() -> None:
    st.subheader("📜 История run'ов")
    st.caption("Выбор run для просмотра делается здесь (а не в Configurator).")
    runs = sorted([p for p in RL_RUNS_DIR.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        st.info("Пока нет сохранённых run'ов.")
        return
    rows = []
    for p in runs[:200]:
        cfg_path = p / "config.json"
        status_path = p / "status.json"
        algo = env = started = status = ""
        if cfg_path.exists():
            try:
                j = json.loads(cfg_path.read_text(encoding="utf-8"))
                started = j.get("started_at", "")
                cc = j.get("config", {})
                algo = cc.get("algo", "")
                env = cc.get("env_name", "")
            except Exception:
                pass
        if status_path.exists():
            try:
                s = json.loads(status_path.read_text(encoding="utf-8"))
                status = s.get("status", "")
            except Exception:
                pass
        rows.append({"run_id": p.name, "algo": algo, "env": env, "status": status, "started_at": started})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    run_labels = ["(live) текущий"] + [p.name for p in runs]
    selected = st.selectbox("Выбрать run для просмотра", run_labels, index=0, key="rl_history_run_select")
    st.session_state.rl_view_mode = (selected != "(live) текущий")
    st.session_state.rl_view_run_id = None if selected == "(live) текущий" else selected

    active = rl_load_active_run()
    if active and active.get("run_id"):
        st.caption(f"Active run: `{active['run_id']}`")
    if st.button("🧹 Очистить active_run.json", width="stretch"):
        rl_clear_active_run()
        st.rerun()


def _render_data(run_id: Optional[str], run_dir: Optional[Path]) -> None:
    st.subheader("💾 Данные")
    if not run_id or not run_dir or not run_dir.exists():
        st.info("Выберите run в 📜 История или запустите новый.")
        return
    st.caption(f"Текущий run: `{run_id}`")
    for name in ["config.json", "metrics.jsonl", "stats.jsonl", "eval.jsonl", "status.json"]:
        path = run_dir / name
        if path.exists():
            st.download_button(
                label=f"⬇️ Скачать {name}",
                data=path.read_bytes(),
                file_name=f"{run_id}_{name}",
                width="stretch",
            )


def _render_tutorial() -> None:
    st.subheader("📚 Учебник")
    st.markdown(
        """
### Как читать метрики
- **Self-play**: это “внутренняя” динамика обучения (в TicTacToe сильная игра стремится к ничьим).
- **Eval**: это внешний показатель качества (например, *net+MCTS vs random*). Он нужен, чтобы понимать, что модель реально улучшает игру.

### Что важно в AlphaZero
- **MCTS sims**: больше — сильнее поиск, но медленнее.
- **c_puct**: баланс exploration/exploitation.
- **Dirichlet ε/α**: exploration в корне (слишком много → шум, слишком мало → стагнация).
- **Temperature**: 0 → детерминированная игра, >0 → больше разнообразия.
""",
        unsafe_allow_html=False,
    )


def render_page() -> None:
    run_id, run_dir, _is_live = _selected_run_dir()

    tabs = st.tabs(["🚀 Запуск", "📊 Мониторинг", "📜 История", "💾 Данные", "📚 Учебник"])

    with tabs[0]:
        st.subheader("🚀 Запуск")
        st.caption("Запуск настраивается в левом Configurator (как в LLM).")
        if run_id:
            st.markdown(f"**Текущий run**: `{run_id}`")
        active = rl_load_active_run()
        if active and active.get("run_id"):
            st.markdown(f"**Active run**: `{active['run_id']}`")
        if st.session_state.rl_state.get("running"):
            st.success("Training: RUNNING")
        else:
            st.info("Training: STOPPED")

    with tabs[1]:
        _render_monitoring(run_id, run_dir)
        _render_replay(run_id, run_dir)
        if st.session_state.rl_state["logs"]:
            with st.expander("🧾 Logs", expanded=False):
                for line in st.session_state.rl_state["logs"][-120:]:
                    st.text(line)

    with tabs[2]:
        _render_history()

    with tabs[3]:
        _render_data(run_id, run_dir)

    with tabs[4]:
        _render_tutorial()


if fragment:
    @fragment(run_every=0.8)
    def _refresh():
        render_page()

    _refresh()
else:
    render_page()
    if st.session_state.rl_state["running"] and not st.session_state.rl_view_mode:
        time.sleep(0.8)
        st.rerun()


