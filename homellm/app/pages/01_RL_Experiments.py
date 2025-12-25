"""Reinforcement Learning Experiments Page"""
import streamlit as st
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import threading
import queue
import pandas as pd
import io
from PIL import Image
import tempfile
import base64

# Check for gymnasium
try:
    # Headless rendering for Docker/Linux (pygame/SDL)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

# Video encoding (mp4)
try:
    import imageio.v2 as imageio  # v2 API is stable
    import imageio_ffmpeg  # noqa: F401  (ensures ffmpeg binary is available)
    HAS_VIDEO = True
except Exception:
    HAS_VIDEO = False

# Check for st.fragment support (Streamlit 1.33+)
fragment = getattr(st, "fragment", getattr(st, "experimental_fragment", None))

st.set_page_config(page_title="RL Experiments", page_icon="🤖", layout="wide")

st.title("🤖 RL Experiments")
st.caption("Обучение агентов с подкреплением в реальном времени")

if not HAS_GYM:
    st.error("Библиотека `gymnasium` не установлена. Пожалуйста, выполните `pip install 'gymnasium[classic_control]'`")
    st.stop()

# ==============================================================================
# MODELS & AGENTS
# ==============================================================================

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, act_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

# ==============================================================================
# TRAINING WORKER
# ==============================================================================

# Global state for the training thread
if 'rl_state' not in st.session_state:
    st.session_state.rl_state = {
        'running': False,
        'stop_event': threading.Event(),
        'metrics_queue': queue.Queue(),
        # Latest replay video (mp4 bytes) - stable browser playback
        'latest_video_bytes': None,
        'latest_video_b64': None,
        'latest_video_episode': 0,
        'video_lock': threading.Lock(),
        'episode_rewards': [],
        'logs': []
    }

def train_worker(env_name, lr, hidden_size, gamma, state):
    """Background training loop (Simple REINFORCE for MVP)"""
    try:
        env = gym.make(env_name, render_mode="rgb_array")
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        
        policy = PolicyNetwork(obs_dim, act_dim, hidden_size)
        optimizer = optim.Adam(policy.parameters(), lr=lr)
        
        state['running'] = True
        
        episodes = 0
        record_every = 5  # record replay every N episodes
        replay_fps = 30
        
        while not state['stop_event'].is_set():
            obs, _ = env.reset()
            done = False
            log_probs = []
            rewards = []
            frames = []
            
            # Render frequently (every 2nd episode or even every episode if fast)
            # For "real time" feel, we want to see almost every episode initially.
            render_this = HAS_VIDEO and (episodes % record_every == 0)
            
            # Collect trajectory
            while not done:
                if render_this:
                    try:
                        frame = env.render()  # numpy RGB array
                        frames.append(frame)
                    except Exception:
                        pass

                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                probs = policy(obs_t)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                obs, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                
                log_probs.append(log_prob)
                rewards.append(reward)
                
                if state['stop_event'].is_set():
                    break

            # Encode replay video at end of episode (mp4) and store bytes
            if render_this and frames:
                try:
                    # Write to a temp mp4 file then read bytes (most compatible with Streamlit)
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
                        writer = imageio.get_writer(
                            tmp.name,
                            fps=replay_fps,
                            codec="libx264",
                            quality=7,
                            macro_block_size=16,  # keep codec-friendly (may resize)
                        )
                        try:
                            for fr in frames:
                                writer.append_data(fr)
                        finally:
                            writer.close()
                        tmp.seek(0)
                        video_bytes = tmp.read()
                    with state['video_lock']:
                        state['latest_video_bytes'] = video_bytes
                        state['latest_video_b64'] = base64.b64encode(video_bytes).decode()
                        state['latest_video_episode'] = episodes + 1
                except Exception as e:
                    state['logs'].append(f"Video encode error: {e}")

            # Calculate returns
            R = 0
            returns = []
            for r in rewards[::-1]:
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
            # Policy Update
            policy_loss = []
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-log_prob * R)
            optimizer.zero_grad()
            if policy_loss:
                policy_loss = torch.cat(policy_loss).sum()
                policy_loss.backward()
                optimizer.step()
            
            total_reward = sum(rewards)
            episodes += 1
            
            # Send metrics
            state['metrics_queue'].put({'episode': episodes, 'reward': total_reward})
            
    except Exception as e:
        state['logs'].append(f"Error: {e}")
    finally:
        env.close()
        state['running'] = False

# ==============================================================================
# UI LAYOUT
# ==============================================================================

col_params, col_viz = st.columns([1, 2])

with col_params:
    st.header("⚙️ Настройки")
    env_name = st.selectbox("Среда (Environment)", ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"])
    
    st.subheader("Гиперпараметры")
    lr = st.number_input("Learning Rate", value=0.001, step=0.0001, format="%.4f")
    gamma = st.slider("Gamma (Discount)", 0.8, 0.999, 0.99, 0.001)
    hidden_size = st.slider("Hidden Size", 32, 256, 128, 32)
    
    st.divider()
    
    # Control Buttons
    run_btn = st.button("▶️ Start Training", type="primary", disabled=st.session_state.rl_state['running'])
    stop_btn = st.button("⏹️ Stop", disabled=not st.session_state.rl_state['running'])

# Logic to handle buttons
if run_btn:
    st.session_state.rl_state['stop_event'].clear()
    st.session_state.rl_state['episode_rewards'] = []
    
    # Clear queues
    with st.session_state.rl_state['metrics_queue'].mutex:
        st.session_state.rl_state['metrics_queue'].queue.clear()
        
    thread = threading.Thread(target=train_worker, args=(env_name, lr, hidden_size, gamma, st.session_state.rl_state))
    thread.daemon = True
    thread.start()
    st.rerun()

if stop_btn:
    st.session_state.rl_state['stop_event'].set()
    st.rerun()

# ==============================================================================
# VISUALIZATION LOGIC
# ==============================================================================

def render_visualization():
    """Renders charts and video frames. Called inside fragment or loop."""
    # Consume metrics
    q = st.session_state.rl_state['metrics_queue']
    while not q.empty():
        item = q.get()
        st.session_state.rl_state['episode_rewards'].append(item)


def render_charts():
    st.subheader("📈 Результаты")
    rewards_df = pd.DataFrame(st.session_state.rl_state['episode_rewards'])

    if not rewards_df.empty:
        rewards_df['MA_10'] = rewards_df['reward'].rolling(window=10, min_periods=1).mean()
        st.line_chart(rewards_df.set_index('episode')[['reward', 'MA_10']], height=250)

        last_ep = rewards_df.iloc[-1]
        window_n = int(min(10, len(rewards_df)))
        st.markdown(
            f"**Episode:** {int(last_ep['episode'])} | "
            f"**Last Reward:** {last_ep['reward']:.1f} | "
            f"**Avg({window_n}):** {last_ep['MA_10']:.1f}"
        )
    else:
        st.info("Waiting for training data...")


def render_video():
    st.subheader("🎬 Replay")
    with st.container(border=True):
        if not HAS_VIDEO:
            st.warning("Видео недоступно: установите `imageio-ffmpeg` (и пересоберите контейнер).")
            return

        vid_b64 = None
        ep = 0
        with st.session_state.rl_state['video_lock']:
            vid_b64 = st.session_state.rl_state.get('latest_video_b64')
            ep = int(st.session_state.rl_state.get('latest_video_episode', 0) or 0)

        if not vid_b64:
            st.markdown("*Идёт обучение... Видео появится после первого записанного эпизода.*")
            return

        # Embed MP4 directly in HTML to avoid Streamlit media-file race conditions.
        html = (
            f"<div>"
            f"<div style='font-weight:600; margin-bottom:6px;'>Replay (episode {ep})</div>"
            f"<video controls autoplay loop muted playsinline style='width:100%; max-height:520px; border:1px solid #444; border-radius:8px;'>"
            f"<source src='data:video/mp4;base64,{vid_b64}' type='video/mp4' />"
            f"</video>"
            f"</div>"
        )
        st.markdown(html, unsafe_allow_html=True)

    # Debug Logs
    if st.session_state.rl_state['logs']:
        with st.expander("System Logs", expanded=True):
            for log in st.session_state.rl_state['logs']:
                st.error(log)
    
    # Queue Status
    # st.caption(f"Queue size: {st.session_state.rl_state['metrics_queue'].qsize()} | Frames: {st.session_state.rl_state['frame_queue'].qsize()}")

# ==============================================================================
# MAIN RENDER LOOP
# ==============================================================================

with col_viz:
    if fragment:
        # Charts can refresh relatively often
        @fragment(run_every=0.3)
        def charts_fragment():
            render_visualization()
            render_charts()

        # Video should refresh редко (иначе будем гонять огромный base64 каждый тик)
        @fragment(run_every=2.0)
        def video_fragment():
            render_video()

        charts_fragment()
        video_fragment()
        
    else:
        # Legacy Rerun approach
        render_visualization()
        render_charts()
        render_video()

        if st.session_state.rl_state['running']:
            time.sleep(0.5)
            st.rerun()
