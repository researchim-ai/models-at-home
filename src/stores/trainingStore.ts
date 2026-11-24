import { create } from 'zustand'
import type { TrainingConfig, TrainingRun, MetricsSnapshot, TrainingStage } from '@/api/types'

interface TrainingState {
  activeRunId: string | null
  runs: TrainingRun[]
  currentMetrics: MetricsSnapshot | null
  metricsHistory: MetricsSnapshot[]
  config: Partial<TrainingConfig>
  selectedChatModel: string | null

  setActiveRunId: (id: string | null) => void
  setRuns: (runs: TrainingRun[]) => void
  pushMetrics: (m: MetricsSnapshot) => void
  hydrateMetrics: (runId: string, history: MetricsSnapshot[], latest: MetricsSnapshot | null) => void
  clearMetrics: () => void
  updateConfig: (partial: Partial<TrainingConfig>) => void
  setStage: (stage: TrainingStage) => void
  resetConfig: () => void
  setSelectedChatModel: (path: string | null) => void
}

// Baseline common to every stage. Values mirror the defaults in the original
// models-at-home Streamlit LLM page (homellm/app/LLM.py) so behaviour matches.
const baseDefaults: Partial<TrainingConfig> = {
  model_type: 'home',
  optimizer: 'adamw',
  batch_size: 4,
  gradient_accumulation: 8,
  weight_decay: 0.1,
  betas: '0.9,0.95',
  eps: 1e-8,
  mixed_precision: 'bf16',
  fp16_pure: false,
  flash_attention: true,
  use_flash_attention: true,
  gradient_checkpointing: true,
  grad_checkpoint: true,
  liger_kernel: true,
  use_liger: true,
  liger_fused_ce: true,
  output_dir: 'out',
  save_every: 100,
  log_every: 10,
  export_on_checkpoint: true,
  merge_lora: true,
  max_grad_norm: 1.0,
  min_lr_ratio: 0.0,
  scheduler_resync_on_resume: true,
  parallel_mode: 'single',
  training_backend: 'models-at-home',
  // Architecture (used for pretrain from scratch)
  hidden_size: 512,
  num_layers: 8,
  n_heads: 8,
  vocab_size: 50257,
  num_experts: 8,
  num_experts_per_tok: 2,
  expert_type: 'swiglu',
  // Tuning
  tuning_method: 'full',
  use_lora: false,
  lora_rank: 32,
  lora_r: 32,
  lora_alpha: 32,
  lora_dropout: 0.05,
  lora_target_modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
}

// Stage-specific presets — these ONLY get applied when switching to that stage
// and replace stage-owned fields without clobbering general toggles the user set.
export const stagePresets: Record<TrainingStage, Partial<TrainingConfig>> = {
  pretrain: {
    stage: 'pretrain',
    learning_rate: 5e-4,
    lr_scheduler: 'cosine',
    warmup_steps: 1000,
    training_mode: 'steps',
    max_steps: 10000,
    num_epochs: 1,
    epochs: 1,
    seq_len: 2048,
    hidden_size: 512,
    num_layers: 8,
    n_heads: 8,
  },
  continual_pretrain: {
    stage: 'continual_pretrain',
    learning_rate: 5e-5,
    lr_scheduler: 'cosine',
    warmup_steps: 200,
    training_mode: 'steps',
    max_steps: 10000,
    num_epochs: 1,
    epochs: 1,
    seq_len: 2048,
  },
  sft: {
    stage: 'sft',
    learning_rate: 2e-5,
    lr_scheduler: 'cosine',
    warmup_steps: 100,
    training_mode: 'epochs',
    max_steps: 1000,
    num_epochs: 3,
    epochs: 3,
    sft_max_seq_length: 2048,
    sft_data_format: 'chat',
    sft_packing: false,
    sft_train_on_completions_only: false,
  },
  grpo: {
    stage: 'grpo',
    learning_rate: 5e-6,
    lr_scheduler: 'constant_with_warmup',
    warmup_steps: 0,
    training_mode: 'steps',
    max_steps: 500,
    grpo_algorithm: 'grpo',
    grpo_group_size: 8,
    grpo_temperature: 0.7,
    grpo_max_new_tokens: 1024,
    grpo_kl_weight: 0,
    grpo_clip_eps_high: 0.2,
    grpo_clip_eps_low: 0.2,
    grpo_prompt_batch_size: 8,
    grpo_train_batch_size: 2,
    grpo_max_prompts: 1000,
    grpo_reward_fn: 'exact_match',
  },
}

const defaultConfig: Partial<TrainingConfig> = {
  ...baseDefaults,
  ...stagePresets.pretrain,
}

export const useTrainingStore = create<TrainingState>((set) => ({
  activeRunId: null,
  runs: [],
  currentMetrics: null,
  metricsHistory: [],
  config: { ...defaultConfig },
  selectedChatModel: null,

  setActiveRunId: (activeRunId) => set(state => activeRunId === state.activeRunId
    ? state
    : { activeRunId, currentMetrics: null, metricsHistory: [] }),
  setSelectedChatModel: (selectedChatModel) => set({ selectedChatModel }),
  setRuns: (runs) => set({ runs }),

  pushMetrics: (m) => set(state => {
    const points = new Map(state.metricsHistory.map(point => [point.step, point]))
    const previous = points.get(m.step)
    if (!previous || m.timestamp >= previous.timestamp) points.set(m.step, m)
    const current = state.currentMetrics
    const newest = !current || m.step > current.step || (m.step === current.step && m.timestamp >= current.timestamp)
    return { currentMetrics: newest ? m : current,
      metricsHistory: [...points.values()].sort((a, b) => a.step - b.step).slice(-500) }
  }),

  hydrateMetrics: (runId, history, latest) => set(state => {
    if (state.activeRunId !== runId) return state
    const points = new Map(history.map(point => [point.step, point]))
    if (latest) points.set(latest.step, latest)
    for (const point of state.metricsHistory) {
      const previous = points.get(point.step)
      if (!previous || point.timestamp >= previous.timestamp) points.set(point.step, point)
    }
    const merged = [...points.values()].sort((a, b) => a.step - b.step).slice(-500)
    const candidates = [state.currentMetrics, latest, merged.at(-1)].filter((point): point is MetricsSnapshot => !!point)
    const current = candidates.sort((a, b) => b.step - a.step || b.timestamp - a.timestamp)[0] ?? null
    return { metricsHistory: merged, currentMetrics: current }
  }),

  clearMetrics: () => set({ currentMetrics: null, metricsHistory: [] }),

  updateConfig: (partial) =>
    set((s) => ({ config: { ...s.config, ...partial } })),

  setStage: (stage) =>
    set((s) => {
      const preset = stagePresets[stage]
      const next = { ...s.config, ...preset, model_type: stage === 'pretrain' ? 'home' : 'hf' }
      return { config: next }
    }),

  resetConfig: () => set({ config: { ...defaultConfig } }),
}))
