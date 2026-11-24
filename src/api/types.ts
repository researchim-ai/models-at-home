export type TrainingStage = 'pretrain' | 'continual_pretrain' | 'sft' | 'grpo'
export type VLMStage = 'vlm_pretrain' | 'vlm_sft' | 'vlm_grpo'
export type RunStatus = 'running' | 'completed' | 'error' | 'stopped' | 'initializing' | 'building_model' | 'resumed' | 'unknown' | 'starting' | 'training'

export type OptimizerName = 'adamw' | 'adamw_8bit' | 'muon' | 'magma_adamw'
export type TuningMethod = 'full' | 'lora' | 'qlora'

export interface VLMPreflightReport {
  ready: boolean
  checks: Array<{ name: string; status: 'pass' | 'warning' | 'error'; detail: string }>
  datasets: Array<{ kind: string; scanned_rows: number; valid_rows: number; images_checked: number; complete: boolean }>
  sample: null | {
    line: number
    images: Array<{ reference: string; src: string; width: number; height: number }>
    expanded_tokens: number
    supervised_tokens: number
    omitted_tokens: number
    tokens: Array<{ index: number; id: number; text: string; trainable: boolean }>
  }
}
export type TrainingMode = 'epochs' | 'steps'

export interface TrainingConfig {
  stage: TrainingStage
  model_type: string
  model_blueprint?: string
  model_blueprint_spec?: Record<string, unknown> | null
  model_path: string
  model_id?: string
  model_name_input?: string
  dataset_path: string
  data_path?: string
  experiment_name: string
  optimizer: OptimizerName
  batch_size: number
  gradient_accumulation: number
  learning_rate: number
  weight_decay?: number
  betas?: string
  eps?: number
  max_steps: number
  epochs: number
  training_mode?: TrainingMode
  warmup_steps: number
  lr_scheduler: string
  lr_schedule?: string
  scheduler_resync_on_resume?: boolean
  mixed_precision: string
  fp16_pure?: boolean
  flash_attention: boolean
  use_flash_attention?: boolean
  gradient_checkpointing: boolean
  grad_checkpoint?: boolean
  liger_kernel: boolean
  use_liger?: boolean
  liger_fused_ce?: boolean
  output_dir: string
  save_every: number
  log_every: number
  export_on_checkpoint?: boolean
  merge_lora?: boolean
  max_grad_norm: number
  min_lr_ratio: number

  // Architecture (pretrain from scratch)
  hidden_size?: number
  num_layers?: number
  n_heads?: number
  vocab_size?: number
  arch_preset?: string
  num_experts?: number
  num_experts_per_tok?: number
  expert_type?: string

  // Tuning
  tuning_method?: TuningMethod

  // LoRA
  use_lora: boolean
  lora_rank: number
  lora_r?: number
  lora_alpha: number
  lora_dropout: number
  lora_target_modules: string[]

  // GRPO-specific
  grpo_algorithm: 'grpo' | 'dapo' | 'dr_grpo' | 'sdpo'
  grpo_group_size: number
  grpo_temperature: number
  grpo_max_new_tokens: number
  grpo_kl_weight: number
  grpo_clip_eps_high: number
  grpo_clip_eps_low: number
  grpo_prompt_batch_size: number
  grpo_train_batch_size: number
  grpo_max_prompts: number
  grpo_learning_rate?: number
  grpo_max_steps?: number
  grpo_max_optim_steps?: number
  grpo_reward_fn: string
  grpo_dataset_path?: string

  // Distributed
  parallel_mode: string
  gpu_ids: number[]
  deepspeed_config: string

  // Unsloth
  training_backend: string

  // SFT
  sft_chat_template?: string
  sft_data_format?: 'chat' | 'instruct'
  sft_max_seq_length?: number
  sft_packing?: boolean
  sft_train_on_completions_only?: boolean

  // Continual pretrain
  base_model_path?: string

  // Seq length (pretrain)
  seq_len?: number

  // Epochs vs max_steps
  num_epochs?: number
}

export interface TrainingRun {
  run_id: string
  stage: TrainingStage | VLMStage | 'vla_sft' | 'world_model' | 'world_foundation'
  final_model_path?: string | null
  process_running?: boolean
  status: RunStatus
  started_at: string
  finished_at?: string
  model_type: string
  experiment_name: string
  config: Partial<TrainingConfig> & { domain?: string; action_type?: string; state_dim?: number }
  current_step: number
  total_steps: number
  checkpoints: string[]
}

export interface MetricsSnapshot {
  elapsed_seconds?: number
  step_seconds?: number
  estimated_remaining_seconds?: number
  action_loss?: number
  action_mae?: number
  action_accuracy?: number
  training_warning?: string | null
  status?: string
  error?: string | null
  final_model_dir?: string
  step: number
  loss: number | null
  learning_rate: number | null
  grad_norm?: number | null
  epoch?: number | null
  samples_per_second?: number | null
  gpu_stats?: GpuStats[]
  val_loss?: number | null
  reward?: number | null
  kl_divergence?: number | null
  timestamp: number
}

export interface GpuStats {
  id: number
  memory_used_gb: number
  memory_total_gb: number
  memory_percent: number
  utilization: number | null
}

export interface ModelInfo {
  name: string
  path: string
  size_bytes: number
  type: 'local' | 'trained' | 'checkpoint' | 'lora'
  architecture?: string
  modified_at: string
}

export interface DatasetInfo {
  name: string
  path: string
  size_bytes: number
  format: 'jsonl' | 'txt' | 'parquet' | 'arrow' | 'csv'
  num_rows?: number
  modified_at: string
}

export interface HFDownloadRequest {
  repo_id: string
  save_name?: string
}

export interface HFDatasetDownloadRequest {
  repo_id: string
  subset?: string
  split?: string
  save_as: string
  limit?: number
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
  image_url?: string
}

export interface ChatRequest {
  messages: ChatMessage[]
  model_path: string
  backend: 'transformers' | 'vllm' | 'llama.cpp'
  temperature: number
  top_p: number
  top_k: number
  max_tokens: number
  stream: boolean
}

export interface AgentSession {
  id: string
  name: string
  messages: ChatMessage[]
  created_at: string
}

export interface AgentTool {
  name: string
  category: string
  description: string
  arguments: Record<string, string>
}

export interface AgentToolGroup {
  category: string
  title: string
  tools: AgentTool[]
}

export interface AgentRun {
  run_id: string
  run_dir: string
  pid: number | null
  is_running: boolean
  metrics: Record<string, unknown>
  config_summary: {
    stage?: string
    base_model_path?: string
    output_dir?: string
    training_backend?: string
    tuning_method?: string
  }
}

export interface SystemInfo {
  gpus: GpuStats[]
  docker_running: boolean
  accelerate_configs: string[]
  deepspeed_configs: string[]
}

export interface VLAPreflightReport {
  ready: boolean
  checks: { name: string; status: 'pass' | 'warning' | 'error'; detail: string }[]
  sample: { image: string; instruction: string; state: number[]; actions: unknown[]; valid_timesteps: number; horizon: number } | null
}
export interface VLAPrediction {
  actions: unknown[]
  action_type: 'continuous' | 'discrete' | 'hybrid'
  action_dim: number
  action_horizon: number
  action_names?: string[]
  discrete_names?: string[]
  discrete_classes?: number[]
  discrete_labels?: string[][]
}

export interface WorldFoundationModel {
  id: string; name: string; provider: string; repo_id: string | null; revision: string | null;
  tasks: string[]; local_weights: boolean; training: boolean; description: string;
  requirements: string; source: string; license: string;
}
export interface WorldFoundationResult {
  model_id: string; task: string; model_path?: string; resources_path?: string;
  resolved_revision?: string; video_path?: string; previews?: string[]; frame_count?: number; fps?: number;
}

export interface LocalWorldFrame {
  frame: string; step: number; seconds?: number; action_names?: string[]; device?: string; resolution?: number[];
  session_id?: string; model_id?: string;
}

export interface WorldResourceReport {
  model_id: string; operation: string; supported: boolean; training_supported: boolean;
  status: 'unsupported' | 'blocked' | 'unknown' | 'risk' | 'likely_fit'; device: string;
  gpu: { id: number; name: string; uuid: string; total_gib: number | null; free_gib: number | null } | null;
  hardware: { cpu_name: string; cpu_threads: number; ram_total_gib: number; ram_available_gib: number;
    disk_free_gib: number | null; cache_disk_free_gib: number | null; cgroup_limited: boolean; sampled_at: number };
  parameters: number; memory_gib: [number | null, number | null]; ram_gib: [number | null, number | null];
  disk_required_gib: number | null; disk_available_gib: number | null; training_state_lower_bound_gib: number | null;
  reasons: string[]; warnings: string[]; hints: string[]; basis: string; profile_key: string;
  benchmark_supported: boolean; benchmark_running: boolean;
  measurement: { status: string; error?: string; device?: string; gpu_name?: string; measured_at?: number;
    peak_allocated_gib?: number; peak_reserved_gib?: number; peak_rss_gib?: number; train_step_seconds?: number;
    warmup_steps?: number; measured_steps?: number } | null;
  eta: { remaining_steps: number; training_seconds: number; validation_seconds: number | null;
    total_seconds_low: number; total_seconds_high: number | null; checkpoint_seconds: number;
    validation_included: boolean; basis: string; note: string } | null;
}
