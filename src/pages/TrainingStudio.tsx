import { useState, useEffect, useCallback, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Select } from '@/components/ui/select'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Progress } from '@/components/ui/progress'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Play,
  Square,
  Trash2,
  Download,
  RefreshCw,
  Cpu,
  HardDrive,
  Clock,
  TrendingDown,
  Zap,
  Database,
  Eye,
  Gauge,
  MessageSquare,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  Boxes,
  Layers,
  Terminal,
  ChevronDown,
  ChevronRight,
  RotateCw,
  FileJson,
} from 'lucide-react'
import { api } from '@/api/client'
import { useTrainingStore } from '@/stores/trainingStore'
import {
  useTrainingRuns,
  useTrainingRun,
  useStartTraining,
  useStopTraining,
  useDeleteRun,
  useContinueTraining,
  useRunLogs,
  useModels,
  useTrainedModels,
  useDownloadModel,
  useDeleteLocalModel,
  useDatasets,
  useDownloadDataset,
  useDeleteDataset,
  useDatasetPreview,
  useGpuInfo,
} from '@/api/hooks'
import { createMetricsWebSocket } from '@/api/client'
import type { TrainingRun } from '@/api/types'
import { formatNumber, formatBytes } from '@/lib/utils'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RTooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import { toast } from 'sonner'

function ConfigSidebar() {
  const { t } = useTranslation()
  const { config, updateConfig, setStage } = useTrainingStore()
  const { data: gpuData } = useGpuInfo()

  const stage = config.stage ?? 'pretrain'

  const stageOptions = [
    { value: 'pretrain', label: t('stage.pretrain') },
    { value: 'continual_pretrain', label: t('stage.continual_pretrain') },
    { value: 'sft', label: t('stage.sft') },
    { value: 'grpo', label: t('stage.grpo') },
  ]

  const precisionOptions = [
    { value: 'no', label: 'Float32 (no)' },
    { value: 'fp16', label: 'Float16' },
    { value: 'bf16', label: 'BFloat16' },
  ]

  const optimizerOptions = [
    { value: 'adamw', label: 'adamw' },
    { value: 'adamw_8bit', label: 'adamw_8bit' },
    { value: 'muon', label: 'muon' },
    { value: 'magma_adamw', label: 'magma_adamw' },
  ]

  const schedulerOptions = [
    { value: 'cosine', label: 'Cosine (with warmup)' },
    { value: 'linear', label: 'Linear (with warmup)' },
    { value: 'constant_with_warmup', label: 'Constant (with warmup)' },
    { value: 'cosine_with_restarts', label: 'Cosine with Restarts (with warmup)' },
  ]

  const tuningOptions = (config.training_backend === 'unsloth'
    ? [
        { value: 'lora', label: 'LoRA' },
        { value: 'qlora', label: 'QLoRA (4-bit + LoRA)' },
      ]
    : [
        { value: 'full', label: 'Full fine-tuning' },
        { value: 'lora', label: 'LoRA' },
        { value: 'qlora', label: 'QLoRA (4-bit + LoRA)' },
      ])

  const LORA_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'lm_head', 'embed_tokens']

  const archOptions =
    stage === 'pretrain'
      ? [
          { value: 'home', label: 'HomeModel (LLaMA-style)' },
          { value: 'gpt2', label: 'GPT-2 (Classic)' },
          { value: 'home_moe', label: 'HomeModel MoE' },
          { value: 'blueprint', label: 'Model Builder blueprint' },
          { value: 'hf', label: 'HuggingFace arch (from scratch)' },
        ]
      : [
          { value: 'hf', label: 'HuggingFace Model' },
          { value: 'checkpoint', label: 'Локальный checkpoint' },
        ]

  const tuningMethod = config.tuning_method ?? 'full'
  const isLora = tuningMethod === 'lora' || tuningMethod === 'qlora'
  const trainingMode = config.training_mode ?? (stage === 'sft' ? 'epochs' : 'steps')
  const isPretrainArch = stage === 'pretrain' && !['hf', 'blueprint'].includes(config.model_type ?? 'home')
  const toggleModule = (m: string) => {
    const cur = config.lora_target_modules ?? []
    updateConfig({
      lora_target_modules: cur.includes(m) ? cur.filter((x) => x !== m) : [...cur, m],
    })
  }

  const parallelOptions = [
    { value: 'single', label: t('parallel.single_gpu') },
    { value: 'multi_gpu', label: 'Multi-GPU (DDP)' },
    { value: 'fsdp', label: 'FSDP' },
    { value: 'deepspeed_zero2', label: 'DeepSpeed ZeRO-2' },
    { value: 'deepspeed_zero3', label: 'DeepSpeed ZeRO-3' },
  ]

  const requiresBaseModel = stage === 'sft' || stage === 'continual_pretrain' || stage === 'grpo'

  return (
    <ScrollArea className="w-80 shrink-0 border-r border-border p-4 space-y-6 h-full">
      <div className="space-y-6">
        {/* Stage */}
        <div className="space-y-2">
          <Label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {t('sidebar.training_stage')}
          </Label>
          <Select
            options={stageOptions}
            value={stage}
            onChange={(e) => setStage(e.target.value as any)}
          />
          <div className="rounded-md bg-muted/40 px-2.5 py-2 text-[11px] text-muted-foreground">
            {stage === 'pretrain' && 'Предобучение с нуля. LR~3e-4, большой warmup, seq_len.'}
            {stage === 'continual_pretrain' && 'Продолжение претрейна на новых токенах. Нужна базовая модель.'}
            {stage === 'sft' && 'Supervised fine-tuning по chat/instruct данным. Нужна базовая модель.'}
            {stage === 'grpo' && 'RL по наградной функции (GRPO/DAPO/Dr.GRPO/SDPO).'}
          </div>
        </div>

        {/* Model / Architecture */}
        <div className="space-y-2">
          <Label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {stage === 'pretrain' ? t('sidebar.architecture') : t('sidebar.base_model')}
          </Label>
          <Select
            options={archOptions}
            value={config.model_type ?? (stage === 'pretrain' ? 'home' : 'hf')}
            onChange={(e) => updateConfig({ model_type: e.target.value })}
          />
          {(config.model_type === 'hf' || requiresBaseModel) && (
            <Input
              placeholder="e.g. Qwen/Qwen2.5-0.5B"
              value={config.base_model_path ?? config.model_path ?? ''}
              onChange={(e) =>
                updateConfig({
                  base_model_path: e.target.value,
                  model_path: e.target.value,
                })
              }
            />
          )}
          {requiresBaseModel && !config.base_model_path && !config.model_path && (
            <p className="text-[11px] text-amber-500">
              Для этого этапа нужна базовая модель (HF repo или локальный путь)
            </p>
          )}
        </div>

        {stage === 'pretrain' && config.model_type === 'blueprint' && (
          <div className="space-y-2">
            <Label className="text-xs">Blueprint</Label>
            <Input value={config.model_blueprint ?? ''} placeholder="blueprints/my_model.json"
              onChange={event => updateConfig({ model_blueprint: event.target.value, model_blueprint_spec: null })} />
            <Label className="text-xs">Tokenizer</Label>
            <Input value={config.model_id ?? ''} placeholder="gpt2 or local tokenizer path"
              onChange={event => updateConfig({ model_id: event.target.value })} />
          </div>
        )}

        {/* Architecture params (pretrain from scratch) */}
        {isPretrainArch && (
          <div className="space-y-3 rounded-lg border border-border bg-muted/20 p-3">
            <Label className="text-xs font-semibold uppercase tracking-wide text-blue-500">
              {t('arch.title')}
            </Label>
            <Slider
              label={t('arch.hidden_size')}
              value={config.hidden_size ?? 512}
              onValueChange={(v) => updateConfig({ hidden_size: v })}
              min={128} max={2048} step={64}
            />
            <Slider
              label={t('arch.num_layers')}
              value={config.num_layers ?? 8}
              onValueChange={(v) => updateConfig({ num_layers: v })}
              min={2} max={32} step={1}
            />
            <Slider
              label={t('arch.n_heads')}
              value={config.n_heads ?? 8}
              onValueChange={(v) => updateConfig({ n_heads: v })}
              min={2} max={32} step={1}
            />
            {config.model_type === 'home_moe' && (
              <>
                <Slider
                  label={t('arch.num_experts')}
                  value={config.num_experts ?? 8}
                  onValueChange={(v) => updateConfig({ num_experts: v })}
                  min={2} max={32} step={1}
                />
                <Slider
                  label={t('arch.experts_per_tok')}
                  value={config.num_experts_per_tok ?? 2}
                  onValueChange={(v) => updateConfig({ num_experts_per_tok: v })}
                  min={1} max={8} step={1}
                />
              </>
            )}
          </div>
        )}

        {/* Tuning method (full / LoRA / QLoRA) */}
        <div className="space-y-3">
          <Label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {t('sidebar.tuning_method')}
          </Label>
          <Select
            options={tuningOptions}
            value={tuningMethod}
            onChange={(e) => {
              const m = e.target.value as 'full' | 'lora' | 'qlora'
              updateConfig({ tuning_method: m, use_lora: m !== 'full' })
            }}
          />
          {isLora && (
            <div className="space-y-3 rounded-lg border border-border bg-muted/20 p-3">
              <Slider
                label={t('tuning.lora_r')}
                value={config.lora_r ?? 32}
                onValueChange={(v) => updateConfig({ lora_r: v, lora_rank: v })}
                min={8} max={128} step={8}
              />
              <Slider
                label={t('tuning.lora_alpha')}
                value={config.lora_alpha ?? 32}
                onValueChange={(v) => updateConfig({ lora_alpha: v })}
                min={8} max={256} step={8}
              />
              <Slider
                label={t('tuning.lora_dropout')}
                value={config.lora_dropout ?? 0.05}
                onValueChange={(v) => updateConfig({ lora_dropout: v })}
                min={0} max={0.5} step={0.05}
                formatValue={(v) => v.toFixed(2)}
              />
              <div className="space-y-1">
                <Label className="text-xs">{t('tuning.target_modules')}</Label>
                <div className="flex flex-wrap gap-1.5">
                  {LORA_MODULES.map((m) => {
                    const active = (config.lora_target_modules ?? []).includes(m)
                    return (
                      <button
                        key={m}
                        type="button"
                        onClick={() => toggleModule(m)}
                        className={`rounded-md px-2 py-0.5 text-[10px] font-mono border transition-colors ${
                          active
                            ? 'bg-primary text-primary-foreground border-primary'
                            : 'bg-muted/40 text-muted-foreground border-border hover:bg-muted'
                        }`}
                      >
                        {m}
                      </button>
                    )
                  })}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Common Hyperparameters */}
        <div className="space-y-3">
          <Label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {t('sidebar.hyperparams')}
          </Label>

          <div className="space-y-1">
            <Label className="text-xs">Optimizer</Label>
            <Select
              options={optimizerOptions}
              value={config.optimizer ?? 'adamw'}
              onChange={(e) => updateConfig({ optimizer: e.target.value as any })}
            />
          </div>

          <Slider
            label={stage === 'grpo' ? 'Train batch size' : t('training.batch_size')}
            value={
              stage === 'grpo'
                ? config.grpo_train_batch_size ?? 2
                : config.batch_size ?? 4
            }
            onValueChange={(v) =>
              stage === 'grpo'
                ? updateConfig({ grpo_train_batch_size: v, batch_size: v })
                : updateConfig({ batch_size: v })
            }
            min={1} max={stage === 'grpo' ? 32 : 256} step={1}
          />

          <Slider
            label={t('training.gradient_accumulation')}
            value={config.gradient_accumulation ?? 8}
            onValueChange={(v) => updateConfig({ gradient_accumulation: v })}
            min={1} max={32} step={1}
          />

          <div className="text-xs text-muted-foreground">
            {t('training.effective_batch')}:{' '}
            {((stage === 'grpo' ? config.grpo_train_batch_size ?? 2 : config.batch_size ?? 4)
              * (config.gradient_accumulation ?? 8))}
          </div>

          <div className="space-y-1">
            <Label className="text-xs">{t('training.learning_rate')}</Label>
            <Input
              type="number"
              step="0.000001"
              value={config.learning_rate ?? 5e-4}
              onChange={(e) => updateConfig({ learning_rate: parseFloat(e.target.value) })}
            />
            {stage === 'grpo' && (
              <p className="text-[11px] text-muted-foreground">
                LoRA: 5e-5 • Full FT: 1e-6 — 5e-6
              </p>
            )}
            {stage === 'sft' && (
              <p className="text-[11px] text-muted-foreground">
                Обычно 1e-5 — 5e-5 для LoRA, 1e-5 — 2e-5 для full FT
              </p>
            )}
          </div>

          <div className="space-y-1">
            <Label className="text-xs">{t('training.lr_scheduler')}</Label>
            <Select
              options={schedulerOptions}
              value={config.lr_scheduler ?? 'cosine'}
              onChange={(e) => updateConfig({ lr_scheduler: e.target.value, lr_schedule: e.target.value })}
            />
          </div>

          <Slider
            label={t('training.min_lr_ratio')}
            value={config.min_lr_ratio ?? 0.0}
            onValueChange={(v) => updateConfig({ min_lr_ratio: v })}
            min={0} max={0.2} step={0.01}
            formatValue={(v) => v.toFixed(2)}
          />

          <Slider
            label={t('training.warmup_steps')}
            value={config.warmup_steps ?? 1000}
            onValueChange={(v) => updateConfig({ warmup_steps: v })}
            min={0} max={10000} step={10}
          />

          {/* Epochs vs Max Steps */}
          <div className="space-y-1">
            <Label className="text-xs">{t('training.mode')}</Label>
            <div className="grid grid-cols-2 gap-1.5">
              {(['epochs', 'steps'] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => updateConfig({ training_mode: m })}
                  className={`rounded-md px-2 py-1 text-xs border transition-colors ${
                    trainingMode === m
                      ? 'bg-primary text-primary-foreground border-primary'
                      : 'bg-muted/40 text-muted-foreground border-border hover:bg-muted'
                  }`}
                >
                  {m === 'epochs' ? t('training.mode_epochs') : t('training.mode_steps')}
                </button>
              ))}
            </div>
          </div>

          {trainingMode === 'epochs' ? (
            <Slider
              label={t('training.epochs')}
              value={config.num_epochs ?? 1}
              onValueChange={(v) => updateConfig({ num_epochs: v, epochs: v })}
              min={1} max={10} step={1}
            />
          ) : (
            <Slider
              label={t('training.max_steps')}
              value={config.max_steps ?? 10000}
              onValueChange={(v) => updateConfig({ max_steps: v })}
              min={stage === 'grpo' ? 10 : 100}
              max={stage === 'pretrain' ? 1000000 : 100000}
              step={stage === 'grpo' ? 10 : 100}
            />
          )}

          <Slider
            label={t('training.max_grad_norm')}
            value={config.max_grad_norm ?? 1.0}
            onValueChange={(v) => updateConfig({ max_grad_norm: v })}
            min={0} max={10} step={0.1}
            formatValue={(v) => v.toFixed(1)}
          />

          <Slider
            label={t('training.weight_decay')}
            value={config.weight_decay ?? 0.1}
            onValueChange={(v) => updateConfig({ weight_decay: v })}
            min={0} max={1} step={0.01}
            formatValue={(v) => v.toFixed(2)}
          />

          <div className="grid grid-cols-2 gap-2">
            <div className="space-y-1">
              <Label className="text-xs">{t('training.betas')}</Label>
              <Input
                value={config.betas ?? '0.9,0.95'}
                onChange={(e) => updateConfig({ betas: e.target.value })}
              />
            </div>
            <div className="space-y-1">
              <Label className="text-xs">{t('training.eps')}</Label>
              <Input
                type="number"
                step="1e-9"
                value={config.eps ?? 1e-8}
                onChange={(e) => updateConfig({ eps: parseFloat(e.target.value) })}
              />
            </div>
          </div>
        </div>

        {/* PRETRAIN / CONTINUAL-specific */}
        {(stage === 'pretrain' || stage === 'continual_pretrain') && (
          <div className="space-y-3 rounded-lg border border-border bg-muted/20 p-3">
            <Label className="text-xs font-semibold uppercase tracking-wide text-blue-500">
              {stage === 'pretrain' ? 'Pretrain' : 'Continual Pretrain'}
            </Label>
            <Slider
              label="Sequence length"
              value={config.seq_len ?? 2048}
              onValueChange={(v) => updateConfig({ seq_len: v })}
              min={128} max={16384} step={128}
            />
            {stage === 'continual_pretrain' && (
              <p className="text-[11px] text-muted-foreground">
                Продолжает обучение LM-головы на новых токенах. Требует путь к базовой модели выше.
              </p>
            )}
          </div>
        )}

        {/* SFT-specific */}
        {stage === 'sft' && (
          <div className="space-y-3 rounded-lg border border-border bg-muted/20 p-3">
            <Label className="text-xs font-semibold uppercase tracking-wide text-green-500">
              SFT
            </Label>
            <div className="space-y-1">
              <Label className="text-xs">Формат данных</Label>
              <Select
                options={[
                  { value: 'chat', label: '💬 Chat (список сообщений)' },
                  { value: 'instruct', label: '📝 Instruct (поля)' },
                ]}
                value={config.sft_data_format ?? 'chat'}
                onChange={(e) => updateConfig({ sft_data_format: e.target.value as any })}
              />
            </div>
            <Slider
              label="Max sequence length"
              value={config.sft_max_seq_length ?? 2048}
              onValueChange={(v) => updateConfig({ sft_max_seq_length: v })}
              min={128} max={16384} step={128}
            />
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={config.sft_packing ?? false}
                onChange={(e) => updateConfig({ sft_packing: e.target.checked })}
                className="rounded"
              />
              Packing (склейка коротких примеров)
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={config.sft_train_on_completions_only ?? false}
                onChange={(e) => updateConfig({ sft_train_on_completions_only: e.target.checked })}
                className="rounded"
              />
              Учиться только на ответах (mask user)
            </label>
            <p className="text-[11px] text-muted-foreground">
              Chat-template настраивается во вкладке «Data».
            </p>
          </div>
        )}

        {/* GRPO-specific */}
        {stage === 'grpo' && (
          <div className="space-y-3 rounded-lg border border-border bg-muted/20 p-3">
            <Label className="text-xs font-semibold uppercase tracking-wide text-purple-500">
              GRPO
            </Label>
            <div className="space-y-1">
              <Label className="text-xs">Алгоритм</Label>
              <Select
                options={[
                  { value: 'grpo', label: '⭐ GRPO (рекомендуется)' },
                  { value: 'dapo', label: 'DAPO (Dynamic sAmpling)' },
                  { value: 'dr_grpo', label: 'Dr.GRPO (улучшенный)' },
                  { value: 'sdpo', label: '🎓 SDPO (Self-Distillation)' },
                ]}
                value={config.grpo_algorithm ?? 'grpo'}
                onChange={(e) => {
                  const algo = e.target.value as any
                  const presets: Record<string, Partial<typeof config>> = {
                    grpo: { grpo_kl_weight: 0, grpo_clip_eps_high: 0.2 },
                    dapo: { grpo_kl_weight: 0, grpo_clip_eps_high: 0.28 },
                    dr_grpo: { grpo_kl_weight: 0.001, grpo_clip_eps_high: 0.2 },
                    sdpo: { grpo_kl_weight: 0.01, grpo_clip_eps_high: 0.2 },
                  }
                  updateConfig({ grpo_algorithm: algo, ...presets[algo] })
                }}
              />
            </div>
            <Slider
              label="Group size (G)"
              value={config.grpo_group_size ?? 8}
              onValueChange={(v) => updateConfig({ grpo_group_size: v })}
              min={config.training_backend === 'unsloth' ? 2 : 8} max={32} step={1}
            />
            <Slider
              label="Prompt batch size"
              value={config.grpo_prompt_batch_size ?? 8}
              onValueChange={(v) => updateConfig({ grpo_prompt_batch_size: v })}
              min={1} max={64} step={1}
            />
            <Slider
              label="Max new tokens"
              value={config.grpo_max_new_tokens ?? 1024}
              onValueChange={(v) => updateConfig({ grpo_max_new_tokens: v })}
              min={128} max={16384} step={128}
            />
            <Slider
              label="Temperature"
              value={config.grpo_temperature ?? 0.7}
              onValueChange={(v) => updateConfig({ grpo_temperature: v })}
              min={0.1} max={2.0} step={0.1}
              formatValue={(v) => v.toFixed(1)}
            />
            <Slider
              label="KL weight"
              value={config.grpo_kl_weight ?? 0}
              onValueChange={(v) => updateConfig({ grpo_kl_weight: v })}
              min={0} max={0.1} step={0.001}
              formatValue={(v) => v.toFixed(3)}
            />
            <Slider
              label="Clip ε (high)"
              value={config.grpo_clip_eps_high ?? 0.2}
              onValueChange={(v) => updateConfig({ grpo_clip_eps_high: v })}
              min={0.05} max={0.5} step={0.01}
              formatValue={(v) => v.toFixed(2)}
            />
            <Slider
              label="Clip ε (low)"
              value={config.grpo_clip_eps_low ?? 0.2}
              onValueChange={(v) => updateConfig({ grpo_clip_eps_low: v })}
              min={0.05} max={0.5} step={0.01}
              formatValue={(v) => v.toFixed(2)}
            />
            <Slider
              label="Max prompts"
              value={config.grpo_max_prompts ?? 1000}
              onValueChange={(v) => updateConfig({ grpo_max_prompts: v })}
              min={10} max={100000} step={10}
            />
            <div className="space-y-1">
              <Label className="text-xs">Reward функция</Label>
              <Select
                options={[
                  { value: 'exact_match', label: 'Exact match' },
                  { value: 'math', label: 'Math correctness' },
                  { value: 'format', label: 'Format (tags)' },
                  { value: 'custom', label: 'Custom (из датасета)' },
                ]}
                value={config.grpo_reward_fn ?? 'exact_match'}
                onChange={(e) => updateConfig({ grpo_reward_fn: e.target.value })}
              />
            </div>
          </div>
        )}

        {/* Precision & Optimizations */}
        <div className="space-y-3">
          <Label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {t('precision.mixed')}
          </Label>
          <Select
            options={precisionOptions}
            value={config.mixed_precision ?? 'bf16'}
            onChange={(e) => updateConfig({ mixed_precision: e.target.value })}
          />

          {config.mixed_precision === 'fp16' && (
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={config.fp16_pure ?? false}
                onChange={(e) => updateConfig({ fp16_pure: e.target.checked })}
                className="rounded"
              />
              {t('precision.fp16_pure')}
            </label>
          )}

          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={config.flash_attention ?? true}
              onChange={(e) => updateConfig({ flash_attention: e.target.checked, use_flash_attention: e.target.checked })}
              className="rounded"
            />
            {t('precision.flash_attention')}
          </label>

          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={config.gradient_checkpointing ?? true}
              onChange={(e) => updateConfig({ gradient_checkpointing: e.target.checked, grad_checkpoint: e.target.checked })}
              className="rounded"
            />
            {t('precision.grad_checkpoint')}
          </label>

          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={config.liger_kernel ?? true}
              onChange={(e) => updateConfig({ liger_kernel: e.target.checked, use_liger: e.target.checked })}
              className="rounded"
            />
            {t('precision.liger_kernel')}
          </label>

          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={config.liger_fused_ce ?? true}
              disabled={!(config.liger_kernel ?? true)}
              onChange={(e) => updateConfig({ liger_fused_ce: e.target.checked })}
              className="rounded disabled:opacity-40"
            />
            {t('precision.liger_fused_ce')}
          </label>
        </div>

        {/* Parallel */}
        <div className="space-y-2">
          <Label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {t('parallel.mode')}
          </Label>
          <Select
            options={parallelOptions}
            value={config.parallel_mode ?? 'single'}
            onChange={(e) => updateConfig({ parallel_mode: e.target.value })}
          />
        </div>

        {/* Output */}
        <div className="space-y-2">
          <Label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {t('save.output_dir')}
          </Label>
          <Input
            value={config.output_dir ?? 'out'}
            onChange={(e) => updateConfig({ output_dir: e.target.value })}
          />
          <Slider
            label={t('save.save_every')}
            value={config.save_every ?? 100}
            onValueChange={(v) => updateConfig({ save_every: v })}
            min={1} max={5000} step={1}
          />
          <Slider
            label={t('save.log_every')}
            value={config.log_every ?? 10}
            onValueChange={(v) => updateConfig({ log_every: v })}
            min={1} max={1000} step={1}
          />
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={config.export_on_checkpoint ?? true}
              onChange={(e) => updateConfig({ export_on_checkpoint: e.target.checked })}
              className="rounded"
            />
            {t('save.export_on_checkpoint')}
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={config.merge_lora ?? true}
              onChange={(e) => updateConfig({ merge_lora: e.target.checked })}
              className="rounded"
            />
            {t('save.merge_lora')}
          </label>
        </div>

        {/* GPU Info */}
        {gpuData?.gpus && gpuData.gpus.length > 0 && (
          <div className="space-y-2">
            <Label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              {t('sidebar.gpu_memory')}
            </Label>
            {gpuData.gpus.map((gpu) => (
              <div key={gpu.id} className="rounded-lg bg-muted/50 p-2.5 space-y-1">
                <div className="flex justify-between text-xs">
                  <span>GPU {gpu.id}</span>
                  <span>{gpu.memory_used_gb.toFixed(1)} / {gpu.memory_total_gb.toFixed(1)} GB</span>
                </div>
                <Progress value={gpu.memory_percent} />
              </div>
            ))}
          </div>
        )}
      </div>
    </ScrollArea>
  )
}

function MemoryEstimatorCard() {
  const { config } = useTrainingStore()
  const [result, setResult] = useState<Record<string, any> | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const requestVersion = useRef(0)
  useEffect(() => {
    requestVersion.current++
    setResult(null)
    setError(null)
    setLoading(false)
    return () => { requestVersion.current++ }
  }, [config])

  const handleEstimate = async () => {
    const version = ++requestVersion.current
    setLoading(true)
    setError(null)
    try {
      const r = await api.estimateMemoryDetailed(
        config as Record<string, unknown>,
        Number(config.batch_size ?? 1),
        (config.parallel_mode as string) ?? 'default',
        Math.max(1, (config.gpu_ids as number[] | undefined)?.length ?? 1),
      )
      if (version !== requestVersion.current) return
      setResult(r)
      if ('error' in r) setError(r.error as string)
    } catch (e: unknown) {
      if (version === requestVersion.current) setError(String(e))
    } finally {
      if (version === requestVersion.current) setLoading(false)
    }
  }

  const fmtGB = (v: number | undefined) =>
    typeof v === 'number' ? `${v.toFixed(2)} GB` : '—'

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between">
        <div>
          <CardTitle className="text-sm flex items-center gap-2">
            <Gauge className="h-4 w-4" />
            VRAM Memory Estimator
          </CardTitle>
          <CardDescription>Приблизительная оценка для выбранной конфигурации</CardDescription>
        </div>
        <Button size="sm" variant="outline" onClick={handleEstimate} disabled={loading || config.model_type === 'blueprint'}>
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
          Estimate
        </Button>
      </CardHeader>
      <CardContent>
        {config.model_type === 'blueprint' && (
          <p className="text-xs text-muted-foreground">Оценка VRAM для произвольного графа пока недоступна.</p>
        )}
        {error && <div className="text-xs text-destructive mb-2">{error}</div>}
        {!result && !loading && (
          <p className="text-xs text-muted-foreground">Нажми Estimate чтобы посчитать</p>
        )}
        {result && (
          <div className="grid grid-cols-4 gap-3 text-sm">
            <div className="rounded-lg bg-muted/50 p-3">
              <div className="text-xs text-muted-foreground">Total VRAM</div>
              <div className="font-mono font-bold text-lg">
                {fmtGB((result.total_gb ?? result.vram_gb) as number)}
              </div>
            </div>
            <div className="rounded-lg bg-muted/50 p-3">
              <div className="text-xs text-muted-foreground">Parameters</div>
              <div className="font-mono font-bold text-lg">
                {result.params_b
                  ? `${(result.params_b as number).toFixed(2)}B`
                  : result.total_params
                  ? formatNumber(result.total_params as number)
                  : '—'}
              </div>
            </div>
            <div className="rounded-lg bg-muted/50 p-3">
              <div className="text-xs text-muted-foreground">Weights</div>
              <div className="font-mono">{fmtGB(result.weights_gb as number)}</div>
            </div>
            <div className="rounded-lg bg-muted/50 p-3">
              <div className="text-xs text-muted-foreground">Gradients</div>
              <div className="font-mono">{fmtGB(result.grads_gb as number)}</div>
            </div>
            <div className="rounded-lg bg-muted/50 p-3">
              <div className="text-xs text-muted-foreground">Optimizer</div>
              <div className="font-mono">{fmtGB(result.optim_gb as number)}</div>
            </div>
            <div className="rounded-lg bg-muted/50 p-3">
              <div className="text-xs text-muted-foreground">Activations</div>
              <div className="font-mono">{fmtGB(result.activations_gb as number)}</div>
            </div>
            <div className="rounded-lg bg-muted/50 p-3">
              <div className="text-xs text-muted-foreground">Buffer</div>
              <div className="font-mono">{fmtGB(result.buffer_gb as number)}</div>
            </div>
            <div className="rounded-lg bg-muted/50 p-3">
              <div className="text-xs text-muted-foreground">Arch</div>
              <div className="font-mono text-xs">{(result.architecture as string) ?? '—'}</div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function ReadinessRow({ ok, label, value }: { ok: boolean; label: string; value: string }) {
  return (
    <div className="flex items-center gap-3 rounded-lg bg-muted/40 p-3">
      {ok ? (
        <CheckCircle2 className="h-5 w-5 shrink-0 text-emerald-500" />
      ) : (
        <AlertCircle className="h-5 w-5 shrink-0 text-amber-500" />
      )}
      <div className="min-w-0 flex-1">
        <div className="text-xs text-muted-foreground">{label}</div>
        <div className="truncate text-sm font-medium">{value}</div>
      </div>
    </div>
  )
}

function ModelPreviewCard() {
  const { t } = useTranslation()
  const { config } = useTrainingStore()
  const isPretrain = (config.stage ?? 'pretrain') === 'pretrain'
  const fromScratch = isPretrain && (config.model_type ?? 'home') !== 'hf'

  const hidden = config.hidden_size ?? 512
  const layers = config.num_layers ?? 8
  const heads = config.n_heads ?? 8
  const headDim = heads ? Math.round(hidden / heads) : 0

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2">
          <Layers className="h-4 w-4" />
          {t('launch.model_preview')}
        </CardTitle>
        <CardDescription>
          {fromScratch ? t('launch.arch_from_scratch') : (config.base_model_path || config.model_path || t('launch.no_base_model'))}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {isPretrain && config.model_type === 'blueprint' ? (
          <div className="rounded-lg bg-muted/50 p-3 text-sm space-y-1">
            <div className="font-mono break-all">{config.model_blueprint || 'Model Builder'}</div>
            {Array.isArray(config.model_blueprint_spec?.blocks) && (
              <div className="text-muted-foreground">{config.model_blueprint_spec.blocks.length} blocks · h={config.model_blueprint_spec.hidden_size as number}</div>
            )}
          </div>
        ) : fromScratch ? (
          <div className="grid grid-cols-4 gap-3 text-sm">
            <div className="rounded-lg bg-muted/50 p-3">
              <div className="text-xs text-muted-foreground">{t('arch.hidden_size')}</div>
              <div className="font-mono font-semibold">{hidden}</div>
            </div>
            <div className="rounded-lg bg-muted/50 p-3">
              <div className="text-xs text-muted-foreground">{t('arch.num_layers')}</div>
              <div className="font-mono font-semibold">{layers}</div>
            </div>
            <div className="rounded-lg bg-muted/50 p-3">
              <div className="text-xs text-muted-foreground">{t('arch.n_heads')}</div>
              <div className="font-mono font-semibold">{heads}</div>
            </div>
            <div className="rounded-lg bg-muted/50 p-3">
              <div className="text-xs text-muted-foreground">{t('launch.head_dim')}</div>
              <div className="font-mono font-semibold">{headDim}</div>
            </div>
          </div>
        ) : (
          <div className="rounded-lg bg-muted/50 p-3 text-sm">
            <div className="text-xs text-muted-foreground">{t('sidebar.base_model')}</div>
            <div className="font-mono text-sm break-all">
              {config.base_model_path || config.model_path || '—'}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function ConfigDumpCard() {
  const { t } = useTranslation()
  const { config } = useTrainingStore()
  const [open, setOpen] = useState(false)
  return (
    <Card>
      <CardHeader
        className="cursor-pointer flex-row items-center justify-between"
        onClick={() => setOpen((o) => !o)}
      >
        <CardTitle className="text-sm flex items-center gap-2">
          <FileJson className="h-4 w-4" />
          {t('launch.full_config')}
        </CardTitle>
        {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
      </CardHeader>
      {open && (
        <CardContent>
          <pre className="max-h-80 overflow-auto rounded-lg bg-muted p-3 text-xs font-mono">
            {JSON.stringify(config, null, 2)}
          </pre>
        </CardContent>
      )}
    </Card>
  )
}

function LaunchTab({ onNavigate }: { onNavigate: (tab: string) => void }) {
  const { t } = useTranslation()
  const { config, activeRunId, setActiveRunId } = useTrainingStore()
  const startMutation = useStartTraining()
  const stopMutation = useStopTraining()
  const { data: selectedRun } = useTrainingRun(activeRunId)

  const stage = config.stage ?? 'pretrain'
  const fromScratch = stage === 'pretrain' && (config.model_type ?? 'home') !== 'hf'

  const modelReady = config.model_type === 'blueprint' && stage === 'pretrain'
    ? !!(config.model_blueprint || config.model_blueprint_spec)
    : fromScratch || !!(config.base_model_path || config.model_path)
  const dataPath = stage === 'grpo' ? (config.grpo_dataset_path || config.dataset_path) : config.dataset_path
  const dataReady = !!dataPath
  const modeReady = !!config.parallel_mode
  const readyCount = [modelReady, dataReady, modeReady].filter(Boolean).length
  const allReady = readyCount === 3

  const modelValue = fromScratch
    ? `${t('launch.arch_from_scratch')} (${config.model_type})`
    : (config.base_model_path || config.model_path || t('launch.not_selected'))

  const handleStart = async () => {
    try {
      const result = await startMutation.mutateAsync(config as Record<string, unknown>)
      setActiveRunId(result.run_id)
      toast.success(t('status.running'))
      onNavigate('monitoring')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t('error.generic'))
    }
  }

  const handleStop = async () => {
    if (!activeRunId) return
    try {
      await stopMutation.mutateAsync(activeRunId)
      setActiveRunId(null)
      toast.success(t('status.stopped'))
    } catch {
      toast.error(t('error.generic'))
    }
  }

  const isRunning = !!activeRunId && (selectedRun?.process_running ?? !['completed', 'error', 'stopped', 'unknown'].includes(selectedRun?.status ?? 'initializing'))

  return (
    <div className="space-y-6">
      {/* Readiness summary */}
      <Card className={allReady ? 'border-emerald-500/40' : 'border-amber-500/40'}>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            {allReady ? <CheckCircle2 className="h-4 w-4 text-emerald-500" /> : <AlertCircle className="h-4 w-4 text-amber-500" />}
            {t('launch.readiness')} — {readyCount}/3
          </CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-3 gap-3">
          <ReadinessRow ok={modelReady} label={t('launch.model')} value={modelValue} />
          <ReadinessRow ok={dataReady} label={t('launch.data')} value={dataPath ? dataPath.split('/').pop()! : t('launch.not_selected')} />
          <ReadinessRow ok={modeReady} label={t('launch.training_mode')} value={config.parallel_mode ?? 'single'} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">{t('launch.config_title')}</CardTitle>
          <CardDescription>
            {t(`stage.${stage}`)} &mdash; {config.model_type ?? 'home'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div className="rounded-lg bg-muted/50 p-3">
              <div className="text-muted-foreground text-xs">{t('training.batch_size')}</div>
              <div className="font-mono font-semibold">{config.batch_size}</div>
            </div>
            <div className="rounded-lg bg-muted/50 p-3">
              <div className="text-muted-foreground text-xs">{t('training.learning_rate')}</div>
              <div className="font-mono font-semibold">{config.learning_rate}</div>
            </div>
            <div className="rounded-lg bg-muted/50 p-3">
              <div className="text-muted-foreground text-xs">
                {config.training_mode === 'epochs' ? t('training.epochs') : t('training.max_steps')}
              </div>
              <div className="font-mono font-semibold">
                {config.training_mode === 'epochs' ? (config.num_epochs ?? config.epochs) : config.max_steps}
              </div>
            </div>
            <div className="rounded-lg bg-muted/50 p-3">
              <div className="text-muted-foreground text-xs">{t('precision.mixed')}</div>
              <div className="font-mono font-semibold">{config.mixed_precision}</div>
            </div>
            <div className="rounded-lg bg-muted/50 p-3">
              <div className="text-muted-foreground text-xs">{t('training.lr_scheduler')}</div>
              <div className="font-mono font-semibold">{config.lr_scheduler}</div>
            </div>
            <div className="rounded-lg bg-muted/50 p-3">
              <div className="text-muted-foreground text-xs">{t('sidebar.tuning_method')}</div>
              <div className="font-mono font-semibold">{config.tuning_method ?? 'full'}</div>
            </div>
          </div>

          <div className="mt-6 flex items-center gap-3">
            {isRunning ? (
              <Button variant="destructive" onClick={handleStop} disabled={stopMutation.isPending}>
                <Square className="h-4 w-4" />
                {t('button.stop_training')}
              </Button>
            ) : (
              <Button onClick={handleStart} disabled={startMutation.isPending || !allReady}>
                <Play className="h-4 w-4" />
                {stage === 'grpo' ? t('button.start_grpo') : t('button.start_training')}
              </Button>
            )}
            {!isRunning && !allReady && (
              <span className="text-xs text-amber-500">{t('launch.not_ready_hint')}</span>
            )}
          </div>
        </CardContent>
      </Card>

      <ModelPreviewCard />
      <MemoryEstimatorCard />
      <ConfigDumpCard />
    </div>
  )
}

function GRPOSamplesCard({ runId }: { runId: string }) {
  const [samples, setSamples] = useState<Array<Record<string, unknown>>>([])
  const [loading, setLoading] = useState(false)
  const [expanded, setExpanded] = useState<number | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const r = await api.grpoSamples(runId)
      setSamples(r.samples ?? [])
    } catch { /* ignore */ }
    finally { setLoading(false) }
  }, [runId])

  useEffect(() => {
    load()
    const interval = setInterval(load, 5000)
    return () => clearInterval(interval)
  }, [load])

  if (samples.length === 0 && !loading) return null

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between">
        <div>
          <CardTitle className="text-sm flex items-center gap-2">
            <Sparkles className="h-4 w-4" />
            GRPO Samples
          </CardTitle>
          <CardDescription>Rollouts с наградами из текущего run</CardDescription>
        </div>
        <Button size="icon" variant="ghost" onClick={load}>
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
        </Button>
      </CardHeader>
      <CardContent>
        <div className="space-y-2 max-h-96 overflow-auto">
          {samples.map((s, i) => {
            const reward = (s.reward ?? s.mean_reward) as number | undefined
            const prompt = s.prompt as string | undefined
            const completion = (s.completion ?? s.response) as string | undefined
            const step = s.step as number | undefined
            return (
              <div
                key={i}
                className="rounded-lg border border-border p-2 text-xs cursor-pointer hover:bg-accent/30"
                onClick={() => setExpanded(expanded === i ? null : i)}
              >
                <div className="flex items-center justify-between mb-1">
                  <Badge variant="outline" className="text-[10px]">step {step ?? '—'}</Badge>
                  {reward !== undefined && (
                    <Badge variant={reward > 0 ? 'success' : 'secondary'} className="text-[10px]">
                      reward: {reward.toFixed(3)}
                    </Badge>
                  )}
                </div>
                {prompt && (
                  <div className="text-muted-foreground truncate">
                    <span className="font-semibold">Q:</span> {prompt}
                  </div>
                )}
                {completion && (
                  <div className={expanded === i ? 'whitespace-pre-wrap' : 'truncate'}>
                    <span className="font-semibold">A:</span> {completion}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}

function RunLogsCard({ runId }: { runId: string }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const { data } = useRunLogs(runId, open)

  return (
    <Card>
      <CardHeader className="cursor-pointer flex-row items-center justify-between" onClick={() => setOpen((o) => !o)}>
        <CardTitle className="text-sm flex items-center gap-2">
          <Terminal className="h-4 w-4" />
          {t('monitoring.logs')}
        </CardTitle>
        {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
      </CardHeader>
      {open && (
        <CardContent className="space-y-3">
          {data?.stderr && (
            <div>
              <div className="mb-1 text-xs font-semibold text-destructive">stderr</div>
              <pre className="max-h-48 overflow-auto rounded-lg bg-muted p-3 text-[11px] font-mono whitespace-pre-wrap">
                {data.stderr || '—'}
              </pre>
            </div>
          )}
          <div>
            <div className="mb-1 text-xs font-semibold text-muted-foreground">stdout</div>
            <pre className="max-h-64 overflow-auto rounded-lg bg-muted p-3 text-[11px] font-mono whitespace-pre-wrap">
              {data?.stdout || t('common.loading')}
            </pre>
          </div>
        </CardContent>
      )}
    </Card>
  )
}

function MonitoringTab() {
  const { t } = useTranslation()
  const { activeRunId, metricsHistory, currentMetrics, pushMetrics, hydrateMetrics } = useTrainingStore()
  const { data: runData } = useTrainingRun(activeRunId)

  useEffect(() => {
    if (!activeRunId) return

    let websocket: WebSocket | null = null
    let cancelled = false
    let reconnect: ReturnType<typeof setTimeout>
    const connect = () => {
      api.runMetrics(activeRunId).then(result => {
        if (!cancelled) hydrateMetrics(activeRunId, result.history ?? [], result.metrics ?? null)
      }).catch(() => { /* Live WebSocket can still provide the current snapshot. */ })
      createMetricsWebSocket(activeRunId, pushMetrics).then((ws) => {
        if (cancelled) { ws.close(); return }
        websocket = ws
        ws.onclose = () => { if (!cancelled) reconnect = setTimeout(connect, 2000) }
      }).catch(() => { if (!cancelled) reconnect = setTimeout(connect, 2000) })
    }
    connect()
    return () => {
      cancelled = true
      clearTimeout(reconnect)
      websocket?.close()
    }
  }, [activeRunId])

  const chartData = metricsHistory.map((m) => ({
    step: m.step,
    loss: m.loss,
    val_loss: m.val_loss,
    lr: m.learning_rate,
    grad_norm: m.grad_norm,
    reward: m.reward,
    kl: m.kl_divergence,
  }))

  const isGrpo = (runData?.stage ?? '').endsWith('grpo')
  const totalSteps = runData?.total_steps ?? 0
  const step = currentMetrics?.step ?? 0
  const progress = totalSteps > 0 ? Math.min(100, (step / totalSteps) * 100) : 0

  // ETA estimate from the last N metric timestamps
  let eta = ''
  if (metricsHistory.length >= 2 && totalSteps > step) {
    const timed = metricsHistory.filter(point => point.timestamp > 0)
    const first = timed[0]
    const last = timed[timed.length - 1]
    const dSteps = first && last ? last.step - first.step : 0
    const dTime = first && last ? last.timestamp - first.timestamp : 0
    if (dSteps > 0 && dTime > 0) {
      const perStep = dTime / dSteps
      const remaining = (totalSteps - last.step) * perStep
      const mins = Math.floor(remaining / 60)
      const hrs = Math.floor(mins / 60)
      eta = hrs > 0 ? `${hrs}ч ${mins % 60}м` : `${mins}м`
    }
  }

  const hasReward = chartData.some((d) => typeof d.reward === 'number')
  const hasVal = chartData.some((d) => typeof d.val_loss === 'number')

  if (!activeRunId) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
        {t('monitoring.select_run')}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Status + progress */}
      <Card>
        <CardContent className="p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Badge variant={runData?.status === 'running' ? 'success' : 'secondary'}>
                {t(`status.${runData?.status ?? 'initializing'}`)}
              </Badge>
              <span className="font-mono text-sm">{runData?.run_id ?? activeRunId}</span>
            </div>
            <div className="text-sm text-muted-foreground">
              {t('status.step')} {step}{totalSteps > 0 ? `/${totalSteps}` : ''}
              {eta && <span className="ml-3">ETA {eta}</span>}
            </div>
          </div>
          {totalSteps > 0 && <Progress value={progress} />}
        </CardContent>
      </Card>

      {/* Stats cards */}
      {currentMetrics && (
        <div className="grid grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2 text-muted-foreground text-xs">
                {isGrpo ? <Sparkles className="h-3.5 w-3.5" /> : <TrendingDown className="h-3.5 w-3.5" />}
                {isGrpo ? t('metrics.reward') : t('metrics.train_loss')}
              </div>
              <div className="mt-1 text-2xl font-bold font-mono">
                {isGrpo
                  ? (currentMetrics.reward?.toFixed(3) ?? '—')
                  : (currentMetrics.loss?.toFixed(4) ?? '—')}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2 text-muted-foreground text-xs">
                <Zap className="h-3.5 w-3.5" />
                {isGrpo ? t('metrics.kl') : t('training.learning_rate')}
              </div>
              <div className="mt-1 text-2xl font-bold font-mono">
                {isGrpo
                  ? (currentMetrics.kl_divergence?.toFixed(4) ?? '—')
                  : (currentMetrics.learning_rate?.toExponential(2) ?? '—')}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2 text-muted-foreground text-xs">
                <Clock className="h-3.5 w-3.5" />
                {t('status.step')}
              </div>
              <div className="mt-1 text-2xl font-bold font-mono">{currentMetrics.step}</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2 text-muted-foreground text-xs">
                <Cpu className="h-3.5 w-3.5" />
                {t('metrics.gpu_load')}
              </div>
              <div className="mt-1 text-2xl font-bold font-mono">
                {currentMetrics.gpu_stats?.[0]?.utilization ?? '—'}%
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Loss / reward chart */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">{isGrpo ? t('metrics.reward') : t('metrics.train_loss')}</CardTitle>
        </CardHeader>
        <CardContent>
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="step" tick={{ fontSize: 11 }} stroke="hsl(var(--muted-foreground))" />
                <YAxis tick={{ fontSize: 11 }} stroke="hsl(var(--muted-foreground))" />
                <RTooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                />
                <Legend />
                {isGrpo ? (
                  <Line type="monotone" dataKey="reward" name="reward" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} />
                ) : (
                  <>
                    <Line type="monotone" dataKey="loss" name="train" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} />
                    {hasVal && <Line type="monotone" dataKey="val_loss" name="val" stroke="hsl(var(--warning))" strokeWidth={1.5} dot={false} />}
                  </>
                )}
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-[300px] items-center justify-center text-muted-foreground text-sm">
              {t('common.loading')}
            </div>
          )}
        </CardContent>
      </Card>

      {/* LR + Grad norm (or KL for GRPO) chart */}
      {chartData.length > 0 && (
        <div className="grid grid-cols-2 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">{isGrpo && hasReward ? t('metrics.kl') : t('training.learning_rate')}</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="step" tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
                  <YAxis tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
                  <Line type="monotone" dataKey={isGrpo && hasReward ? 'kl' : 'lr'} stroke="hsl(var(--success))" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">{t('metrics.grad_norm')}</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="step" tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
                  <YAxis tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
                  <Line type="monotone" dataKey="grad_norm" stroke="hsl(var(--warning))" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      )}

      {/* GPU stats */}
      {currentMetrics?.gpu_stats && currentMetrics.gpu_stats.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Cpu className="h-4 w-4" />
              {t('monitoring.gpu_stats')}
            </CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-3">
            {currentMetrics.gpu_stats.map((gpu) => (
              <div key={gpu.id} className="rounded-lg bg-muted/50 p-3 space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span>GPU {gpu.id}</span>
                  <span>{gpu.memory_used_gb.toFixed(1)} / {gpu.memory_total_gb.toFixed(1)} GB</span>
                </div>
                <Progress value={gpu.memory_percent} />
                {gpu.utilization !== null && (
                  <div className="flex justify-between text-[11px] text-muted-foreground">
                    <span>{t('monitoring.util')}</span>
                    <span>{gpu.utilization}%</span>
                  </div>
                )}
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {isGrpo && <GRPOSamplesCard runId={activeRunId} />}
      <RunLogsCard runId={activeRunId} />
    </div>
  )
}

const SYSTEM_PROMPT_PRESETS: Record<string, string> = {
  'chat.sys_none': '',
  'chat.sys_assistant': 'You are a helpful assistant.',
  'chat.sys_reasoning': 'You are a helpful assistant. Think step by step before answering.',
  'chat.sys_coder': 'You are an expert programmer. Provide clean, correct, well-explained code.',
  'chat.sys_translator': 'You are a professional translator. Translate accurately and naturally.',
}

function ChatTab() {
  const { t } = useTranslation()
  const { selectedChatModel } = useTrainingStore()
  const streamRef = useRef<AbortController | null>(null)
  useEffect(() => () => streamRef.current?.abort(), [])
  const [messages, setMessages] = useState<{ role: string; content: string }[]>([])
  const [input, setInput] = useState('')
  const [modelPath, setModelPath] = useState('')
  const [backend, setBackend] = useState('transformers')
  const [isGenerating, setIsGenerating] = useState(false)
  const [availableBackends, setAvailableBackends] = useState<Array<'transformers' | 'vllm' | 'llama.cpp'>>(['transformers'])
  const [showParams, setShowParams] = useState(false)
  const [maxTokens, setMaxTokens] = useState(512)
  const [temperature, setTemperature] = useState(0.7)
  const [topP, setTopP] = useState(1.0)
  const [topK, setTopK] = useState(0)
  const [systemPreset, setSystemPreset] = useState('chat.sys_none')
  const [systemPrompt, setSystemPrompt] = useState('')
  const { data: trainedModels } = useTrainedModels()
  const { data: localModels } = useModels()

  const allModels = [
    ...(trainedModels?.models ?? []),
    ...(localModels?.models ?? []),
  ]

  useEffect(() => {
    api.chatBackends()
      .then((r) => setAvailableBackends(r.backends.length ? r.backends : ['transformers']))
      .catch(() => setAvailableBackends(['transformers']))
  }, [])

  useEffect(() => {
    if (selectedChatModel) setModelPath(selectedChatModel)
  }, [selectedChatModel])

  const handleSend = async () => {
    if (!input.trim() || !modelPath || isGenerating) return
    const userMsg = { role: 'user', content: input }
    const history = [...messages, userMsg]
    setMessages((prev) => [...prev, userMsg])
    setInput('')
    setIsGenerating(true)

    let assistantContent = ''
    setMessages((prev) => [...prev, { role: 'assistant', content: '' }])

    const outgoing = systemPrompt
      ? [{ role: 'system', content: systemPrompt }, ...history]
      : history

    const { createChatStream } = await import('@/api/client')
    streamRef.current = createChatStream(
      {
        messages: outgoing as any,
        model_path: modelPath,
        backend: backend as any,
        temperature,
        top_p: topP,
        top_k: topK,
        max_tokens: maxTokens,
        stream: true,
      },
      (text) => {
        assistantContent += text
        setMessages((prev) => {
          const copy = [...prev]
          copy[copy.length - 1] = { role: 'assistant', content: assistantContent }
          return copy
        })
      },
      () => setIsGenerating(false),
      (err) => { setIsGenerating(false); toast.error(err.message) },
    )
  }

  return (
    <div className="flex flex-col h-[calc(100vh-200px)]">
      <div className="flex gap-3 mb-3">
        <Select
          options={allModels.map((m) => ({ value: m.path, label: m.name }))}
          value={modelPath}
          onChange={(e) => setModelPath(e.target.value)}
          placeholder={t('chat.model_select')}
          className="flex-1"
        />
        <Select
          options={availableBackends.map((b) => ({
            value: b,
            label: b === 'vllm' ? 'vLLM' : b === 'llama.cpp' ? 'llama.cpp' : 'Transformers',
          }))}
          value={backend}
          onChange={(e) => setBackend(e.target.value)}
          className="w-40"
        />
        <Button variant="outline" size="icon" onClick={() => setShowParams((s) => !s)} title={t('chat.gen_params')}>
          <Gauge className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="icon" onClick={() => setMessages([])} title={t('chat.clear')}>
          <RotateCw className="h-4 w-4" />
        </Button>
      </div>

      {showParams && (
        <Card className="mb-3">
          <CardContent className="p-4 space-y-3">
            <div className="grid grid-cols-2 gap-4">
              <Slider label={t('chat.max_tokens')} value={maxTokens} onValueChange={setMaxTokens} min={16} max={4096} step={16} />
              <Slider label={t('chat.temperature')} value={temperature} onValueChange={setTemperature} min={0} max={2} step={0.05} formatValue={(v) => v.toFixed(2)} />
              <Slider label={t('chat.top_p')} value={topP} onValueChange={setTopP} min={0.1} max={1} step={0.05} formatValue={(v) => v.toFixed(2)} />
              <Slider label={t('chat.top_k')} value={topK} onValueChange={setTopK} min={0} max={100} step={1} />
            </div>
            <div className="space-y-1">
              <Label className="text-xs">{t('chat.system_prompt')}</Label>
              <Select
                options={Object.keys(SYSTEM_PROMPT_PRESETS).map((k) => ({ value: k, label: t(k) }))}
                value={systemPreset}
                onChange={(e) => {
                  const k = e.target.value
                  setSystemPreset(k)
                  setSystemPrompt(SYSTEM_PROMPT_PRESETS[k] ?? '')
                }}
              />
              <textarea
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                rows={2}
                placeholder={t('chat.system_placeholder')}
                className="w-full rounded-md border border-border bg-transparent px-3 py-2 text-xs resize-y focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              />
            </div>
          </CardContent>
        </Card>
      )}

      <ScrollArea className="flex-1 space-y-3 pr-2">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} mb-3`}
          >
            <div
              className={`max-w-[70%] rounded-2xl px-4 py-2.5 text-sm whitespace-pre-wrap ${
                msg.role === 'user'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted'
              }`}
            >
              {msg.content || (isGenerating && i === messages.length - 1 ? t('chat.thinking') : '')}
            </div>
          </div>
        ))}
      </ScrollArea>

      <div className="mt-4 flex gap-2">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
          placeholder={t('chat.message_placeholder')}
          disabled={isGenerating}
        />
        <Button onClick={handleSend} disabled={isGenerating || !modelPath}>
          {t('button.send')}
        </Button>
      </div>
    </div>
  )
}

function HistoryRunCard({ run, onNavigate }: { run: TrainingRun; onNavigate: (tab: string) => void }) {
  const { t } = useTranslation()
  const deleteMutation = useDeleteRun()
  const continueMutation = useContinueTraining()
  const { setActiveRunId, setSelectedChatModel } = useTrainingStore()
  const [expanded, setExpanded] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)

  const statusColor = (s: string) => {
    switch (s) {
      case 'running': return 'success'
      case 'completed': return 'default'
      case 'error': return 'destructive'
      default: return 'secondary'
    }
  }

  const checkpoints = run.checkpoints ?? []
  const finalModelPath = run.final_model_path
  const canResume = !run.process_running && checkpoints.length > 0 && !['completed', 'running', 'training', 'starting', 'initializing', 'building_model', 'resumed'].includes(run.status) && run.stage !== 'grpo'

  const handleMonitor = () => {
    setActiveRunId(run.run_id)
    onNavigate('monitoring')
  }
  const handleChat = () => {
    if (!finalModelPath) return
    setSelectedChatModel(finalModelPath)
    onNavigate('chat')
    toast.success(t('history.chat_selected'))
  }
  const handleResume = async () => {
    try {
      const r = await continueMutation.mutateAsync(run.run_id)
      setActiveRunId(r.run_id)
      onNavigate('monitoring')
      toast.success(t('status.running'))
    } catch {
      toast.error(t('error.generic'))
    }
  }

  return (
    <Card className="hover:border-primary/30 transition-colors">
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <button className="flex items-center gap-3 min-w-0" onClick={() => setExpanded((e) => !e)}>
            {expanded ? <ChevronDown className="h-4 w-4 shrink-0" /> : <ChevronRight className="h-4 w-4 shrink-0" />}
            <Badge variant={statusColor(run.status) as any}>{t(`status.${run.status}`)}</Badge>
            <span className="font-mono text-sm truncate">{run.experiment_name || run.run_id.slice(0, 8)}</span>
            <span className="text-xs text-muted-foreground">{t(`stage.${run.stage}`)}</span>
          </button>
          <span className="text-xs text-muted-foreground shrink-0">
            {t('status.step')} {run.current_step}/{run.total_steps}
          </span>
        </div>

        {expanded && (
          <div className="mt-4 space-y-4">
            <div className="grid grid-cols-4 gap-3 text-sm">
              <div className="rounded-lg bg-muted/50 p-3">
                <div className="text-xs text-muted-foreground">{t('status.step')}</div>
                <div className="font-mono font-semibold">{run.current_step}/{run.total_steps}</div>
              </div>
              <div className="rounded-lg bg-muted/50 p-3">
                <div className="text-xs text-muted-foreground">{t('sidebar.base_model')}</div>
                <div className="font-mono text-xs truncate">{run.model_type || '—'}</div>
              </div>
              <div className="rounded-lg bg-muted/50 p-3">
                <div className="text-xs text-muted-foreground">{t('history.checkpoints')}</div>
                <div className="font-mono font-semibold">{checkpoints.length}</div>
              </div>
              <div className="rounded-lg bg-muted/50 p-3">
                <div className="text-xs text-muted-foreground">{t('sidebar.tuning_method')}</div>
                <div className="font-mono text-xs">{run.config?.tuning_method ?? 'full'}</div>
              </div>
            </div>

            {checkpoints.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {checkpoints.map((c) => (
                  <Badge key={c} variant="outline" className="text-[10px] font-mono">{c}</Badge>
                ))}
              </div>
            )}

            <div className="flex flex-wrap items-center gap-2">
              <Button variant="outline" size="sm" onClick={handleMonitor}>
                <Eye className="h-3.5 w-3.5" />
                {t('metrics.title')}
              </Button>
              <Button variant="outline" size="sm" onClick={handleChat} disabled={!finalModelPath || run.status === 'running'}>
                <MessageSquare className="h-3.5 w-3.5" />
                {t('tabs.chat')}
              </Button>
              <Button variant="outline" size="sm" onClick={handleResume} disabled={!canResume || continueMutation.isPending}>
                <Play className="h-3.5 w-3.5" />
                {t('history.continue')}
              </Button>
              {confirmDelete ? (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground">{t('history.confirm_delete')}</span>
                  <Button variant="destructive" size="sm" onClick={() => deleteMutation.mutate(run.run_id)}>
                    {t('common.yes')}
                  </Button>
                  <Button variant="ghost" size="sm" onClick={() => setConfirmDelete(false)}>
                    {t('common.no')}
                  </Button>
                </div>
              ) : (
                <Button variant="ghost" size="sm" onClick={() => setConfirmDelete(true)}>
                  <Trash2 className="h-3.5 w-3.5" />
                  {t('common.delete')}
                </Button>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function HistoryTab({ onNavigate }: { onNavigate: (tab: string) => void }) {
  const { t } = useTranslation()
  const { data, isLoading } = useTrainingRuns()
  const runs = data?.runs ?? []

  return (
    <div className="space-y-3">
      {isLoading && <div className="text-muted-foreground text-sm">{t('common.loading')}</div>}
      {runs.length === 0 && !isLoading && (
        <div className="text-muted-foreground text-sm py-8 text-center">
          {t('history.empty')}
        </div>
      )}
      {runs.map((run: TrainingRun) => (
        <HistoryRunCard key={run.run_id} run={run} onNavigate={onNavigate} />
      ))}
    </div>
  )
}

interface DatasetPreset {
  label: string
  repo: string
  subset?: string
  split?: string
}

const DATASET_PRESETS: Record<string, DatasetPreset[]> = {
  pretrain: [
    { label: 'FineWeb-2 (Russian)', repo: 'HuggingFaceFW/fineweb-2', subset: 'rus_Cyrl', split: 'train' },
    { label: 'FineWeb-Edu', repo: 'HuggingFaceFW/fineweb-edu', subset: 'sample-10BT', split: 'train' },
    { label: 'Wikipedia (ru)', repo: 'wikimedia/wikipedia', subset: '20231101.ru', split: 'train' },
    { label: 'C4 (en)', repo: 'allenai/c4', subset: 'en', split: 'train' },
  ],
  continual_pretrain: [
    { label: 'FineWeb-2 (Russian)', repo: 'HuggingFaceFW/fineweb-2', subset: 'rus_Cyrl', split: 'train' },
    { label: 'Wikipedia (ru)', repo: 'wikimedia/wikipedia', subset: '20231101.ru', split: 'train' },
  ],
  sft: [
    { label: 'OpenOrca (ru)', repo: 'd0rj/OpenOrca-ru', split: 'train' },
    { label: 'Alpaca (cleaned)', repo: 'yahma/alpaca-cleaned', split: 'train' },
    { label: 'UltraChat 200k', repo: 'HuggingFaceH4/ultrachat_200k', split: 'train_sft' },
    { label: 'OpenHermes 2.5', repo: 'teknium/OpenHermes-2.5', split: 'train' },
  ],
  grpo: [
    { label: 'GSM8K (math)', repo: 'openai/gsm8k', subset: 'main', split: 'train' },
    { label: 'MATH', repo: 'hendrycks/competition_math', split: 'train' },
    { label: 'ru-math (reasoning)', repo: 'd0rj/gsm8k-ru', split: 'train' },
  ],
}

function DataTab() {
  const { t } = useTranslation()
  const { data: datasets, isLoading } = useDatasets()
  const downloadMutation = useDownloadDataset()
  const deleteMutation = useDeleteDataset()
  const { config, updateConfig } = useTrainingStore()
  const stage = config.stage ?? 'pretrain'
  const presets = DATASET_PRESETS[stage] ?? DATASET_PRESETS.pretrain

  const [repoId, setRepoId] = useState('')
  const [subset, setSubset] = useState('')
  const [split, setSplit] = useState('train')
  const [saveAs, setSaveAs] = useState('')
  const [limitEnabled, setLimitEnabled] = useState(true)
  const [limit, setLimit] = useState(100000)
  const [previewName, setPreviewName] = useState<string | null>(null)
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  const { data: preview } = useDatasetPreview(previewName)

  const applyPreset = (p: DatasetPreset) => {
    setRepoId(p.repo)
    setSubset(p.subset ?? '')
    setSplit(p.split ?? 'train')
    const base = p.repo.split('/').pop()!
    setSaveAs(p.subset ? `${base}-${p.subset}` : base)
  }

  const handleDownload = () => {
    if (!repoId) return
    downloadMutation.mutate(
      {
        repo_id: repoId,
        subset: subset || undefined,
        split: split || 'train',
        save_as: saveAs || repoId.split('/').pop()!,
        limit: limitEnabled ? limit : undefined,
      },
      {
        onSuccess: () => toast.success(t('success.downloaded')),
        onError: () => toast.error(t('error.generic')),
      },
    )
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">{t('data.download_hf')}</CardTitle>
          <CardDescription>{t('data.presets_for')} {t(`stage.${stage}`)}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-1.5">
            {presets.map((p) => (
              <button
                key={p.label}
                type="button"
                onClick={() => applyPreset(p)}
                className="rounded-md border border-border bg-muted/40 px-2.5 py-1 text-xs hover:bg-muted transition-colors"
              >
                {p.label}
              </button>
            ))}
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label className="text-xs">{t('data.repo_id')}</Label>
              <Input placeholder="e.g. HuggingFaceFW/fineweb-2" value={repoId} onChange={(e) => setRepoId(e.target.value)} />
            </div>
            <div className="space-y-1">
              <Label className="text-xs">{t('data.save_as')}</Label>
              <Input placeholder="my-dataset" value={saveAs} onChange={(e) => setSaveAs(e.target.value)} />
            </div>
            <div className="space-y-1">
              <Label className="text-xs">{t('data.subset')}</Label>
              <Input placeholder={t('data.subset_ph')} value={subset} onChange={(e) => setSubset(e.target.value)} />
            </div>
            <div className="space-y-1">
              <Label className="text-xs">{t('data.split')}</Label>
              <Input placeholder="train" value={split} onChange={(e) => setSplit(e.target.value)} />
            </div>
          </div>

          <div className="flex items-end gap-3">
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={limitEnabled} onChange={(e) => setLimitEnabled(e.target.checked)} className="rounded" />
              {t('data.limit_rows')}
            </label>
            <Input
              type="number"
              value={limit}
              onChange={(e) => setLimit(parseInt(e.target.value) || 0)}
              disabled={!limitEnabled}
              className="w-40"
            />
            <div className="flex-1" />
            <Button onClick={handleDownload} disabled={downloadMutation.isPending || !repoId}>
              <Download className="h-4 w-4" />
              {downloadMutation.isPending ? t('common.loading') : t('button.download')}
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">{t('data.local_datasets')}</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="text-sm text-muted-foreground">{t('common.loading')}</div>
          ) : (datasets?.datasets ?? []).length === 0 ? (
            <div className="text-sm text-muted-foreground text-center py-4">{t('data.empty')}</div>
          ) : (
            <div className="space-y-2">
              {(datasets?.datasets ?? []).map((ds) => {
                const active = config.dataset_path === ds.path
                return (
                  <div
                    key={ds.name}
                    className={`flex items-center justify-between rounded-lg p-3 transition-colors cursor-pointer ${active ? 'bg-primary/10 border border-primary/40' : 'bg-muted/50 hover:bg-muted'}`}
                    onClick={() => { updateConfig({ dataset_path: ds.path }); toast.success(t('data.selected')) }}
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <Database className="h-4 w-4 text-muted-foreground shrink-0" />
                      <span className="text-sm font-medium truncate">{ds.name}</span>
                      <Badge variant="outline" className="text-[10px]">{ds.format}</Badge>
                      {active && <Badge variant="success" className="text-[10px]">{t('data.selected_badge')}</Badge>}
                    </div>
                    <div className="flex items-center gap-3 shrink-0">
                      <span className="text-xs text-muted-foreground">{formatBytes(ds.size_bytes)}</span>
                      <Button variant="ghost" size="icon" onClick={(e) => { e.stopPropagation(); setPreviewName(ds.name) }}>
                        <Eye className="h-3.5 w-3.5" />
                      </Button>
                      {confirmDelete === ds.name ? (
                        <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                          <Button variant="destructive" size="sm" onClick={() => { deleteMutation.mutate(ds.name); setConfirmDelete(null) }}>
                            {t('common.yes')}
                          </Button>
                          <Button variant="ghost" size="sm" onClick={() => setConfirmDelete(null)}>{t('common.no')}</Button>
                        </div>
                      ) : (
                        <Button variant="ghost" size="icon" onClick={(e) => { e.stopPropagation(); setConfirmDelete(ds.name) }}>
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </CardContent>
      </Card>

      {preview && previewName && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">{t('data.preview')}: {previewName}</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="max-h-64 overflow-auto rounded-lg bg-muted p-3 text-xs font-mono">
              {JSON.stringify(preview.rows?.slice(0, 5), null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}

      <ChatTemplateCard />
    </div>
  )
}

function ChatTemplateCard() {
  const { config, updateConfig } = useTrainingStore()
  const [template, setTemplate] = useState('')
  const [loaded, setLoaded] = useState(false)
  const [loading, setLoading] = useState(false)

  const modelPath = config.base_model_path ?? ''

  const fetchTemplate = async () => {
    if (!modelPath) { toast.error('Select base model first'); return }
    setLoading(true)
    try {
      const r = await api.getChatTemplate(modelPath)
      if (r.has_template && r.template) {
        setTemplate(r.template)
        setLoaded(true)
        toast.success('Chat template loaded')
      } else {
        toast.info('No chat template in tokenizer_config.json')
      }
    } catch (e) {
      toast.error(`Failed: ${e}`)
    } finally { setLoading(false) }
  }

  const applyTemplate = () => {
    updateConfig({ sft_chat_template: template })
    toast.success('Template saved to config')
  }

  const presets = {
    'ChatML': '{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}',
    'Llama-3': '{% for message in messages %}<|start_header_id|>{{ message.role }}<|end_header_id|>\n\n{{ message.content }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}',
    'Alpaca': '{% for message in messages %}{% if message.role == "user" %}### Instruction:\n{{ message.content }}\n\n{% elif message.role == "assistant" %}### Response:\n{{ message.content }}\n\n{% endif %}{% endfor %}{% if add_generation_prompt %}### Response:\n{% endif %}',
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm flex items-center gap-2">
          <MessageSquare className="h-4 w-4" />
          SFT Chat Template
        </CardTitle>
        <CardDescription>Jinja2-шаблон для форматирования диалогов при SFT</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={fetchTemplate} disabled={loading || !modelPath}>
            {loading ? <Loader2Icon /> : <Download className="h-3.5 w-3.5" />}
            Load from model
          </Button>
          {Object.entries(presets).map(([name, tmpl]) => (
            <Button key={name} variant="outline" size="sm" onClick={() => { setTemplate(tmpl); setLoaded(true) }}>
              {name}
            </Button>
          ))}
        </div>
        <textarea
          value={template}
          onChange={(e) => setTemplate(e.target.value)}
          placeholder="{% for message in messages %}...{% endfor %}"
          rows={8}
          className="w-full rounded-md border border-border bg-transparent px-3 py-2 text-xs font-mono resize-y focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        />
        <div className="flex gap-2">
          <Button size="sm" onClick={applyTemplate} disabled={!template}>
            Apply to config
          </Button>
          {loaded && (
            <Badge variant="secondary" className="text-[10px]">
              {template.length} chars
            </Badge>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

function Loader2Icon() {
  return <RefreshCw className="h-3.5 w-3.5 animate-spin" />
}

const MODEL_PRESETS: { label: string; repo: string; save: string }[] = [
  { label: 'SmolLM2-135M', repo: 'HuggingFaceTB/SmolLM2-135M', save: 'SmolLM2-135M' },
  { label: 'SmolLM2-360M', repo: 'HuggingFaceTB/SmolLM2-360M', save: 'SmolLM2-360M' },
  { label: 'SmolLM2-1.7B', repo: 'HuggingFaceTB/SmolLM2-1.7B', save: 'SmolLM2-1.7B' },
  { label: 'Qwen2.5-0.5B', repo: 'Qwen/Qwen2.5-0.5B', save: 'Qwen2.5-0.5B' },
  { label: 'Qwen2.5-1.5B', repo: 'Qwen/Qwen2.5-1.5B', save: 'Qwen2.5-1.5B' },
  { label: 'TinyLlama-1.1B', repo: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', save: 'TinyLlama-1.1B' },
  { label: 'Pythia-160M', repo: 'EleutherAI/pythia-160m', save: 'pythia-160m' },
  { label: 'GPT-2', repo: 'openai-community/gpt2', save: 'gpt2' },
  { label: 'ruGPT3-Small', repo: 'ai-forever/rugpt3small_based_on_gpt2', save: 'rugpt3small' },
]

function ModelsTab({ onNavigate }: { onNavigate: (tab: string) => void }) {
  const { t } = useTranslation()
  const { data: localModels, isLoading: loadingLocal } = useModels()
  const { data: trainedModels, isLoading: loadingTrained } = useTrainedModels()
  const downloadMutation = useDownloadModel()
  const deleteMutation = useDeleteLocalModel()
  const { updateConfig, setSelectedChatModel } = useTrainingStore()
  const [repoId, setRepoId] = useState('')
  const [saveName, setSaveName] = useState('')
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)

  const handleDownload = () => {
    if (!repoId) return
    downloadMutation.mutate(
      { repo_id: repoId, save_name: saveName || undefined },
      {
        onSuccess: () => toast.success(t('success.downloaded')),
        onError: () => toast.error(t('error.generic')),
      },
    )
  }

  const useAsBase = (m: any) => {
    updateConfig({ base_model_path: m.path, model_path: m.path, model_type: 'hf' })
    toast.success(t('models.used_as_base'))
    onNavigate('launch')
  }

  const renderModelList = (
    models: any[] | undefined,
    loading: boolean,
    title: string,
    opts: { deletable?: boolean; chatable?: boolean },
  ) => (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="text-sm text-muted-foreground">{t('common.loading')}</div>
        ) : (
          <div className="space-y-2">
            {(models ?? []).map((m: any) => (
              <div key={m.path} className="flex items-center justify-between rounded-lg bg-muted/50 p-3">
                <div className="flex items-center gap-3 min-w-0">
                  <HardDrive className="h-4 w-4 text-muted-foreground shrink-0" />
                  <span className="text-sm font-medium truncate">{m.name}</span>
                  {m.architecture && <Badge variant="outline" className="text-[10px]">{m.architecture}</Badge>}
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <span className="text-xs text-muted-foreground">{formatBytes(m.size_bytes)}</span>
                  <Button variant="outline" size="sm" onClick={() => useAsBase(m)} title={t('models.use_as_base')}>
                    <Boxes className="h-3.5 w-3.5" />
                    {t('models.use')}
                  </Button>
                  {opts.chatable && (
                    <Button variant="ghost" size="icon" onClick={() => { setSelectedChatModel(m.path); onNavigate('chat') }} title={t('tabs.chat')}>
                      <MessageSquare className="h-3.5 w-3.5" />
                    </Button>
                  )}
                  {opts.deletable && (
                    confirmDelete === m.name ? (
                      <div className="flex items-center gap-1">
                        <Button variant="destructive" size="sm" onClick={() => { deleteMutation.mutate(m.name); setConfirmDelete(null) }}>
                          {t('common.yes')}
                        </Button>
                        <Button variant="ghost" size="sm" onClick={() => setConfirmDelete(null)}>{t('common.no')}</Button>
                      </div>
                    ) : (
                      <Button variant="ghost" size="icon" onClick={() => setConfirmDelete(m.name)}>
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    )
                  )}
                </div>
              </div>
            ))}
            {(models ?? []).length === 0 && (
              <div className="text-sm text-muted-foreground text-center py-4">
                {t('models.empty')}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">{t('models.download_hf')}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-1.5">
            {MODEL_PRESETS.map((p) => (
              <button
                key={p.label}
                type="button"
                onClick={() => { setRepoId(p.repo); setSaveName(p.save) }}
                className="rounded-md border border-border bg-muted/40 px-2.5 py-1 text-xs hover:bg-muted transition-colors"
              >
                {p.label}
              </button>
            ))}
          </div>
          <div className="flex gap-3">
            <Input
              placeholder="e.g. Qwen/Qwen2.5-0.5B"
              value={repoId}
              onChange={(e) => setRepoId(e.target.value)}
              className="flex-1"
            />
            <Input
              placeholder={t('models.save_name')}
              value={saveName}
              onChange={(e) => setSaveName(e.target.value)}
              className="w-48"
            />
            <Button onClick={handleDownload} disabled={downloadMutation.isPending || !repoId}>
              <Download className="h-4 w-4" />
              {t('button.download_model')}
            </Button>
          </div>
        </CardContent>
      </Card>

      {renderModelList(localModels?.models, loadingLocal, t('models.local_models'), { deletable: true })}
      {renderModelList(trainedModels?.models, loadingTrained, t('models.trained_models'), { chatable: true })}
    </div>
  )
}

function DocsTab() {
  const { t } = useTranslation()
  const [section, setSection] = useState('quickstart')
  return (
    <Tabs value={section} onValueChange={setSection}>
      <TabsList>
        <TabsTrigger value="quickstart">{t('docs.quickstart')}</TabsTrigger>
        <TabsTrigger value="stages">{t('docs.stages')}</TabsTrigger>
        <TabsTrigger value="grpo">GRPO</TabsTrigger>
        <TabsTrigger value="lora">LoRA</TabsTrigger>
        <TabsTrigger value="distributed">{t('docs.distributed')}</TabsTrigger>
        <TabsTrigger value="optim">{t('docs.optimizations')}</TabsTrigger>
        <TabsTrigger value="trouble">{t('docs.troubleshooting')}</TabsTrigger>
      </TabsList>

      <TabsContent value="quickstart">
        <Card><CardContent className="prose prose-invert max-w-none text-sm p-6">
          <h3>{t('docs.quickstart')}</h3>
          <ol>
            <li>{t('docs.qs_1')}</li>
            <li>{t('docs.qs_2')}</li>
            <li>{t('docs.qs_3')}</li>
            <li>{t('docs.qs_4')}</li>
            <li>{t('docs.qs_5')}</li>
            <li>{t('docs.qs_6')}</li>
          </ol>
        </CardContent></Card>
      </TabsContent>

      <TabsContent value="stages">
        <Card><CardContent className="prose prose-invert max-w-none text-sm p-6">
          <h3>{t('docs.stages')}</h3>
          <ul>
            <li><strong>Pretrain</strong> — {t('docs.stage_pretrain')}</li>
            <li><strong>Continual Pretrain</strong> — {t('docs.stage_continual')}</li>
            <li><strong>SFT</strong> — {t('docs.stage_sft')}</li>
            <li><strong>GRPO</strong> — {t('docs.stage_grpo')}</li>
          </ul>
          <h3>{t('docs.architectures')}</h3>
          <ul>
            <li><strong>Home</strong> — LLaMA-style (RoPE, SwiGLU, RMSNorm)</li>
            <li><strong>Home MoE</strong> — Mixture-of-Experts</li>
            <li><strong>GPT-2</strong> — classic baseline</li>
            <li><strong>HuggingFace</strong> — {t('docs.arch_hf')}</li>
          </ul>
        </CardContent></Card>
      </TabsContent>

      <TabsContent value="grpo">
        <Card><CardContent className="prose prose-invert max-w-none text-sm p-6">
          <h3>GRPO — Group Relative Policy Optimization</h3>
          <p>{t('docs.grpo_intro')}</p>
          <ul>
            <li><strong>GRPO</strong> — {t('docs.grpo_grpo')}</li>
            <li><strong>Dr.GRPO</strong> — {t('docs.grpo_dr')}</li>
            <li><strong>DAPO</strong> — {t('docs.grpo_dapo')}</li>
            <li><strong>SDPO</strong> — {t('docs.grpo_sdpo')}</li>
          </ul>
          <p>{t('docs.grpo_tips')}</p>
        </CardContent></Card>
      </TabsContent>

      <TabsContent value="lora">
        <Card><CardContent className="prose prose-invert max-w-none text-sm p-6">
          <h3>LoRA / QLoRA</h3>
          <p>{t('docs.lora_intro')}</p>
          <ul>
            <li><strong>rank (r)</strong> — {t('docs.lora_r')}</li>
            <li><strong>alpha</strong> — {t('docs.lora_alpha')}</li>
            <li><strong>dropout</strong> — {t('docs.lora_dropout')}</li>
            <li><strong>target modules</strong> — {t('docs.lora_modules')}</li>
          </ul>
          <p>{t('docs.lora_tip')}</p>
        </CardContent></Card>
      </TabsContent>

      <TabsContent value="distributed">
        <Card><CardContent className="prose prose-invert max-w-none text-sm p-6">
          <h3>{t('docs.distributed')}</h3>
          <ul>
            <li><strong>Single GPU</strong> — {t('docs.dist_single')}</li>
            <li><strong>Multi-GPU (DDP)</strong> — {t('docs.dist_ddp')}</li>
            <li><strong>FSDP</strong> — {t('docs.dist_fsdp')}</li>
            <li><strong>DeepSpeed ZeRO-2/3</strong> — {t('docs.dist_deepspeed')}</li>
          </ul>
        </CardContent></Card>
      </TabsContent>

      <TabsContent value="optim">
        <Card><CardContent className="prose prose-invert max-w-none text-sm p-6">
          <h3>{t('docs.optimizations')}</h3>
          <ul>
            <li><strong>Flash Attention</strong> — {t('docs.opt_flash')}</li>
            <li><strong>Gradient Checkpointing</strong> — {t('docs.opt_gc')}</li>
            <li><strong>Liger Kernel</strong> — {t('docs.opt_liger')}</li>
            <li><strong>Mixed Precision (bf16)</strong> — {t('docs.opt_bf16')}</li>
            <li><strong>8-bit optimizer</strong> — {t('docs.opt_8bit')}</li>
          </ul>
        </CardContent></Card>
      </TabsContent>

      <TabsContent value="trouble">
        <Card><CardContent className="prose prose-invert max-w-none text-sm p-6">
          <h3>{t('docs.troubleshooting')}</h3>
          <ul>
            <li><strong>OOM</strong> — {t('docs.tr_oom')}</li>
            <li><strong>{t('docs.tr_loss_spike_t')}</strong> — {t('docs.tr_loss_spike')}</li>
            <li><strong>{t('docs.tr_plateau_t')}</strong> — {t('docs.tr_plateau')}</li>
            <li><strong>{t('docs.tr_gibberish_t')}</strong> — {t('docs.tr_gibberish')}</li>
            <li><strong>{t('docs.tr_slow_t')}</strong> — {t('docs.tr_slow')}</li>
          </ul>
        </CardContent></Card>
      </TabsContent>
    </Tabs>
  )
}

export function TrainingStudio() {
  const { t } = useTranslation()
  const [activeTab, setActiveTab] = useState('launch')

  return (
    <div className="flex h-full">
      <ConfigSidebar />
      <div className="flex-1 p-6 overflow-auto">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList>
            <TabsTrigger value="launch">{t('tabs.launch')}</TabsTrigger>
            <TabsTrigger value="monitoring">{t('tabs.monitoring')}</TabsTrigger>
            <TabsTrigger value="chat">{t('tabs.chat')}</TabsTrigger>
            <TabsTrigger value="history">{t('tabs.history')}</TabsTrigger>
            <TabsTrigger value="data">{t('tabs.data')}</TabsTrigger>
            <TabsTrigger value="models">{t('tabs.models')}</TabsTrigger>
            <TabsTrigger value="docs">{t('tabs.docs')}</TabsTrigger>
          </TabsList>

          <TabsContent value="launch"><LaunchTab onNavigate={setActiveTab} /></TabsContent>
          <TabsContent value="monitoring"><MonitoringTab /></TabsContent>
          <TabsContent value="chat"><ChatTab /></TabsContent>
          <TabsContent value="history"><HistoryTab onNavigate={setActiveTab} /></TabsContent>
          <TabsContent value="data"><DataTab /></TabsContent>
          <TabsContent value="models"><ModelsTab onNavigate={setActiveTab} /></TabsContent>
          <TabsContent value="docs"><DocsTab /></TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
