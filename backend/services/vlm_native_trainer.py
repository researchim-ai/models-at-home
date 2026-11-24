"""Single-GPU VLM transport/training loop over homellm's own RL core.

Objective, advantages, replay filtering, overlong shaping and SDPO distillation
are imported from homellm.training.rl, never reimplemented through TRL.
"""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random


class NativeVLMTrainer:
    def __init__(self, config, metrics):
        import torch
        from homellm.training import vlm_common as common
        from homellm.training.rl.experience import ReplayBuffer
        from homellm.training.rl.loss import create_loss_function
        from backend.services.vlm_runtime import load_model, apply_adapters, runtime_types
        from backend.services.vlm_grpo import native_config
        self.config, self.metrics, self.torch = config, metrics, torch
        self.rl = native_config(config)
        common.ensure_cuda_available()
        random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        metrics.update(status='loading_model', stage='vlm_grpo', rl_engine='homellm_native',
                       native_algorithm=self.rl.algorithm.value)
        self.processor = common.load_vlm_processor(config['base_model_path'], config)
        self.model = apply_adapters(load_model(config['base_model_path'], config, 'cuda'), config)
        if config['mixed_precision'] == 'fp16':
            # GradScaler needs fp32 master gradients; autocast still runs fp16.
            for parameter in self.model.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.float()
        # Policy ratios compare eval rollouts and training forwards: disable dropout.
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        self.reference = None
        if self.rl.kl_weight > 0 and not hasattr(self.model, 'disable_adapter'):
            self.reference = load_model(config['base_model_path'], {**config, 'tuning_method': 'full'}, 'cuda')
            self.reference.requires_grad_(False).eval()
        self.loss_fn = create_loss_function(self.model, self.rl)
        self.buffer = ReplayBuffer()
        self.data = runtime_types(config)[0](config['data_path'], field_map=config.get('vlm_columns') or None)
        if not len(self.data):
            raise ValueError('GRPO dataset contains no valid examples')
        self.cursor = self.step = self.filtered = self.generated = self.teacher_forwards = 0
        self.successful = {}
        self.sidecars = {}
        self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                           lr=config['learning_rate'])
        from transformers import get_scheduler
        self.scheduler = get_scheduler(config['lr_schedule'], self.optimizer,
            num_warmup_steps=config['warmup_steps'], num_training_steps=config['max_steps'])
        self.scaler = torch.amp.GradScaler('cuda', enabled=config['mixed_precision'] == 'fp16')
        self.signature = self._signature()
        self._restore()
        self._training_mode()
        metrics.update(native_loss_class=type(self.loss_fn).__module__ + '.' + type(self.loss_fn).__name__,
            trainable_params=sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def _signature(self):
        keys = ('base_model_path', 'tuning_method', 'training_backend', 'lora_r', 'lora_alpha',
                'lora_target_modules', 'grpo_algorithm', 'grpo_group_size', 'grpo_temperature',
                'grpo_max_new_tokens', 'grpo_reward_fn', 'grpo_clip_eps_low', 'grpo_clip_eps_high',
                'grpo_kl_weight', 'seq_len', 'min_pixels', 'max_pixels', 'batch_size',
                'gradient_accumulation', 'sdpo_success_threshold', 'sdpo_alpha', 'sdpo_loss_weight',
                'grpo_overlong_buffer', 'grpo_overlong_penalty', 'grpo_max_refill_rounds', 'data_base_dir',
                'mixed_precision', 'learning_rate', 'warmup_steps', 'lr_schedule',
                'freeze_vision_tower', 'freeze_projector', 'freeze_language_model',
                'lora_dropout', 'lora_modules_to_save', 'gradient_checkpointing')
        payload = {key: self.config.get(key) for key in keys}
        for key in ('grpo_temperature', 'grpo_clip_eps_low', 'grpo_clip_eps_high', 'grpo_kl_weight',
                    'sdpo_success_threshold', 'sdpo_alpha', 'sdpo_loss_weight', 'grpo_overlong_penalty',
                    'learning_rate', 'lora_dropout'):
            if payload[key] is not None:
                payload[key] = float(payload[key])
        digest = hashlib.sha256()
        with Path(self.config['data_path']).open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        payload['dataset_sha256'] = digest.hexdigest()
        model_config = Path(self.config.get('base_model_path') or '') / 'config.json'
        if model_config.is_file():
            payload['model_config_sha256'] = hashlib.sha256(model_config.read_bytes()).hexdigest()
        self.signature_payload = payload
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _amp(self):
        dtype = self.torch.bfloat16 if self.config['mixed_precision'] == 'bf16' else self.torch.float16
        return self.torch.autocast('cuda', dtype=dtype) if self.config['mixed_precision'] != 'no' else nullcontext()

    def _training_mode(self):
        if self.config['training_backend'] == 'unsloth':
            from unsloth import FastVisionModel
            FastVisionModel.for_training(self.model, use_gradient_checkpointing=self.config['gradient_checkpointing'])
        elif self.config['gradient_checkpointing']:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.model.train()
        self.model.config.use_cache = False

    def _encode(self, messages, images):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        result = self.processor(text=[text], images=[images], return_tensors='pt', return_token_type_ids=False)
        if result['input_ids'].shape[1] + self.rl.max_new_tokens > self.config['seq_len']:
            raise ValueError('Expanded image prompt + completion budget exceeds seq_len; image tokens cannot be truncated')
        return result.to(self.model.device)

    def _log_probs(self, model, inputs, length):
        from backend.services.vlm_grpo import selected_log_probs
        with self._amp():
            return selected_log_probs(model, inputs, length)

    def _reference_log_probs(self, inputs, length):
        if self.rl.kl_weight <= 0:
            return None
        with self.torch.no_grad():
            if self.reference is not None:
                return self._log_probs(self.reference, inputs, length).detach()
            with self.model.disable_adapter():
                return self._log_probs(self.model, inputs, length).detach()

    def _collect(self):
        from homellm.training.vlm_data import load_image
        from homellm.training.rl.experience import Experience
        from homellm.training.rl.loss import compute_advantages, compute_overlong_penalty
        from homellm.training.rl.rewards.base import BinaryReward
        from backend.services.vlm_grpo import prompt_and_answer, reward_function, sequence_inputs
        torch = self.torch
        index = self.cursor % len(self.data)
        self.cursor += 1
        example = self.data[index]
        prompt, answer = prompt_and_answer(example)
        images = [load_image(ref) for ref in example['image_refs']]
        if not images or any(image is None for image in images):
            raise ValueError('GRPO image could not be loaded')
        inputs = self._encode(prompt, images)
        prompt_length = inputs['input_ids'].shape[1]
        suppressed = sorted({getattr(self.model.config, key) for key in
            ('image_token_id', 'video_token_id', 'vision_start_token_id', 'vision_end_token_id')
            if isinstance(getattr(self.model.config, key, None), int)})
        self.model.eval()
        if self.config['training_backend'] == 'unsloth':
            from unsloth import FastVisionModel
            FastVisionModel.for_inference(self.model)
        with torch.no_grad(), self._amp():
            sequences = self.model.generate(**inputs, max_new_tokens=self.rl.max_new_tokens,
                do_sample=True, temperature=self.rl.temperature, top_p=1.0, top_k=0,
                num_return_sequences=self.rl.group_size, suppress_tokens=suppressed,
                pad_token_id=self.processor.tokenizer.pad_token_id, use_cache=True)
        self._training_mode()
        self.model.eval()  # no checkpoint graph needed for frozen rollout probabilities
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        completions = []
        for sequence in sequences:
            completion = sequence[prompt_length:]
            # Keep first EOS as an action, discard padding after it, even PAD=EOS.
            for position, value in enumerate(completion.tolist()):
                if value in eos:
                    completion = completion[:position + 1]
                    break
            if not completion.numel():
                raise ValueError('Empty VLM completion has no policy actions')
            completions.append(completion)
        texts = self.processor.batch_decode(completions, skip_special_tokens=True)
        score = reward_function(self.config['grpo_reward_fn'])
        reward = BinaryReward(lambda completion, reference: bool(score([completion], [reference])[0]))
        rewards = reward.batch_call(texts, [answer] * len(texts)).to(self.model.device)
        if self.rl.algorithm.value == 'dapo':
            lengths = torch.tensor([len(c) for c in completions], device=rewards.device)
            rewards += compute_overlong_penalty(lengths, self.rl.max_new_tokens,
                                                self.rl.overlong_buffer, self.rl.overlong_penalty)
        advantages = compute_advantages(rewards, use_std_normalization=self.rl.use_std_normalization)
        if self.config['grpo_algorithm'] == 'sdpo':
            candidates = [(float(reward), text) for reward, text in zip(rewards, texts) if reward >= self.rl.sdpo_success_threshold]
            if candidates:
                self.successful[index] = max(candidates, key=lambda item: item[0])[1]
        items = []
        uid = self.generated
        for completion, text, reward, advantage in zip(completions, texts, rewards, advantages):
            seq = torch.cat([inputs['input_ids'][0], completion]).unsqueeze(0)
            model_inputs = sequence_inputs(inputs, seq)
            with torch.no_grad():
                old = self._log_probs(self.model, model_inputs, len(completion))
                ref = self._reference_log_probs(model_inputs, len(completion))
            items.append(Experience(sequences=seq[0].cpu(), prompt_length=prompt_length,
                action_log_probs=old[0].cpu(), log_probs_ref=ref[0].cpu() if ref is not None else None,
                returns=reward.detach().cpu().view(1), advantages=advantage.detach().cpu().view(1),
                action_mask=torch.ones(len(completion)), attention_mask=torch.ones(seq.shape[1]),
                prompt_id=uid, completion_text=text))
            self.metrics.log_sample_output('VLM native rollout', text, float(reward))
        self.generated += 1
        added = self.buffer.append_group(items, uid, filter_zero_gradient=self.rl.dynamic_sampling)
        if added:
            self.sidecars[uid] = {'inputs': {key: value.cpu() for key, value in inputs.items()},
                                  'index': index, 'prompt': prompt, 'refs': example['image_refs']}
        else:
            self.filtered += 1
        return added, rewards.detach().cpu()

    def _teacher(self, experience, completion):
        from homellm.training.vlm_data import load_image
        from backend.services.vlm_grpo import teacher_messages, sequence_inputs
        sidecar = self.sidecars[experience.prompt_id]
        solution = self.successful.get(sidecar['index'])
        if solution is None:
            return None
        images = [load_image(ref) for ref in sidecar['refs']]
        if any(image is None for image in images):
            raise ValueError('SDPO teacher image failed to load')
        inputs = self._encode(teacher_messages(sidecar['prompt'], solution), images)
        sequence = self.torch.cat([inputs['input_ids'], completion], dim=-1)
        self.teacher_forwards += 1
        with self.torch.no_grad():
            return self._log_probs(self.model, sequence_inputs(inputs, sequence), completion.shape[1]).detach()

    def _update(self):
        from backend.services.vlm_grpo import sequence_inputs
        torch = self.torch
        self._training_mode()
        self.optimizer.zero_grad(set_to_none=True)
        items = self.buffer.items
        token_count = sum(item.action_mask.sum().item() for item in items)
        losses, kls, distills = [], [], []
        for experience in items:
            sidecar = self.sidecars[experience.prompt_id]
            inputs = {key: value.to(self.model.device) for key, value in sidecar['inputs'].items()}
            sequence = experience.sequences.to(self.model.device).unsqueeze(0)
            completion = sequence[:, experience.prompt_length:]
            inputs = sequence_inputs(inputs, sequence)
            # Every model forward, including teacher/reference, carries the vision tensors.
            batch = experience.to(self.model.device)
            batch.action_log_probs = batch.action_log_probs.unsqueeze(0)
            batch.log_probs_ref = batch.log_probs_ref.unsqueeze(0) if batch.log_probs_ref is not None else None
            batch.advantages = batch.advantages.view(1, 1)
            batch.action_mask = batch.action_mask.unsqueeze(0)
            teacher = self._teacher(experience, completion) if self.config['grpo_algorithm'] == 'sdpo' else None
            logps = self._log_probs(self.model, inputs, completion.shape[1])
            if self.config['grpo_algorithm'] == 'sdpo':
                loss, values = self.loss_fn(logps, batch, teacher_log_probs=teacher,
                    distillation_mask=torch.ones(1, device=self.model.device) if teacher is not None else None)
            else:
                loss, values = self.loss_fn(logps, batch)
            # Exact native reduction over the entire accumulated update, including
            # variable-length completions (DAPO token-level vs sample-level GRPO).
            weight = completion.shape[1] / token_count if self.rl.token_level_loss else 1 / len(items)
            if not torch.isfinite(loss):
                raise ValueError('Native VLM policy loss is not finite')
            self.scaler.scale(loss * weight).backward()
            losses.append(float(loss.detach()) * weight)
            kls.append(float(values.get('kl_mean', 0)) * weight)
            distills.append(float(values.get('sdpo_distill_loss', 0)) * weight)
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.0)
        if not torch.isfinite(norm):
            raise ValueError('Native VLM policy gradient is not finite')
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        return sum(losses), sum(kls), float(norm), sum(distills)

    def _save(self, directory, *, state):
        directory.mkdir(parents=True, exist_ok=False)
        self.model.save_pretrained(directory)
        self.processor.save_pretrained(directory)
        if state:
            self.torch.save({'engine': 'homellm_native_v1', 'signature': self.signature,
                'signature_payload': self.signature_payload,
                'step': self.step, 'cursor': self.cursor, 'successful': self.successful,
                'filtered': self.filtered, 'generated': self.generated,
                'teacher_forwards': self.teacher_forwards,
                'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(),
                'scaler': self.scaler.state_dict(), 'python_rng': random.getstate(),
                'torch_rng': self.torch.get_rng_state(), 'cuda_rng': self.torch.cuda.get_rng_state_all(),
                }, directory / 'native_state.pt')
            (directory / 'native_manifest.json').write_text(json.dumps({'engine': 'homellm_native_v1',
                'step': self.step, 'signature': self.signature}))
        self.metrics.log_checkpoint(str(directory), step=self.step)

    def _restore(self):
        checkpoint = self.config.get('resume_from_checkpoint')
        if not checkpoint:
            return
        checkpoint = Path(checkpoint)
        if not (checkpoint / 'native_state.pt').is_file():
            raise ValueError('Checkpoint is not a native VLM GRPO optimizer checkpoint; TRL/legacy resume is incompatible')
        state = self.torch.load(checkpoint / 'native_state.pt', map_location='cpu', weights_only=False)
        if state['engine'] != 'homellm_native_v1' or state['signature'] != self.signature:
            differences = {key: (value, self.signature_payload.get(key)) for key, value in state.get('signature_payload', {}).items()
                           if value != self.signature_payload.get(key)}
            raise ValueError(f'Native GRPO checkpoint model/data/objective settings differ from this run: {differences}')
        if hasattr(self.model, 'peft_config'):
            from peft import set_peft_model_state_dict
            from peft.utils.save_and_load import load_peft_weights
            set_peft_model_state_dict(self.model, load_peft_weights(str(checkpoint)))
        else:
            from safetensors.torch import load_file
            weights = {}
            for path in checkpoint.glob('*.safetensors'):
                weights.update(load_file(str(path)))
            self.model.load_state_dict(weights, strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        # The newly requested plan may extend the original scheduler horizon.
        if hasattr(self.scheduler, 'lr_lambdas'):
            for group, base, fn in zip(self.optimizer.param_groups, self.scheduler.base_lrs, self.scheduler.lr_lambdas):
                group['lr'] = base * fn(state['step'])
        self.scaler.load_state_dict(state['scaler'])
        self.step, self.cursor = state['step'], state['cursor']
        if self.step >= self.config['max_steps']:
            raise ValueError('Checkpoint has reached max_steps; provide a larger native training plan')
        self.successful = state['successful']
        self.filtered, self.generated = state['filtered'], state['generated']
        self.teacher_forwards = state.get('teacher_forwards', 0)
        random.setstate(state['python_rng'])
        self.torch.set_rng_state(state['torch_rng'])
        self.torch.cuda.set_rng_state_all(state['cuda_rng'])

    def train(self):
        from homellm.training.vlm_common import mark_training_completed
        target = self.config['max_steps']
        self.metrics.update(status='training', total_steps=target, current_step=self.step)
        while self.step < target:
            self.buffer.clear()
            self.sidecars.clear()
            rewards = []
            required = self.config['batch_size'] * self.config['gradient_accumulation']
            groups = attempts = 0
            while groups < required and attempts < required * (1 + self.rl.max_refill_rounds):
                added, values = self._collect()
                rewards.append(values)
                groups += int(added)
                attempts += 1
            if groups < required:
                raise ValueError('Native dynamic sampling exhausted refill rounds: insufficient nonconstant reward groups; revise reward or dataset')
            loss, kl, norm, distill = self._update()
            self.step += 1
            all_rewards = self.torch.stack(rewards)
            std = float(all_rewards.std(dim=-1).mean())
            warning = 'Во всех группах одинаковые награды: сигнал от reward равен нулю.' if std == 0 else None
            self.metrics.log_step(self.step, loss, self.optimizer.param_groups[0]['lr'], reward=float(all_rewards.mean()))
            for key, value in [('grad_norm_history', norm), ('kl_history', kl),
                               ('reward_std_history', std), ('sdpo_distill_history', distill)]:
                self.metrics.metrics.setdefault(key, []).append(value)
            self.metrics.update(current_kl=kl, training_warning=warning, filtered_groups=self.filtered,
                generated_groups=self.generated, sdpo_teacher_forwards=self.teacher_forwards,
                optimizer_step=self.step, data_cursor=self.cursor)
            if self.step % self.config['save_every'] == 0:
                self._save(Path(self.config['output_dir']) / f'checkpoint-{self.step}', state=True)
        final = Path(self.config['output_dir']) / 'final_model'
        self._save(final, state=False)
        mark_training_completed(self.metrics, str(final))
