#!/bin/bash
# =============================================================================
# HomeLLM - Скрипт быстрого запуска претрейна
# =============================================================================
# Использование:
#   ./scripts/run_pretrain.sh                    # Базовый запуск
#   ./scripts/run_pretrain.sh --fp16             # С mixed precision fp16
#   ./scripts/run_pretrain.sh --bf16             # С mixed precision bf16
#
# Distributed Training (FSDP / DeepSpeed):
#   DISTRIBUTED=fsdp ./scripts/run_pretrain.sh           # FSDP
#   DISTRIBUTED=deepspeed_zero2 ./scripts/run_pretrain.sh # DeepSpeed ZeRO-2
#   DISTRIBUTED=deepspeed_zero3 ./scripts/run_pretrain.sh # DeepSpeed ZeRO-3
#   DISTRIBUTED=deepspeed_zero3_offload ./scripts/run_pretrain.sh # ZeRO-3 + CPU offload
#   DISTRIBUTED=multi_gpu ./scripts/run_pretrain.sh       # Multi-GPU DDP
# =============================================================================

set -e

# Переходим в корень проекта
cd "$(dirname "$0")/.."

# Параметры по умолчанию
DATA_PATH="${DATA_PATH:-datasets/fineweb_ru_1GB.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-out/home_pretrain}"
TOKENIZER="${TOKENIZER:-gpt2}"

# Размер модели (маленькая модель для домашнего обучения)
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"
NUM_LAYERS="${NUM_LAYERS:-8}"
N_HEADS="${N_HEADS:-8}"
SEQ_LEN="${SEQ_LEN:-512}"

# Параметры обучения
BATCH_SIZE="${BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
EPOCHS="${EPOCHS:-1}"

# Логирование и сохранение
SAVE_EVERY="${SAVE_EVERY:-5000}"
LOG_EVERY="${LOG_EVERY:-100}"

# Архитектура: "home" (наша) или "gpt2" (HuggingFace GPT-2)
ARCH="${ARCH:-home}"

# Distributed training: fsdp | deepspeed_zero2 | deepspeed_zero3 | deepspeed_zero3_offload | multi_gpu | none
DISTRIBUTED="${DISTRIBUTED:-none}"

# Количество GPU (для distributed)
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"

echo "=============================================="
echo "  HomeLLM Pretrain"
echo "=============================================="
echo "Датасет:      $DATA_PATH"
echo "Выход:        $OUTPUT_DIR"
echo "Архитектура:  $ARCH"
echo "Размер:       hidden=$HIDDEN_SIZE, layers=$NUM_LAYERS, heads=$N_HEADS"
echo "Batch:        $BATCH_SIZE x $GRADIENT_ACCUMULATION (effective: $((BATCH_SIZE * GRADIENT_ACCUMULATION)))"
echo "Distributed:  $DISTRIBUTED"
if [ "$DISTRIBUTED" != "none" ]; then
    echo "Num GPUs:     $NUM_GPUS"
fi
echo "=============================================="

# Базовые аргументы для pretrain.py
TRAIN_ARGS=(
    --data_path "$DATA_PATH"
    --output_dir "$OUTPUT_DIR"
    --tokenizer_path "$TOKENIZER"
    --arch "$ARCH"
    --hidden_size "$HIDDEN_SIZE"
    --num_layers "$NUM_LAYERS"
    --n_heads "$N_HEADS"
    --seq_len "$SEQ_LEN"
    --batch_size "$BATCH_SIZE"
    --gradient_accumulation "$GRADIENT_ACCUMULATION"
    --learning_rate "$LEARNING_RATE"
    --warmup_steps "$WARMUP_STEPS"
    --epochs "$EPOCHS"
    --save_every "$SAVE_EVERY"
    --log_every "$LOG_EVERY"
)

# Запуск в зависимости от режима distributed
case "$DISTRIBUTED" in
    fsdp)
        echo "[INFO] Запуск с FSDP..."
        # Обновляем num_processes в конфиге
        sed -i "s/num_processes: .*/num_processes: $NUM_GPUS/" configs/accelerate_fsdp.yaml
        accelerate launch --config_file configs/accelerate_fsdp.yaml \
            -m homellm.training.pretrain "${TRAIN_ARGS[@]}" "$@"
        ;;
    deepspeed_zero2)
        echo "[INFO] Запуск с DeepSpeed ZeRO-2..."
        sed -i "s/num_processes: .*/num_processes: $NUM_GPUS/" configs/accelerate_deepspeed_zero2.yaml
        accelerate launch --config_file configs/accelerate_deepspeed_zero2.yaml \
            -m homellm.training.pretrain "${TRAIN_ARGS[@]}" "$@"
        ;;
    deepspeed_zero3)
        echo "[INFO] Запуск с DeepSpeed ZeRO-3..."
        sed -i "s/num_processes: .*/num_processes: $NUM_GPUS/" configs/accelerate_deepspeed_zero3.yaml
        accelerate launch --config_file configs/accelerate_deepspeed_zero3.yaml \
            -m homellm.training.pretrain "${TRAIN_ARGS[@]}" "$@"
        ;;
    deepspeed_zero3_offload)
        echo "[INFO] Запуск с DeepSpeed ZeRO-3 + CPU Offload..."
        sed -i "s/num_processes: .*/num_processes: $NUM_GPUS/" configs/accelerate_deepspeed_zero3_offload.yaml
        accelerate launch --config_file configs/accelerate_deepspeed_zero3_offload.yaml \
            -m homellm.training.pretrain "${TRAIN_ARGS[@]}" "$@"
        ;;
    multi_gpu)
        echo "[INFO] Запуск с Multi-GPU (DDP)..."
        sed -i "s/num_processes: .*/num_processes: $NUM_GPUS/" configs/accelerate_multi_gpu.yaml
        accelerate launch --config_file configs/accelerate_multi_gpu.yaml \
            -m homellm.training.pretrain "${TRAIN_ARGS[@]}" "$@"
        ;;
    none|*)
        echo "[INFO] Запуск без distributed (single GPU/CPU)..."
        python -m homellm.training.pretrain "${TRAIN_ARGS[@]}" "$@"
        ;;
esac

echo "=============================================="
echo "  Обучение завершено!"
echo "  Модель сохранена в: $OUTPUT_DIR/final_model"
echo "=============================================="
