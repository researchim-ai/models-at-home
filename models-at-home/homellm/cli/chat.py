#!/usr/bin/env python
"""
homellm.cli.chat
================
Простой интерактивный (или однократный) скрипт для тестирования любой модели,
сохранённой `homellm.training.pretrain` (support arch="home" или "gpt2").

Пример запуска:
python -m homellm.cli.chat --model_dir out/home_pretrain/final_model --device cuda --prompt "Привет, как дела?"
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, GenerationConfig

# Регистрируем наши классы, чтобы AutoModel их «увидел»
from homellm.models.home_model import HomeForCausalLM, HomeConfig  # noqa: F401

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description="HomeLLM chat / evaluation script")
    p.add_argument("--model_dir", type=str, required=True, help="Каталог с моделью (config.json, pytorch_model.bin)")
    p.add_argument("--tokenizer_dir", type=str, default=None, help="При необходимости иной токенизатор")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--prompt", type=str, default=None, help="Одноразовый prompt. Если не задан – interactive mode")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model_dir = Path(args.model_dir)
    tok_dir = Path(args.tokenizer_dir) if args.tokenizer_dir else model_dir

    tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).to(args.device)
    model.eval()

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def generate_once(prompt: str):
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            model.generate(**inputs, generation_config=gen_cfg, streamer=streamer)
        print("\n")

    if args.prompt:
        generate_once(args.prompt)
        sys.exit(0)

    # interactive
    print("Interactive mode (type 'exit' to quit)")
    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.strip().lower() in {"exit", "quit"}:
            break
        generate_once(prompt)


if __name__ == "__main__":
    main() 