"""
Qwen2.5-7B (base) 를 개인 일기 텍스트로 LoRA fine-tuning.

N=1 탐색적 실험: 단순 말투 모방을 넘어 심리적 특성/사고 패턴이
LoRA adapter 파라미터에 인코딩되는지 탐구.

실행 환경:
    A100 40GB × 8 (VESSL AI) / HuggingFace accelerate + PEFT
    bf16, gradient checkpointing, cosine LR with warmup.

데이터/토크나이저는 ``dataset.py`` 의 ``build_dataloaders`` 를 통해 구성된다.
이 파일은 모델/LoRA/옵티마이저/스케줄러/학습루프/체크포인트 저장만 담당.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
from datetime import datetime
from typing import List, Optional, Sequence

import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, set_seed
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from dataset import build_dataloaders, discover_instagram_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning Qwen2.5-7B on personal diary texts.",
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Instagram 내보내기 폴더들이 모여있는 상위 디렉터리.")
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="LoRA adapter / 토크나이저를 저장할 디렉터리. "
             "기본적으로 학습 시작 시각이 '_YYYYMMDD-HHMMSS' 형태로 자동 부착됨 "
             "(예: ./output → ./output_20260526-193200). "
             "기존 동작을 원하면 --no_timestamp_suffix.",
    )
    parser.add_argument(
        "--no_timestamp_suffix", action="store_true",
        help="output_dir 뒤에 학습 시작 시각 suffix 를 붙이지 않음.",
    )
    parser.add_argument(
        "--instagram_ids", type=str, nargs="+", default=["all"],
        help='사용할 인스타그램 ID prefix 목록 (예: "5081hjb 5082hjb"). '
             '"all" 하나만 넘기면 data_dir 안 모든 instagram-* 폴더를 자동 탐색. '
             '기본값은 "all" (= data_dir 안 모든 instagram-* 폴더).',
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B",
                        help="베이스 모델 (instruct 아님).")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="GPU 1대당 micro batch size.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="train loss 를 몇 step 마다 출력할지.")

    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (scaling = alpha / r).")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_bias", type=str, default="none",
                        choices=["none", "all", "lora_only"])
    parser.add_argument(
        "--lora_target_modules", type=str, nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="LoRA 를 부착할 nn.Linear 의 이름들.",
    )

    parser.add_argument(
        "--save_steps", type=int, default=0,
        help="N optimizer step 마다 LoRA adapter 체크포인트 저장 (0=비활성).",
    )
    parser.add_argument(
        "--save_each_epoch", action="store_true",
        help="매 epoch 종료 시점에도 체크포인트를 추가로 저장.",
    )
    parser.add_argument(
        "--save_total_limit", type=int, default=0,
        help="step 기반 체크포인트(checkpoint-*) 를 최근 N 개만 유지 (0=전부 유지).",
    )

    parser.add_argument("--use_wandb", action="store_true",
                        help="wandb 로깅 활성화.")
    parser.add_argument("--wandb_project", type=str, default="diary-personality-lora")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


def _yaml_scalar(value) -> str:
    """argparse 값 1개를 YAML scalar 표현으로 변환."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    # 특수문자/공백이 있으면 따옴표로 감싸 안전하게.
    if text == "" or any(c in text for c in ':#{}[],&*?|<>=!%@`"\'') or text != text.strip():
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return text


def _dump_args_yaml(path: str, args_dict: dict) -> None:
    """argparse 정보를 의존성 없이 YAML 파일로 저장.

    값이 list 면 YAML block sequence 로, 그 외에는 scalar 로 기록한다.
    PyYAML 이 없는 환경에서도 동작하도록 직접 직렬화한다.
    """
    lines: List[str] = []
    for key in sorted(args_dict.keys()):
        value = args_dict[key]
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                lines.append(f"{key}: []")
            else:
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {_yaml_scalar(item)}")
        else:
            lines.append(f"{key}: {_yaml_scalar(value)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _resolve_output_dir(output_dir: str, accelerator: Accelerator) -> str:
    """``output_dir`` 뒤에 학습 시작 시각 ``_YYYYMMDD-HHMMSS`` 를 붙여 반환.

    분산 학습에서 각 rank 의 wall-clock 이 살짝 다를 수 있으므로 main rank 가
    timestamp 를 만든 뒤 broadcast 하여 모든 rank 가 동일한 경로를 가진다.
    """
    holder: List[Optional[str]] = [None]
    if accelerator.is_main_process:
        holder[0] = datetime.now().strftime("%Y%m%d-%H%M%S")
    if accelerator.num_processes > 1:
        broadcast_object_list(holder, from_process=0)
    return f"{output_dir}_{holder[0]}"


def build_lora_model(
    model_name: str,
    *,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_bias: str = "none",
    lora_target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    gradient_checkpointing: bool = True,
):
    """베이스 모델 로드 후 LoRA adapter 부착.

    ``lora_target_modules`` 기본값은 Qwen2.5 의 attention projection 이름
    (``q_proj / k_proj / v_proj / o_proj``) 이다.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    if gradient_checkpointing:
        # use_reentrant=False: PEFT/accelerate 와의 호환을 위해 필요.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # gradient checkpointing 사용 시 input 에 grad 가 흐르도록 강제.
        model.enable_input_require_grads()

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(lora_target_modules),
    )
    model = get_peft_model(model, lora_config)
    return model


def _save_adapter(ckpt_dir: str, model, tokenizer, accelerator: Accelerator) -> None:
    """LoRA adapter + tokenizer 를 ``ckpt_dir`` 아래에 저장 (main process 만)."""
    if not accelerator.is_main_process:
        return
    os.makedirs(ckpt_dir, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)


def _prune_step_checkpoints(output_dir: str, limit: int) -> None:
    """``output_dir/checkpoint-{step}`` 중 최근 ``limit`` 개만 남기고 삭제."""
    if limit <= 0:
        return
    candidates: List[tuple[int, str]] = []
    for path in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        name = os.path.basename(path)
        suffix = name[len("checkpoint-"):]
        if not suffix.isdigit():
            continue
        candidates.append((int(suffix), path))
    candidates.sort(key=lambda x: x[0])
    while len(candidates) > limit:
        _, old_path = candidates.pop(0)
        shutil.rmtree(old_path, ignore_errors=True)


@torch.no_grad()
def evaluate(model, eval_loader, accelerator: Accelerator) -> float:
    """eval split 평균 loss 를 분산 환경에서 안전하게 계산."""
    model.eval()
    total_loss = torch.zeros(1, device=accelerator.device)
    total_count = torch.zeros(1, device=accelerator.device)

    for batch in eval_loader:
        outputs = model(**batch)
        # batch 마다 토큰 수가 다를 수 있으나, 단순 평균으로 충분.
        total_loss += outputs.loss.detach().float()
        total_count += 1

    gathered_loss = accelerator.reduce(total_loss, reduction="sum")
    gathered_count = accelerator.reduce(total_count, reduction="sum")

    model.train()
    if gathered_count.item() == 0:
        return float("nan")
    return (gathered_loss / gathered_count).item()


def main() -> None:
    args = parse_args()

    # 시작 시각: 프로세스 진입 시점 (셋업 포함 전체 소요시간 측정용).
    script_start = datetime.now()

    log_with: Optional[str] = "wandb" if args.use_wandb else None
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision="bf16",
        log_with=log_with,
    )

    set_seed(args.seed)

    if not args.no_timestamp_suffix:
        args.output_dir = _resolve_output_dir(args.output_dir, accelerator)

    # "all" 1 개만 넘어왔으면 data_dir 안 모든 instagram-* 폴더로 확장.
    if (
        args.instagram_ids is not None
        and len(args.instagram_ids) == 1
        and args.instagram_ids[0].lower() == "all"
    ):
        discovered = discover_instagram_ids(args.data_dir)
        if not discovered:
            raise RuntimeError(
                f"'--instagram_ids all' 이지만 {args.data_dir} 안에 "
                f"instagram-* 폴더가 없습니다."
            )
        args.instagram_ids = discovered

    if accelerator.is_main_process:
        print("=" * 60)
        print("[config]")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")
        print(f"  num_processes: {accelerator.num_processes}")
        print(f"  device: {accelerator.device}")
        print("=" * 60)

    if args.use_wandb:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.wandb_run_name}}
                if args.wandb_run_name else None,
        )

    if accelerator.is_main_process:
        print(f"[1/4] tokenizer 로드: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if accelerator.is_main_process:
        ids_for_log = args.instagram_ids if args.instagram_ids else "<default>"
        print(f"[2/4] DataLoader 구성: {args.data_dir} (ids={ids_for_log})")
    train_loader, eval_loader, train_dataset, eval_dataset = build_dataloaders(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        instagram_ids=args.instagram_ids,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
        verbose=accelerator.is_main_process,
    )

    # prepare() 로 분산 sharding 되기 전의 "원본" 길이 (extrapolation 기준).
    n_train_samples = len(train_dataset)
    n_eval_samples = len(eval_dataset)
    n_train_batches = len(train_loader)
    n_eval_batches = len(eval_loader)
    if accelerator.is_main_process:
        print(
            f"  dataset: train={n_train_samples} samples / "
            f"eval={n_eval_samples} samples"
        )
        print(
            f"  dataloader: train={n_train_batches} batches / "
            f"eval={n_eval_batches} batches (batch_size={args.batch_size})"
        )

    if accelerator.is_main_process:
        print(f"[3/4] 모델 로드 + LoRA 부착: {args.model_name}")
    model = build_lora_model(
        args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_bias=args.lora_bias,
        lora_target_modules=args.lora_target_modules,
        gradient_checkpointing=True,
    )

    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(
            f"  trainable params: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.4f}%)"
        )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 1 epoch 당 optimizer step 수 = ceil(num_batches / grad_accum_steps).
    # accelerate 가 sharding 후의 train_loader 길이를 알려주므로 prepare 이후에 계산해도 되지만,
    # prepare 전에 길이를 계산해두면 LR scheduler 도 함께 prepare 할 수 있어 단순하다.
    steps_per_epoch = math.ceil(
        len(train_loader) / (args.grad_accum_steps * accelerator.num_processes)
    )
    max_train_steps = steps_per_epoch * args.epochs

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_train_steps,
    )

    model, optimizer, train_loader, eval_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, lr_scheduler
    )

    log_file_path = os.path.join(args.output_dir, "training_log.txt")

    def log_line(message: str) -> None:
        """main process 에서만 training_log.txt 에 한 줄 append."""
        if not accelerator.is_main_process:
            return
        with open(log_file_path, "a", encoding="utf-8") as logf:
            logf.write(message + "\n")

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        # 재현용으로 args 를 함께 저장 (JSON + YAML).
        with open(os.path.join(args.output_dir, "train_args.json"), "w",
                  encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)
        _dump_args_yaml(
            os.path.join(args.output_dir, "train_args.yaml"), vars(args)
        )

        print(f"[4/4] 학습 시작 — epochs={args.epochs}, "
              f"steps/epoch≈{steps_per_epoch}, total_steps≈{max_train_steps}")

    # 중간 시각: 셋업(토크나이저/데이터/모델 로드) 완료, 실제 train loop 진입 직전.
    train_start = datetime.now()
    setup_elapsed = (train_start - script_start).total_seconds()
    log_line(
        f"# [time] script_start : {script_start.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    log_line(
        f"# [time] train_start  : {train_start.strftime('%Y-%m-%d %H:%M:%S')} "
        f"(setup took {setup_elapsed:.1f}s)"
    )
    log_line(
        f"# dataset: train={n_train_samples} samples / "
        f"eval={n_eval_samples} samples"
    )
    log_line(
        f"# dataloader: train={n_train_batches} batches / "
        f"eval={n_eval_batches} batches (batch_size={args.batch_size})"
    )
    log_line(
        f"# epochs={args.epochs}, grad_accum={args.grad_accum_steps}, "
        f"num_processes={accelerator.num_processes}, "
        f"steps/epoch≈{steps_per_epoch}, total_steps≈{max_train_steps}"
    )

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % args.logging_steps == 0:
                    # 모든 프로세스에서 loss 를 평균낸 뒤 main 에서만 출력.
                    loss_value = accelerator.gather(
                        loss.detach().float().unsqueeze(0)
                    ).mean().item()
                    current_lr = lr_scheduler.get_last_lr()[0]

                    if accelerator.is_main_process:
                        msg = (
                            f"[epoch {epoch + 1}/{args.epochs}] "
                            f"step {global_step}/{max_train_steps}  "
                            f"loss={loss_value:.4f}  lr={current_lr:.3e}"
                        )
                        print(msg, flush=True)
                        log_line(msg)

                    if args.use_wandb:
                        accelerator.log(
                            {
                                "train/loss": loss_value,
                                "train/lr": current_lr,
                                "train/epoch": epoch + 1,
                            },
                            step=global_step,
                        )

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    accelerator.wait_for_everyone()
                    ckpt_dir = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    _save_adapter(ckpt_dir, model, tokenizer, accelerator)
                    if accelerator.is_main_process:
                        _prune_step_checkpoints(
                            args.output_dir, args.save_total_limit
                        )
                        print(f"[ckpt] step {global_step} → {ckpt_dir}",
                              flush=True)

        eval_loss = evaluate(model, eval_loader, accelerator)
        if accelerator.is_main_process:
            eval_msg = (
                f"=== epoch {epoch + 1} 종료 — eval_loss={eval_loss:.4f} "
                f"(ppl={math.exp(eval_loss):.2f}) ==="
            )
            print(eval_msg, flush=True)
            log_line(eval_msg)
        if args.use_wandb:
            accelerator.log(
                {"eval/loss": eval_loss, "eval/ppl": math.exp(eval_loss)},
                step=global_step,
            )

        if args.save_each_epoch:
            accelerator.wait_for_everyone()
            ckpt_dir = os.path.join(
                args.output_dir, f"checkpoint-epoch-{epoch + 1}"
            )
            _save_adapter(ckpt_dir, model, tokenizer, accelerator)
            if accelerator.is_main_process:
                print(f"[ckpt] epoch {epoch + 1} → {ckpt_dir}", flush=True)

    accelerator.wait_for_everyone()
    _save_adapter(args.output_dir, model, tokenizer, accelerator)
    if accelerator.is_main_process:
        print(f"[done] LoRA adapter 저장 완료 → {args.output_dir}")

    # 끝 시각: 학습/저장 완료 시점 + 전체·학습 소요시간.
    script_end = datetime.now()
    train_elapsed = (script_end - train_start).total_seconds()
    total_elapsed = (script_end - script_start).total_seconds()
    log_line(
        f"# [time] train_end    : {script_end.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    log_line(
        f"# [time] train elapsed: {train_elapsed:.1f}s "
        f"({train_elapsed / 60:.2f} min)"
    )
    log_line(
        f"# [time] total elapsed: {total_elapsed:.1f}s "
        f"({total_elapsed / 60:.2f} min, incl. setup)"
    )

    if args.use_wandb:
        accelerator.end_training()


if __name__ == "__main__":
    main()


# 실행 예시 (VESSL AI / 8×A100):
# accelerate launch --num_processes 8 train.py \
#   --data_dir ./data \
#   --output_dir ./output \
#   --epochs 3
# → 실제 저장 경로는 ./output_20260526-193200 처럼 시각 suffix 가 자동으로 붙음.
#   suffix 가 싫으면 --no_timestamp_suffix 로 끌 수 있음.
#
# 단일 GPU 디버그:
# accelerate launch --num_processes 1 train.py \
#   --data_dir ./data --output_dir ./output --epochs 1 --batch_size 2
#
# data_dir 안 모든 계정 + 체크포인트 저장 + wandb:
# accelerate launch --num_processes 8 train.py \
#   --data_dir ./data --output_dir ./output --epochs 3 \
#   --instagram_ids all \
#   --save_steps 200 --save_total_limit 3 --save_each_epoch \
#   --use_wandb --wandb_run_name qwen25-7b-diary-r16
#
# LoRA 하이퍼파라미터 튜닝 예시 (MLP 까지 부착, rank 32):
# accelerate launch --num_processes 8 train.py \
#   --data_dir ./data --output_dir ./output \
#   --lora_r 32 --lora_alpha 64 --lora_dropout 0.1 \
#   --lora_target_modules q_proj k_proj v_proj o_proj \
#                         gate_proj up_proj down_proj
