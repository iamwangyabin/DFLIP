#!/usr/bin/env python3
"""Train DFLIP Profiler (BHEP) with single-node multi-GPU (DDP).

用法示例（单机 4 卡）：

    torchrun --nproc_per_node=4 scripts/train_profiler_ddp.py \
        --config configs/dflip_config.yaml

"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from peft import PeftModel
import yaml

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.transforms import build_train_val_transforms  # noqa: E402
from utils.seed import seed_everything  # noqa: E402
from models import create_profiler  # noqa: E402
from models.profiler import BHEPLoss  # noqa: E402
from dataset import create_profiling_dataloaders_ddp  # noqa: E402
from utils.training_funcs import (  # noqa: E402
    create_optimizer,
    create_scheduler,
    train_epoch,
    evaluate,
    save_checkpoint,
)
from utils.logger import create_logger  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train DFLIP Profiler (BHEP) with DDP")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dflip_config.yaml",
        help="Path to config file",
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint or LoRA dir")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (few steps, still DDP)")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max training steps per epoch (mainly for debug)",
    )
    # DDP 通常通过 torchrun 设置 LOCAL_RANK，这里做一个兜底
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank passed by torchrun (do not set manually)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_distributed(args):
    """Initialize torch.distributed for single-node multi-GPU."""
    # 优先从环境变量读取 LOCAL_RANK / RANK / WORLD_SIZE
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = args.local_rank
        if local_rank < 0:
            raise RuntimeError("LOCAL_RANK is not set. Please launch with torchrun.")

    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = local_rank

    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = torch.cuda.device_count()

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    return local_rank, rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    args = parse_args()
    config = load_config(args.config)

    local_rank, rank, world_size = setup_distributed(args)
    is_main_process = rank == 0

    if is_main_process:
        print(f"Running DDP training on {world_size} GPUs")

    # 设置随机种子（建议加上 rank 以避免完全相同的 shuffling，但这里主要靠 DistributedSampler）
    seed_everything(config["hardware"]["seed"] + rank)

    # 只在主进程创建 logger，避免多份日志
    logger = create_logger(config) if is_main_process else None

    if is_main_process:
        print("Building transforms...")
    train_transform, val_transform = build_train_val_transforms(config)

    if is_main_process:
        print("Creating datasets...")

    train_loader, val_loader, train_sampler, val_sampler = create_profiling_dataloaders_ddp(
        config, train_transform, val_transform, world_size=world_size, rank=rank
    )

    if is_main_process:
        print("Creating model...")

    model = create_profiler(config)

    device = torch.device("cuda", local_rank)
    model.to(device)
    # 下游代码里通过 model.device 取 device
    model.device = device

    # 包一层 DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    if is_main_process:
        print("Using BHEP Loss...")
    loss_fn = BHEPLoss(
        lambda_edl=config["bhep"].get("lambda_edl", 0.1),
        lambda_kl=config["bhep"].get("lambda_kl", 0.5),
    )

    train_config = config["training"]
    # 每个 rank 的 dataloader 长度不同于总样本，这里使用 train_loader 本身长度
    num_training_steps = (
        len(train_loader) // train_config["gradient_accumulation_steps"] * train_config["num_epochs"]
    )

    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, num_training_steps, config)

    # Resume 只在主进程负责加载，然后广播参数给其他进程
    start_epoch = 0
    if args.resume and is_main_process:
        print(f"Resuming from {args.resume}...")
        try:
            # 尝试把 resume 路径当成 LoRA adapter 目录
            # 注意：这里要访问 DDP 里真正的模型：model.module
            model.module.vision_encoder.backbone = PeftModel.from_pretrained(
                model.module.vision_encoder.backbone, args.resume
            )
            print("Loaded LoRA adapter from", args.resume)
        except Exception as e:  # noqa: BLE001
            print("Could not load LoRA adapter, trying full model state_dict.")
            checkpoint = torch.load(args.resume, map_location=device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.module.load_state_dict(checkpoint["state_dict"])
            else:
                model.module.load_state_dict(checkpoint)

    # 广播模型和优化器状态到所有 rank（即使没有 resume 也可以保证一致）
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    # 这里简化：不做 optimizer/scheduler state 的广播，如有需要可以补充

    best_val_loss = float("inf")

    for epoch in range(start_epoch, train_config["num_epochs"]):
        # 每个 epoch 之前要设置 sampler 的 epoch，保证 shuffle 一致
        train_sampler.set_epoch(epoch)

        if is_main_process:
            print("\n" + "=" * 50)
            print(f"Epoch {epoch + 1}/{train_config['num_epochs']}")
            print("=" * 50)

        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            epoch,
            config,
            args.debug,
            args.max_steps,
            logger=logger,
        )

        # 只在主进程打印/记录日志
        if is_main_process:
            print(f"\nTrain metrics (rank 0 only, not averaged across GPUs): {train_metrics}")

        # 验证和保存 checkpoint 也只在主进程执行
        if not args.debug and is_main_process:
            # 这里 evaluate 只在 rank 0 上跑一遍，一般够用；
            # 如果你希望严格在所有 GPU 上评估再聚合，可以进一步改造 evaluate 函数做 all_reduce。
            val_metrics = evaluate(model, val_loader, loss_fn)
            print(f"Val metrics: {val_metrics}")

            if logger is not None and logger.is_active():
                logger.log(
                    {
                        "val/loss": val_metrics["loss"],
                        "val/detection_loss": val_metrics["detection_loss"],
                        "val/detection_accuracy": val_metrics["detection_accuracy"],
                        "val/family_accuracy": val_metrics["family_accuracy"],
                        "val/version_accuracy": val_metrics["version_accuracy"],
                        "epoch": epoch,
                    }
                )

            is_best = val_metrics["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["loss"]
                print(f"New best model! Val loss: {best_val_loss:.4f}")

            # save_checkpoint 需要拿到真实模型（去掉 DDP 外壳）
            save_checkpoint(model.module, optimizer, scheduler, epoch, config, is_best)
        elif args.debug and is_main_process:
            # debug 模式下每个 epoch 也保存一份
            save_checkpoint(model.module, optimizer, scheduler, epoch, config, False)

    if is_main_process:
        print("\n" + "=" * 50)
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {config['training']['output_dir']}")
        print("=" * 50)

        if logger is not None and logger.is_active():
            logger.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
