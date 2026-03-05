#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from baselines.models.llama import LlamaBlock, RMSNorm

logger = logging.getLogger(__name__)


@dataclass
class LlamaSSTConfig:
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1376
    max_seq_len: int = 128
    vocab_size: int = 50257
    num_classes: int = 2
    dropout: float = 0.0
    use_flash: bool = True


class LlamaForSequenceClassification(nn.Module):
    def __init__(self, cfg: LlamaSSTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(
            [
                LlamaBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    d_ff=cfg.d_ff,
                    max_seq_len=cfg.max_seq_len,
                    dropout=cfg.dropout,
                    use_flash=cfg.use_flash,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.final_norm = RMSNorm(cfg.d_model)
        self.classifier = nn.Linear(cfg.d_model, cfg.num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _pool_last_token(
        self,
        x: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        if attention_mask is None:
            return x[:, -1]
        token_counts = attention_mask.long().sum(dim=1)
        last_idx = (token_counts - 1).clamp(min=0)
        batch_idx = torch.arange(x.size(0), device=x.device)
        return x[batch_idx, last_idx]

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        pooled = self._pool_last_token(x, attention_mask)
        return self.classifier(pooled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LLaMA on SST-2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--cuda-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greenctx-enabled", action="store_true")
    parser.add_argument(
        "--greenctx-backend",
        type=str,
        choices=["auto", "cpp", "torch_native", "off"],
        default="auto",
    )
    parser.add_argument("--greenctx-trace", type=str, default=None)
    parser.add_argument("--greenctx-num-partitions", type=int, default=1)
    parser.add_argument("--greenctx-partition-idx", type=int, default=0)
    parser.add_argument(
        "--max-sm-count",
        type=int,
        default=48,
        help="Max SM count for green context",
    )
    parser.add_argument("--gemm-log", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/llama_sst_single",
    )
    return parser.parse_args()


def set_seed(seed: int, cuda_id: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(cuda_id)


def cosine_lr(
    step_idx: int,
    max_iters: int,
    warmup_iters: int,
    base_lr: float,
    min_lr: float,
) -> float:
    if step_idx < warmup_iters:
        return base_lr * float(step_idx + 1) / float(max(1, warmup_iters))
    if step_idx >= max_iters:
        return min_lr
    denom = max(1, max_iters - warmup_iters)
    ratio = float(step_idx - warmup_iters) / float(denom)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (base_lr - min_lr)


def build_dataloaders(
    batch_size: int,
    max_seq_len: int,
    use_cuda: bool,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("glue", "sst2")

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, list[int]]:
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_len,
        )

    tokenized = dataset.map(tokenize_batch, batched=True)
    removable = ["sentence", "idx"]
    columns_to_remove = [
        c for c in removable if c in tokenized["train"].column_names
    ]
    if columns_to_remove:
        tokenized = tokenized.remove_columns(columns_to_remove)

    columns = ["input_ids", "attention_mask", "label"]
    tokenized.set_format(type="torch", columns=columns)

    train_loader = DataLoader(
        tokenized["train"],
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        tokenized["validation"],
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_cuda,
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(logits, labels)
        total_loss += float(loss.item()) * int(labels.size(0))
        total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
        total_count += int(labels.size(0))

    model.train()
    if total_count == 0:
        return 0.0, 0.0
    return total_loss / float(total_count), total_correct / float(total_count)


def maybe_init_greenctx(args: argparse.Namespace) -> Any | None:
    if not args.greenctx_enabled:
        return None
    try:
        from morphling.runtime.green_context import (
            GreenContextConfig,
            GreenContextController,
        )
    except Exception as exc:
        logger.warning("Green context import failed: %s", exc)
        return None

    cfg = GreenContextConfig(
        enabled=True,
        backend=args.greenctx_backend,
        trace_path=args.greenctx_trace,
        clock_mode="step",
        strict=False,
        switch_sync="event_chain",
        num_partitions=args.greenctx_num_partitions,
        partition_idx=args.greenctx_partition_idx,
        stream_priority=-1,
    )

    try:
        controller = GreenContextController.from_config(
            device_id=args.cuda_id,
            cfg=cfg,
        )
    except Exception as exc:
        logger.warning("Green context init failed: %s", exc)
        return None

    if not controller.is_supported:
        backend = getattr(controller, "backend", None)
        reason = ""
        if backend is not None and hasattr(backend, "unsupported_reason"):
            reason = str(backend.unsupported_reason())
        suffix = f" ({reason})" if reason else ""
        logger.warning(
            "Green context backend unsupported%s; using default streams",
            suffix,
        )
    else:
        avail = sorted(
            int(s)
            for s in controller.available_sm_counts()
        )
        over = [
            s for s in avail
            if s > args.max_sm_count
        ]
        if over:
            logger.info(
                "greenctx SM counts above max=%d: %s",
                args.max_sm_count,
                over,
            )

    return controller


@contextmanager
def clamped_step_scope(
    controller,
    step_idx: int,
    max_sm: int,
):
    """Wrap step_scope; log SM counts above max."""
    with controller.step_scope(step_idx) as bundle:
        if bundle.sm_count > max_sm:
            logger.debug(
                "step=%d sm_count=%d exceeds max=%d",
                step_idx,
                bundle.sm_count,
                max_sm,
            )
        yield bundle


def maybe_configure_gemm(args: argparse.Namespace, script_path: Path) -> None:
    if not args.gemm_log:
        return

    gemm_log_path = str(Path(args.gemm_log).expanduser())
    os.environ["GEMM_LOG_PATH"] = gemm_log_path

    lib_path = script_path.parent / "gemm_intercept" / "libgemm_intercept.so"
    preload = os.environ.get("LD_PRELOAD", "")

    if not lib_path.exists():
        logger.warning(
            "GEMM intercept library not found at %s; "
            "build it with `make -C %s`",
            lib_path,
            lib_path.parent,
        )

    preload_parts = [p for p in preload.split(":") if p]
    if str(lib_path) not in preload_parts:
        logger.warning(
            "LD_PRELOAD does not include %s. GEMM interception will be off.",
            lib_path,
        )

    logger.info(
        "To enable GEMM interception, run with: "
        "GEMM_LOG_PATH=%s LD_PRELOAD=%s python3 %s ...",
        gemm_log_path,
        lib_path,
        script_path,
    )


def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device(f"cuda:{args.cuda_id}")
        torch.cuda.set_device(args.cuda_id)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        logger.warning("CUDA not available; falling back to CPU mode")
        device = torch.device("cpu")

    set_seed(args.seed, args.cuda_id)

    cfg = LlamaSSTConfig()
    train_loader, val_loader = build_dataloaders(
        batch_size=args.batch_size,
        max_seq_len=cfg.max_seq_len,
        use_cuda=use_cuda,
    )

    model = LlamaForSequenceClassification(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    use_amp = use_cuda
    scaler = GradScaler(enabled=use_amp)
    warmup_iters = max(1, args.max_iters // 10)
    min_lr = args.lr * 0.1

    greenctx = maybe_init_greenctx(args) if use_cuda else None
    greenctx_supported = bool(greenctx is not None and greenctx.is_supported)
    if args.greenctx_enabled and not greenctx_supported:
        logger.warning(
            "Green context requested but unavailable; using default streams"
        )

    step_metrics: list[dict[str, float | int]] = []
    step_boundaries: list[dict[str, int]] = []
    best_val_acc = -1.0
    train_iter = iter(train_loader)
    total_start = time.perf_counter()

    try:
        for step_idx in range(args.max_iters):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(
                device,
                non_blocking=True,
            )
            labels = batch["label"].to(device, non_blocking=True)

            lr = cosine_lr(
                step_idx,
                args.max_iters,
                warmup_iters,
                args.lr,
                min_lr,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            step_start = time.perf_counter()
            step_start_ns = time.clock_gettime_ns(
                time.CLOCK_MONOTONIC
            )

            if greenctx_supported:
                assert greenctx is not None
                with clamped_step_scope(
                    greenctx,
                    step_idx,
                    args.max_sm_count,
                ) as bundle:
                    active_sm = int(bundle.sm_count)
                    with torch.cuda.stream(bundle.comp):
                        optimizer.zero_grad(set_to_none=True)
                        with autocast(enabled=use_amp):
                            logits = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                            )
                            loss = F.cross_entropy(logits, labels)
                        preds = logits.argmax(dim=-1)
                        train_acc = float(
                            (preds == labels).float().mean().item()
                        )

                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()

                    bundle.comp.synchronize()
            else:
                active_sm = 0
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=use_amp):
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    loss = F.cross_entropy(logits, labels)
                preds = logits.argmax(dim=-1)
                train_acc = float((preds == labels).float().mean().item())

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                if use_cuda:
                    torch.cuda.synchronize(device)

            step_end_ns = time.clock_gettime_ns(
                time.CLOCK_MONOTONIC
            )
            step_boundaries.append(
                {
                    "step": step_idx,
                    "start_ns": step_start_ns,
                    "end_ns": step_end_ns,
                    "sm_count": min(
                        active_sm,
                        args.max_sm_count,
                    ),
                }
            )

            step_time = max(
                time.perf_counter() - step_start, 1e-12
            )
            throughput = float(labels.size(0)) / step_time

            if use_cuda:
                mem_alloc = (
                    torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
                )
                mem_resv = (
                    torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)
                )
                mem_peak = (
                    torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                )
            else:
                mem_alloc = 0.0
                mem_resv = 0.0
                mem_peak = 0.0

            row = {
                "step": step_idx,
                "loss": float(loss.item()),
                "accuracy": train_acc,
                "lr": lr,
                "throughput_samples_per_sec": throughput,
                "mem_alloc_mb": mem_alloc,
                "mem_reserved_mb": mem_resv,
                "mem_peak_alloc_mb": mem_peak,
                "sm_count": min(
                    active_sm, args.max_sm_count
                ),
            }
            step_metrics.append(row)

            if step_idx % args.log_interval == 0:
                logger.info(
                    "step=%d loss=%.4f acc=%.4f lr=%.3e "
                    "throughput=%.2f samples/s mem=%.1f/%.1f MB",
                    step_idx,
                    row["loss"],
                    row["accuracy"],
                    row["lr"],
                    row["throughput_samples_per_sec"],
                    row["mem_alloc_mb"],
                    row["mem_reserved_mb"],
                )

            should_eval = (step_idx + 1) % args.eval_interval == 0
            should_eval = should_eval or step_idx == (args.max_iters - 1)
            if should_eval:
                val_loss, val_acc = evaluate(
                    model=model,
                    loader=val_loader,
                    device=device,
                    use_amp=use_amp,
                )
                logger.info(
                    "eval step=%d val_loss=%.4f val_acc=%.4f",
                    step_idx,
                    val_loss,
                    val_acc,
                )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    ckpt = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step_idx,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "config": asdict(cfg),
                    }
                    best_path = output_dir / "best.pt"
                    torch.save(ckpt, best_path)
                    logger.info("saved best checkpoint: %s", best_path)

        total_time = time.perf_counter() - total_start
        logger.info("training done in %.2f sec", total_time)

    finally:
        if greenctx is not None:
            greenctx.close()

    final_ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": args.max_iters - 1,
        "config": asdict(cfg),
    }
    final_path = output_dir / "final.pt"
    torch.save(final_ckpt, final_path)

    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(step_metrics[0].keys()) if step_metrics else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in step_metrics:
            writer.writerow(row)

    boundaries_path = output_dir / "step_boundaries.csv"
    with boundaries_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        bw = csv.DictWriter(
            f,
            fieldnames=[
                "step", "start_ns",
                "end_ns", "sm_count",
            ],
        )
        bw.writeheader()
        for brow in step_boundaries:
            bw.writerow(brow)
    logger.info(
        "wrote step boundaries: %s",
        boundaries_path,
    )

    summary_path = output_dir / "summary.json"
    summary = {
        "best_val_acc": best_val_acc,
        "max_iters": args.max_iters,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_sm_count": args.max_sm_count,
        "greenctx_enabled": args.greenctx_enabled,
        "greenctx_supported": greenctx_supported,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("wrote metrics: %s", csv_path)
    logger.info("wrote summary: %s", summary_path)
    logger.info("wrote final checkpoint: %s", final_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args()

    script_path = Path(__file__).resolve()
    maybe_configure_gemm(args, script_path)

    if args.greenctx_trace and not args.greenctx_enabled:
        logger.warning(
            "--greenctx-trace provided without --greenctx-enabled; "
            "trace is ignored"
        )

    train(args)


if __name__ == "__main__":
    main()
