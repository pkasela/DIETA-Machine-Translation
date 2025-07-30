"""
Neural Machine Translation Training Script for Leonardo HPC
Decoder-only LLM for ITA-ENG / ENG-ITA translation
"""

import os
import sys
import contextlib
import subprocess
import time
from datetime import timedelta

# from typing import Dict, List, Optional
import bitsandbytes as bnb
import datasets
import GPUtil
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers

# from accelerate import Accelerator, DataLoaderConfiguration, DistributedType
# from datasets import Features, IterableDataset, Sequence, Value, load_dataset
from einops import rearrange
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from x_transformers import Decoder, TransformerWrapper
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper


class Config:
    """Here we have all the parameters"""

    # model
    NUM_TOKENS = 51200
    MAX_SEQ_LEN = 1024
    MODEL_DIM = 2048
    MODEL_DEPTH = 6  # 24, 12, 6
    MODEL_HEADS = 32
    ROTARY_EMB_DIM = 64  # 128
    ROTARY_XPOS_SCALE_BASE = 32768  # 2048

    # training
    NUM_EPOCHS = 1
    BATCH_SIZE = 22  # 6 layers
    ACCUM_STEPS = 5
    LEARNING_RATE = 1e-5 * ACCUM_STEPS
    CLIP_GRAD_NORM = 1.0
    WEIGHT_DECAY = 1e-5
    BETAS = (0.9, 0.95)

    WARMUP_PERCENTAGE = 10
    SAVE_EVERY = 10000
    PRINT_EVERY = 100
    THROUGHPUT_EVERY = 500  # Print throughput stats every N steps

    # paths - to be set via environment variables in sh file
    # TOKENIZER_PATH = os.environ["TOKENIZER_PATH"]
    LOCAL_TOKENIZER_PATH = os.environ["LOCAL_TOKENIZER_PATH"]
    MODEL_SAVE_PATH = os.environ["MODEL_SAVE_PATH"]
    CACHE_DIR = os.environ["HF_DATASETS_CACHE"]
    BASE_DATASET_PATH = os.environ["BASE_DATASET_PATH"]
    BASE_CACHE_ROOT = os.environ["BASE_CACHE_ROOT"]

    # others
    USE_AMP = True
    SEED = 0
    TIMEOUT_SECONDS = 3600


class KeepInputIdsOnly:
    """Custom collator: keeps only the input_ids"""

    def __init__(self, data_collator):
        self.data_collator = data_collator

    def __call__(self, batch):
        collated = self.data_collator(batch)
        return {"input_ids": collated["input_ids"]}


class DistributedTrainer:
    """MAIN TRAINING CLASS: handles training setup and execution"""

    def __init__(self, rank: int, world_size: int, config: Config):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.local_gpu_id = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.dataloader = None
        self.scaler = None

        # Throughput tracking - simplified approach
        self.throughput_start_time = None
        self.total_tokens_processed = 0
        self.tokens_per_step = self.config.MAX_SEQ_LEN * self.config.BATCH_SIZE

        # Window-based tracking
        self.window_start_time = None
        self.window_start_step = 0
        self.window_tokens_processed = 0

    def setup_environment(self):
        """Setup distributed training environment and per-GPU cache directories"""
        # Setup GPU
        gpus_per_node = torch.cuda.device_count()
        self.local_gpu_id = self.rank % gpus_per_node
        print(
            f"Rank: {self.rank}, GPUs per node: {gpus_per_node}, local GPU ID: {self.local_gpu_id}"
        )

        # Create per-rank cache directories (each GPU gets its own cache)
        jobid = str(os.environ["SLURM_JOB_ID"])
        base_cache_root = f"{self.config.BASE_CACHE_ROOT}/{jobid}"
        rank_cache_dir = f"{base_cache_root}/rank_{self.rank}"
        triton_cache_dir = f"{rank_cache_dir}/triton"
        os.makedirs(triton_cache_dir, exist_ok=True)

        # ! we must have a cache per rank for torchinductor
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = rank_cache_dir
        os.environ["TRITON_CACHE_DIR"] = triton_cache_dir

        print(f"Rank {self.rank}: TORCHINDUCTOR_CACHE_DIR = {rank_cache_dir}")
        print(f"Rank {self.rank}: TRITON_CACHE_DIR = {triton_cache_dir}")

        # Set other environment variables for this rank
        os.environ["RANK"] = str(self.rank)
        os.environ["LOCAL_RANK"] = str(self.local_gpu_id)

        # Initialize distributed process group
        dist.init_process_group(
            "nccl",
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=self.config.TIMEOUT_SECONDS),
        )

        torch.cuda.set_device(self.local_gpu_id)

    def load_dataset(self):
        """load the dataset for the current rank"""
        if not (0 <= self.rank < self.world_size):
            raise ValueError(
                f"Rank must be between 0 and {self.world_size - 1}, got {self.rank}"
            )

        dataset_path = os.path.join(self.config.BASE_DATASET_PATH, str(self.rank))
        print(f"Loading dataset from: {dataset_path}")
        return datasets.load_from_disk(dataset_path)

    def setup_tokenizer_and_collator(self):
        """Setup tokenizer and data collator"""
        # Use the local path directly since we don't have internet on compute nodes
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.LOCAL_TOKENIZER_PATH,
            local_files_only=True,  # Ensure no internet access is attempted
        )
        tokenizer.eos_token = "</s>"
        tokenizer.add_eos_token = True

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        return tokenizer, KeepInputIdsOnly(data_collator)

    def create_dataloader(self, dataset, collate_fn):
        num_rows = dataset.num_rows
        print(f"Dataset size: {num_rows} rows")

        cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        num_workers = max(1, cpus_per_task - 1)
        print(f"CPUs per task: {cpus_per_task}, Using {num_workers} workers")

        return DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            drop_last=True,
            multiprocessing_context="spawn",
        )

    def create_model(self):
        """Create the transformer model via x_transformers and compile it"""
        model_trf = TransformerWrapper(
            num_tokens=self.config.NUM_TOKENS,
            max_seq_len=self.config.MAX_SEQ_LEN,
            use_abs_pos_emb=False,
            attn_layers=Decoder(
                dim=self.config.MODEL_DIM,
                depth=self.config.MODEL_DEPTH,
                heads=self.config.MODEL_HEADS,
                pre_norm=False,
                residual_attn=True,
                ff_relu_squared=True,
                attn_add_zero_kv=True,
                attn_dropout=0.0,
                ff_dropout=0.0,
                rotary_pos_emb=True,
                rotary_emb_dim=self.config.ROTARY_EMB_DIM,
                rotary_xpos_scale_base=self.config.ROTARY_XPOS_SCALE_BASE,
                rotary_xpos=True,
                attn_qk_norm=True,
                attn_qk_norm_dim_scale=True,
            ),
        )

        model_trf = torch.compile(model_trf)
        auto_model = AutoregressiveWrapper(model_trf)
        auto_model.to(self.local_gpu_id)
        auto_model.train()

        param_count = sum(p.numel() for p in auto_model.parameters() if p.requires_grad)
        print(f"Language model has {param_count:,} trainable parameters")

        return DDP(
            auto_model, device_ids=[self.local_gpu_id], output_device=self.local_gpu_id
        )

    def setup_optimizer_and_scheduler(self, total_steps: int):
        self.optimizer = ZeroRedundancyOptimizer(
            self.model.parameters(),
            optimizer_class=bnb.optim.Lion8bit,
            lr=self.config.LEARNING_RATE,
            betas=self.config.BETAS,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        warmup_steps = int((self.config.WARMUP_PERCENTAGE / 100) * total_steps)
        print(f"Warmup steps: {warmup_steps}, Total steps: {total_steps}")

        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.config.USE_AMP)

    def print_throughput_stats(self, step: int):
        """Calculate and print throughput statistics using window start/end times"""
        dist.barrier()  # Synchronize all processes

        current_time = time.time()

        # Calculate windowed throughput
        if self.window_start_time is not None:
            window_duration = current_time - self.window_start_time
            window_steps = step - self.window_start_step

            if window_duration > 0 and window_steps > 0:
                windowed_throughput = self.window_tokens_processed / window_duration
                avg_step_time = window_duration / window_steps
            else:
                windowed_throughput = 0
                avg_step_time = 0
        else:
            windowed_throughput = 0
            avg_step_time = 0
            window_steps = 0

        # Overall statistics (from start of the training)
        total_elapsed = current_time - self.throughput_start_time
        overall_tokens_per_second = (
            self.total_tokens_processed / total_elapsed if total_elapsed > 0 else 0
        )

        if self.rank == 0:
            print(f"\n=== THROUGHPUT STATS (Step {step}) ===")
            print(
                f"Configuration: {self.world_size} GPUs across {self.world_size // torch.cuda.device_count()} nodes"
            )
            tokens_per_step_total = self.tokens_per_step * self.world_size
            print(f"Total tokens per step (all GPUs): {tokens_per_step_total:,}")
            print(f"Total tokens processed so far: {self.total_tokens_processed:,}")
            print(f"")
            if self.window_start_time is not None:
                print(f"Window: {window_steps} steps over {window_duration:.2f}s")
                print(f"Windowed throughput: {windowed_throughput:,.0f} tokens/sec")
                print(f"Windowed avg step time: {avg_step_time:.3f}s")
                print(
                    f"Windowed throughput per GPU: {windowed_throughput/self.world_size:,.0f} tokens/sec per GPU"
                )
            else:
                print(f"Window: Initializing...")
            print(f"")
            print(
                f"Overall elapsed time: {total_elapsed:.2f}s ({total_elapsed/60:.1f} minutes)"
            )
            print(f"Overall throughput: {overall_tokens_per_second:,.0f} tokens/sec")
            print(
                f"Overall throughput per GPU: {overall_tokens_per_second/self.world_size:,.0f} tokens/sec per GPU"
            )
            print(f"========================================\n")
            sys.stdout.flush()

        # Reset window for next measurement
        self.window_start_time = current_time
        self.window_start_step = step
        self.window_tokens_processed = 0

    def save_checkpoint(self, epoch: int, step: int):
        """Save model, optimizer, and scheduler checkpoints"""
        dist.barrier()  # Synchronize all processes
        self.optimizer.consolidate_state_dict(to=0)

        if self.rank == 0:
            checkpoint_prefix = (
                f"{self.config.MODEL_SAVE_PATH}{epoch}_rank{self.rank}_{step}"
            )

            torch.save(self.model.module.state_dict(), f"{checkpoint_prefix}.pt")
            torch.save(self.optimizer.state_dict(), f"{checkpoint_prefix}.optimizer.pt")
            torch.save(self.scheduler.state_dict(), f"{checkpoint_prefix}.scheduler.pt")

            print(f"Checkpoint saved: {checkpoint_prefix}")

    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        total_steps = len(self.dataloader)
        print(f"Start epoch {epoch + 1}, total steps: {total_steps}")

        self.optimizer.zero_grad(set_to_none=True)
        start_time = time.time()

        # Initialize throughput tracking for this epoch
        if self.throughput_start_time is None:
            self.throughput_start_time = time.time()
            self.window_start_time = self.throughput_start_time
            self.window_start_step = 0

        for step, batch in enumerate(tqdm(self.dataloader, disable=True), 1):
            step_start_time = time.time()
            input_ids = batch["input_ids"].to(self.local_gpu_id, non_blocking=True)

            # Decide if we should synchronize gradients
            at_step = step % self.config.ACCUM_STEPS == 0
            at_end = step == total_steps
            is_last = at_step or at_end

            sync_ctx = self.model.no_sync() if not is_last else contextlib.nullcontext()

            with sync_ctx, torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=self.config.USE_AMP
            ):
                logits, target = self.model(input_ids)
                loss = F.cross_entropy(rearrange(logits, "b n c -> b c n"), target)
                self.scaler.scale(loss / self.config.ACCUM_STEPS).backward()

            if is_last:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.CLIP_GRAD_NORM
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            # Update token counters for every step
            tokens_this_step = self.tokens_per_step * self.world_size
            self.total_tokens_processed += tokens_this_step
            self.window_tokens_processed += tokens_this_step

            # Logging
            if step % self.config.PRINT_EVERY == 0:
                if self.rank == 0:
                    tqdm.write(f"Step: {step}, Loss: {loss.item():.4f}")
                    sys.stdout.flush()

            # Throughput stats every N steps
            if step % self.config.THROUGHPUT_EVERY == 0:
                self.print_throughput_stats(step)

            # Checkpoints
            if step % self.config.SAVE_EVERY == 0:
                print(f"Saving checkpoint at step {step}")
                execution_time = time.time() - start_time
                print(
                    f"Execution time for last {self.config.SAVE_EVERY} steps: {execution_time:.2f}s"
                )
                start_time = time.time()

                self.print_throughput_stats(step)
                self.save_checkpoint(epoch + 1, step)

            # Last step
            if step == total_steps:
                if self.rank == 0:
                    tqdm.write(f"Final step: {step}, Loss: {loss.item():.4f}")
                    sys.stdout.flush()

                execution_time = time.time() - start_time
                print(f"Final execution time: {execution_time:.2f}s")

                self.print_throughput_stats(step)
                self.save_checkpoint(epoch + 1, step)

    def train(self):
        """Main training loop"""
        self.setup_environment()

        # data
        dataset = self.load_dataset()
        tokenizer, collate_fn = self.setup_tokenizer_and_collator()
        self.dataloader = self.create_dataloader(dataset, collate_fn)

        # model
        self.model = self.create_model()

        # training components
        total_steps = len(self.dataloader)
        self.setup_optimizer_and_scheduler(total_steps)

        # training loop
        print("STARTING TRAINING")
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            self.train_epoch(epoch)

            # last epoch checkpoint
            print(
                f"Rank {self.rank}: Starting final checkpoint saving for epoch {epoch + 1}"
            )
            self.save_checkpoint(epoch + 1, 0)  # 0 for final checkpoint

        dist.barrier()
        print(f"Rank {self.rank}: Training completed")

        # Print final comprehensive throughput summary
        if self.rank == 0 and self.throughput_start_time:
            total_elapsed = time.time() - self.throughput_start_time
            overall_tokens_per_second = (
                self.total_tokens_processed / total_elapsed if total_elapsed > 0 else 0
            )

            print(f"\n" + "=" * 80)
            print(f"FINAL THROUGHPUT SUMMARY")
            print(f"=" * 80)
            print(
                f"Configuration: {self.world_size} GPUs across {self.world_size // torch.cuda.device_count()} nodes"
            )
            print(f"Batch size per GPU: {self.config.BATCH_SIZE}")
            print(f"Sequence length: {self.config.MAX_SEQ_LEN}")
            print(
                f"Tokens per step (all GPUs): {self.tokens_per_step * self.world_size:,}"
            )
            print(f"")
            print(
                f"Total training time: {total_elapsed:.2f}s ({total_elapsed/60:.1f} minutes)"
            )
            print(f"Total tokens processed: {self.total_tokens_processed:,}")
            print(f"")
            print(f"Overall throughput: {overall_tokens_per_second:,.0f} tokens/sec")
            print(
                f"Overall throughput per GPU: {overall_tokens_per_second/self.world_size:,.0f} tokens/sec/GPU"
            )

            # Calculate average step time based on total time and tokens
            if self.total_tokens_processed > 0:
                total_steps = self.total_tokens_processed // (
                    self.tokens_per_step * self.world_size
                )
                avg_step_time = total_elapsed / total_steps if total_steps > 0 else 0
                print(f"")
                print(f"Estimated total steps: {total_steps}")
                print(f"Average step time: {avg_step_time:.3f}s")

            print(f"=" * 80)
            sys.stdout.flush()

    def cleanup(self):
        """Clean up dist"""
        dist.destroy_process_group()


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_gpu_utilization():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU ID: {gpu.id}, Name: {gpu.name}")
        print(f"  Load: {gpu.load*100:.1f}%")
        print(f"  Memory Free: {gpu.memoryFree} MB")
        print(f"  Memory Used: {gpu.memoryUsed} MB")
        print(f"  Total Memory: {gpu.memoryTotal} MB\n")


def get_nvidia_smi():
    result = subprocess.run(
        ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return result.stdout


def validate_environment():
    """Validate that all required environment variables are set"""
    required_env_vars = [
        # "TOKENIZER_PATH",
        "LOCAL_TOKENIZER_PATH",
        "MODEL_SAVE_PATH",
        "HF_DATASETS_CACHE",
        "BASE_DATASET_PATH",
        "BASE_CACHE_ROOT",
        "WORLD_SIZE",
        "SLURM_PROCID",
        "SLURM_JOB_ID",
    ]

    missing_vars = []
    for var in required_env_vars:
        if var not in os.environ:
            missing_vars.append(var)

    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            f"These must be set in the shell script before running the training."
        )

    print("All required environment variables are set")


def main():
    # check that all the environment variables are set (to be set in sh file)
    validate_environment()

    set_seed(Config.SEED)

    # get params from SLURM environment
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    print(f"World size: {world_size}, Rank: {rank}")

    # train
    config = Config()
    trainer = DistributedTrainer(rank, world_size, config)
    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed on rank {rank}: {e}")
        raise
    finally:
        trainer.cleanup()
    print("Training completed successfully")


if __name__ == "__main__":
    main()
