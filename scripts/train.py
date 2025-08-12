import argparse
import gc
import glob
import math
import os
import random
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import psutil

# Make src package importable when running from project root
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from lipreading.data.keypoint_dataset import (
    KeypointTextDataset,
    collate_keypoint_batch,
    load_manifest,
    split_records,
)
from lipreading.models.face_keypoint_encoder import (
    FaceKeypointConditionProvider,
)

# -----------------------------
# NEW: xAttn adapter components
# -----------------------------

class GatedCrossAttention(nn.Module):
    """
    Residual cross-attention with zero-init gate (identity at init).
    Expects cond_tokens as [B, S, d_model] in forward().
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        # zero-init output proj so residual starts at (almost) exact identity with gate=0
        with torch.no_grad():
            self.mha.out_proj.weight.zero_()
            if self.mha.out_proj.bias is not None:
                self.mha.out_proj.bias.zero_()
        # Initialize gate to be effectively closed (~0) so residual is identity at start
        self.gate = nn.Parameter(torch.tensor(-10.0))

    def forward(self, hidden_states: torch.Tensor, cond_tokens: torch.Tensor | None):
        if cond_tokens is None or cond_tokens.numel() == 0:
            return hidden_states
        # [B, T, D] xattn [B, S, D]
        q = self.ln_q(hidden_states)
        # ensure dtype & device match q before MHA
        kv = self.ln_kv(cond_tokens.to(dtype=q.dtype, device=q.device))
        z, _ = self.mha(q, kv, kv, need_weights=False)  # [B, T, D]
        # gate in (0,1)
        g = torch.sigmoid(self.gate)
        return hidden_states + g * z


class _BlockWithAdapter(nn.Module):
    """
    Wraps a transformer block: calls original block, then applies cross-attn residual.
    Accepts (*args, **kwargs) from the base model; returns same structure but with
    hidden_states post-processed by the adapter.
    """
    def __init__(self, orig_block: nn.Module, adapter: nn.Module, cond_getter):
        super().__init__()
        self.block = orig_block
        self.adapter = adapter
        self._cond_getter = cond_getter  # lambda -> cond_tokens [B,S,D] or None

    def forward(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        # Block outputs across HF decoders are usually Tensor or tuple where first is hidden states
        if isinstance(out, tuple):
            hs = out[0]
            hs2 = self.adapter(hs, self._cond_getter())
            return (hs2,) + out[1:]
        elif hasattr(out, "last_hidden_state"):  # rare for block-level, but guard
            out.last_hidden_state = self.adapter(out.last_hidden_state, self._cond_getter())
            return out
        else:  # Tensor
            return self.adapter(out, self._cond_getter())


def _get_hidden_size(cfg):
    for name in ("hidden_size", "n_embd", "d_model"):
        if hasattr(cfg, name):
            return getattr(cfg, name)
    raise ValueError("Cannot find hidden size in config.")

def _get_num_heads(cfg):
    for name in ("num_attention_heads", "n_head"):
        if hasattr(cfg, name):
            return getattr(cfg, name)
    raise ValueError("Cannot find num_attention_heads in config.")

def _find_block_list(base_model: nn.Module):
    """
    Find the sequential list of transformer blocks in common HF decoders.
    Supports LLaMA-like (model.layers), GPT2-like (transformer.h), etc.
    """
    # Common containers
    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        return base_model.model.layers
    if hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h"):
        return base_model.transformer.h
    if hasattr(base_model, "model") and hasattr(base_model.model, "decoder") and hasattr(base_model.model.decoder, "layers"):
        return base_model.model.decoder.layers
    # Fallback: first ModuleList with > 1 elements
    for m in base_model.modules():
        if isinstance(m, nn.ModuleList) and len(m) > 1:
            return m
    raise RuntimeError("Could not locate the transformer block list to wrap.")

class ConditionedCausalLM(nn.Module):
    """
    Wrapper that holds a frozen base CausalLM and interleaves zero-init gated cross-attn adapters.
    Call forward(cond_tokens=..., **lm_kwargs).
    """
    def __init__(self, base_lm: AutoModelForCausalLM, adapter_every: int = 4, adapter_top_k: int | None = None):
        super().__init__()
        self.base = base_lm
        # Freeze base model
        for p in self.base.parameters():
            p.requires_grad = False

        self.d_model = _get_hidden_size(self.base.config)
        self.n_heads = _get_num_heads(self.base.config)

        # Condition storage for current forward()
        self._current_cond = None

        # Locate blocks and wrap a subset with adapters
        blocks = _find_block_list(self.base)
        self._wrapped_indices = []
        self.adapters = nn.ModuleList()

        n_layers = len(blocks)
        indices = list(range(n_layers))
        if adapter_top_k is not None:
            # put adapters only on the top adapter_top_k layers
            indices = indices[-adapter_top_k:]
        else:
            # every N layers (1-indexed feel), biased to topish layers
            indices = [i for i in indices if (i % adapter_every) == (adapter_every - 1)]

        def cond_getter():
            return self._current_cond

        for i in indices:
            adapter = GatedCrossAttention(self.d_model, self.n_heads)
            wrapped = _BlockWithAdapter(blocks[i], adapter, cond_getter)
            blocks[i] = wrapped  # in-place swap
            self.adapters.append(adapter)
            self._wrapped_indices.append(i)

        # Disable KV cache warnings for training
        if hasattr(self.base.config, "use_cache"):
            self.base.config.use_cache = False

    def forward(self, *, cond_tokens: torch.Tensor | None = None, **lm_kwargs):
        self._current_cond = cond_tokens  # seen by adapters via cond_getter()
        return self.base(**lm_kwargs)

    # Convenience helpers
    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def num_trainable(self):
        return sum(p.numel() for p in self.trainable_parameters())

    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        # Save reference to base and adapters state only
        with open(os.path.join(save_dir, "base_reference.txt"), "w", encoding="utf-8") as f:
            f.write(getattr(self.base, "name_or_path", "UNKNOWN"))
        torch.save({
            "indices": self._wrapped_indices,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "adapters": self.adapters.state_dict(),
        }, os.path.join(save_dir, "adapters.pt"))

    def load_adapters(self, save_dir: str):
        pkg = torch.load(os.path.join(save_dir, "adapters.pt"), map_location="cpu")
        self.adapters.load_state_dict(pkg["adapters"], strict=True)

# -----------------------------
# END new components
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_texts(path: str):
    # Deprecated path for pure-text training. We now expect a JSONL manifest via --manifest.
    # Kept for backward compatibility if needed.
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".txt":
        return [p.read_text(encoding="utf-8", errors="ignore")]
    elif p.is_dir():
        files = sorted(glob.glob(str(p / "**" / "*.txt"), recursive=True))
        texts = []
        for f in files:
            try:
                texts.append(Path(f).read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
        if not texts:
            raise ValueError(f"No .txt files found under {path}")
        return texts
    else:
        raise ValueError("data_path must be a .txt file or a directory containing .txt files")

def build_packed_ids(tokenizer, texts, add_bos=False):
    """Tokenize each doc, append EOS, optionally prepend BOS, then flatten."""
    all_ids = []
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id; cannot proceed safely.")
    for doc in texts:
        ids = tokenizer.encode(doc, add_special_tokens=False)
        if add_bos and tokenizer.bos_token_id is not None:
            all_ids.append(tokenizer.bos_token_id)
        all_ids.extend(ids)
        all_ids.append(eos_id)
    return torch.tensor(all_ids, dtype=torch.long)

class PackedSequenceDataset(Dataset):
    """Yield contiguous blocks of length seq_len from a single long token stream."""
    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        self.seq_len = seq_len
        n_full = (len(token_ids) // seq_len)
        self.data = token_ids[: n_full * seq_len].view(n_full, seq_len)

    def __len__(self): return self.data.size(0)

    def __getitem__(self, idx):
        input_ids = self.data[idx]
        labels = input_ids.clone()
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def split_train_val(token_ids: torch.Tensor, val_ratio: float, seq_len: int):
    n = len(token_ids)
    cut = int(n * (1.0 - val_ratio))
    train_ids = token_ids[:cut]
    val_ids   = token_ids[cut:]
    return (PackedSequenceDataset(train_ids, seq_len),
            PackedSequenceDataset(val_ids,   seq_len))

def collate_no_pad(batch):
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}

# Tokenizer cache that lives per-process (workers will each have their own copy)
_COLLATE_TOKENIZER_CACHE: dict[str, AutoTokenizer] = {}

def keypoint_collate_with_model_id(batch, *, model_id: str, seq_len: int, pad_token_id: int, eos_token_id: int | None):
    """Top-level collate function suitable for multiprocessing on Windows.

    Lazily initializes and caches a tokenizer per-process using model_id to avoid
    capturing a large, potentially non-pickleable object inside the collate_fn.
    """
    tok = _COLLATE_TOKENIZER_CACHE.get(model_id)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        _COLLATE_TOKENIZER_CACHE[model_id] = tok
    return collate_keypoint_batch(batch, tok, seq_len, pad_token_id, eos_token_id)

def save_checkpoint(checkpoint_path: str, conditioned_model, cond_provider, optimizer, scheduler, 
                   epoch: int, global_step: int, best_val_ppl: float, args):
    """Save complete training state for resuming."""
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'best_val_ppl': best_val_ppl,
        'model_state_dict': conditioned_model.adapters.state_dict(),
        'cond_provider_state_dict': cond_provider.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'args': vars(args),
        'model_config': {
            'model_id': args.model_id,
            'adapter_every': args.adapter_every,
            'adapter_top_k': args.adapter_top_k if args.adapter_top_k > 0 else None,
            'd_model': conditioned_model.d_model,
            'n_heads': conditioned_model.n_heads,
            'wrapped_indices': conditioned_model._wrapped_indices,
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Save with atomic write (write to temp file, then rename)
    temp_path = checkpoint_path + '.tmp'
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, checkpoint_path)
    print(f"💾 Checkpoint saved to {checkpoint_path}")

def load_checkpoint(checkpoint_path: str, conditioned_model, cond_provider, optimizer, scheduler, device):
    """Load training state from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"📂 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model states
    conditioned_model.adapters.load_state_dict(checkpoint['model_state_dict'])
    cond_provider.load_state_dict(checkpoint['cond_provider_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Return training state
    return {
        'epoch': checkpoint['epoch'],
        'global_step': checkpoint['global_step'],
        'best_val_ppl': checkpoint['best_val_ppl'],
        'args': checkpoint.get('args', {}),
        'model_config': checkpoint.get('model_config', {})
    }

# -----------------------------
# Conditioning provider is implemented in FaceKeypointEmbedding.py
# -----------------------------

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    # Avoid CUDA pinned-memory event recording issues on Windows
    use_pin_memory = (device.type == "cuda") and (not sys.platform.startswith("win"))
    use_persistent_workers = args.workers > 0

    print(f"Loading tokenizer/model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_max = getattr(tokenizer, "model_max_length", 2048)
    seq_len = min(args.seq_len, model_max if model_max and model_max > 0 else args.seq_len)
    print(f"Using sequence length: {seq_len} (tokenizer model_max_length={model_max})")

    if args.manifest is not None and os.path.isfile(args.manifest):
        print("Reading dataset manifest…")
        records = load_manifest(args.manifest)
        train_recs, val_recs = split_records(records, args.val_ratio, seed=args.seed)
        train_ds = KeypointTextDataset(train_recs)
        val_ds = KeypointTextDataset(val_recs)
        # Create a pickle-safe collate function via partial with simple args
        keypoint_collate_fn = partial(
            keypoint_collate_with_model_id,
            model_id=args.model_id,
            seq_len=seq_len,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=use_pin_memory,
            persistent_workers=use_persistent_workers,
            drop_last=True,
            collate_fn=keypoint_collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=use_pin_memory,
            persistent_workers=use_persistent_workers,
            drop_last=False,
            collate_fn=keypoint_collate_fn,
        )
    else:
        print("Reading text dataset…")
        texts = read_texts(args.data_path)
        print(f"Loaded {len(texts)} documents. Tokenizing & packing…")
        all_ids = build_packed_ids(tokenizer, texts, add_bos=False)
        train_ds, val_ds = split_train_val(all_ids, args.val_ratio, seq_len)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=use_pin_memory,
            persistent_workers=use_persistent_workers,
            drop_last=True,
            collate_fn=collate_no_pad,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=use_pin_memory,
            persistent_workers=use_persistent_workers,
            drop_last=False,
            collate_fn=collate_no_pad,
        )

    # Base LM
    base = AutoModelForCausalLM.from_pretrained(args.model_id)

    # Wrap with gated x-attn adapters, freeze base
    conditioned = ConditionedCausalLM(
        base_lm=base,
        adapter_every=args.adapter_every,
        adapter_top_k=args.adapter_top_k if args.adapter_top_k > 0 else None
    ).to(device)

    print(f"Inserted adapters on layers: {conditioned._wrapped_indices}")
    print(f"Trainable params (adapters only): {conditioned.num_trainable():,}")

    conditioned.train()
    if hasattr(conditioned.base.config, "use_cache"):
        conditioned.base.config.use_cache = False

    # Instantiate conditioner (face keypoints)
    d_model = _get_hidden_size(conditioned.base.config)
    cond_provider = FaceKeypointConditionProvider(
        d_model=d_model,
        cond_len=args.cond_len,
        default_frames=args.default_kp_frames,
        kp_embed_dim=args.kp_embed_dim,
        spatial_layers=args.kp_spatial_layers,
        temporal_layers=args.kp_temporal_layers,
        n_heads=args.kp_heads,
        dropout=args.kp_dropout,
        ff_mult=args.kp_ff_mult,
    ).to(device)

    # Optimizer: adapters + conditioner
    optim_params = list(p for p in conditioned.parameters() if p.requires_grad)
    optim_params += list(cond_provider.parameters())
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8
    )

    total_steps = math.ceil(len(train_loader) * args.epochs / max(1, args.grad_accum))
    warmup_steps = args.warmup_steps if args.warmup_steps >= 0 else int(0.03 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    use_amp = args.amp and device.type == "cuda"
    use_fp16 = args.fp16
    autocast_dtype = torch.float16 if use_fp16 else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and use_fp16))

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up TensorBoard logging
    tb_log_dir = os.path.join(args.output_dir, "tensorboard_logs")
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    
    # Initialize training state
    start_epoch = 0
    global_step = 0
    best_val_ppl = float("inf")
    
    # Handle resume logic
    resume_path = None
    if args.resume:
        resume_path = args.resume
    elif args.auto_resume:
        potential_resume_path = os.path.join(args.output_dir, "last_epoch.pth")
        if os.path.exists(potential_resume_path):
            resume_path = potential_resume_path
            
    if resume_path:
        try:
            checkpoint_data = load_checkpoint(resume_path, conditioned, cond_provider, optimizer, scheduler, device)
            start_epoch = checkpoint_data['epoch']
            global_step = checkpoint_data['global_step']
            best_val_ppl = checkpoint_data['best_val_ppl']
            
            print(f"✅ Resumed training from epoch {start_epoch + 1}, step {global_step}")
            print(f"📊 Best validation perplexity so far: {best_val_ppl:.3f}")
            
            # Log resume event to TensorBoard
            writer.add_text('training/resume_info', f'Resumed from {resume_path} at epoch {start_epoch + 1}, step {global_step}', global_step)
            
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint from {resume_path}: {e}")
            print("🔄 Starting training from scratch...")
            start_epoch = 0
            global_step = 0
            best_val_ppl = float("inf")

    # Ensure train mode for conditioner
    cond_provider.train()

    def evaluate():
        conditioned.eval()
        cond_provider.eval()
        nll, ntok = 0.0, 0
        val_losses = []
        
        # Progress bar for validation
        val_pbar = tqdm(val_loader, desc="Validating", leave=False, 
                       bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        
        with torch.no_grad():
            for batch in val_pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                keypoints = batch.get("face_kp", None) if isinstance(batch, dict) else None
                keypoint_mask = batch.get("face_kp_mask", None) if isinstance(batch, dict) else None
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype):
                    cond_tokens = cond_provider(batch, device, keypoints=keypoints, keypoint_mask=keypoint_mask)
                    out = conditioned(cond_tokens=cond_tokens, input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]) 
                    loss = out.loss
                tokens_in_batch = batch["input_ids"].numel()
                nll += loss.item() * tokens_in_batch
                ntok += tokens_in_batch
                val_losses.append(loss.item())
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{sum(val_losses)/len(val_losses):.4f}'
                })
        
        conditioned.train()
        cond_provider.train()
        ppl = math.exp(nll / max(1, ntok))
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
        
        # Log validation metrics to TensorBoard
        writer.add_scalar('validation/perplexity', ppl, global_step)
        writer.add_scalar('validation/loss', avg_val_loss, global_step)
        writer.add_scalar('validation/nll_per_token', nll / max(1, ntok), global_step)
        
        return ppl

    # Training loop with comprehensive logging
    training_start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        tokens_seen = 0
        step_times = []
        
        # Progress bar for epoch
        epoch_pbar = tqdm(
            total=len(train_loader), 
            desc=f"Epoch {epoch+1}/{args.epochs}",
            bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}'
        )
        
        # Progress bar for training steps within epoch
        train_pbar = tqdm(
            train_loader, 
            desc="Training", 
            leave=False,
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )
        
        for step, batch in enumerate(train_pbar, start=1):
            step_start_time = time.time()
            
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            keypoints = batch.get("face_kp", None) if isinstance(batch, dict) else None
            keypoint_mask = batch.get("face_kp_mask", None) if isinstance(batch, dict) else None
            
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype):
                cond_tokens = cond_provider(batch, device, keypoints=keypoints, keypoint_mask=keypoint_mask)
                out = conditioned(cond_tokens=cond_tokens, input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]) 
                loss = out.loss / args.grad_accum
                
            if use_amp and use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            running_loss += loss.item()
            tokens_seen += batch["input_ids"].numel()
            step_time = time.time() - step_start_time
            step_times.append(step_time)

            if step % args.grad_accum == 0:
                grad_norm = None
                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    if use_amp and use_fp16:
                        scaler.unscale_(optimizer)
                    all_trainables = list(p for p in conditioned.parameters() if p.requires_grad) + list(cond_provider.parameters())
                    grad_norm = torch.nn.utils.clip_grad_norm_(all_trainables, args.max_grad_norm)
                    
                if use_amp and use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                # Calculate metrics
                lr = scheduler.get_last_lr()[0]
                avg_step_time = sum(step_times[-10:]) / min(len(step_times), 10)  # Average of last 10 steps
                tokens_per_sec = tokens_seen / sum(step_times[-10:]) if step_times else 0
                
                # Update progress bars
                train_pbar.set_postfix({
                    'loss': f'{loss.item() * args.grad_accum:.4f}',
                    'lr': f'{lr:.2e}',
                    'tok/s': f'{tokens_per_sec:.0f}'
                })
                epoch_pbar.set_postfix({
                    'avg_loss': f'{running_loss / step:.4f}',
                    'step_time': f'{avg_step_time:.3f}s'
                })
                epoch_pbar.update(1)

                # Log training metrics to TensorBoard
                writer.add_scalar('training/loss', loss.item() * args.grad_accum, global_step)
                writer.add_scalar('training/learning_rate', lr, global_step)
                writer.add_scalar('training/tokens_per_second', tokens_per_sec, global_step)
                writer.add_scalar('training/step_time', avg_step_time, global_step)
                if grad_norm is not None:
                    writer.add_scalar('training/grad_norm', grad_norm, global_step)
                
                # Log system metrics
                if device.type == "cuda":
                    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
                    writer.add_scalar('system/gpu_memory_allocated_gb', memory_allocated, global_step)
                    writer.add_scalar('system/gpu_memory_reserved_gb', memory_reserved, global_step)
                
                cpu_percent = psutil.cpu_percent()
                ram_percent = psutil.virtual_memory().percent
                writer.add_scalar('system/cpu_percent', cpu_percent, global_step)
                writer.add_scalar('system/ram_percent', ram_percent, global_step)
                
                # Log model-specific metrics (adapter gate values)
                for i, adapter in enumerate(conditioned.adapters):
                    gate_value = torch.sigmoid(adapter.gate).item()
                    writer.add_scalar(f'model/adapter_{i}_gate_value', gate_value, global_step)
                
                # Log conditioning token statistics if available
                if cond_tokens is not None:
                    writer.add_scalar('model/cond_tokens_mean', cond_tokens.mean().item(), global_step)
                    writer.add_scalar('model/cond_tokens_std', cond_tokens.std().item(), global_step)

                if global_step % args.log_every == 0:
                    print(f"epoch {epoch+1} | step {global_step}/{total_steps} | loss {running_loss/step:.4f} | lr {lr:.2e} | tokens/s {tokens_per_sec:.0f}")
                    
                    # Force garbage collection periodically
                    if global_step % (args.log_every * 4) == 0:
                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                if args.eval_every > 0 and global_step % args.eval_every == 0:
                    val_ppl = evaluate()
                    print(f"[eval] step {global_step}: val perplexity = {val_ppl:.3f}")
                    if val_ppl < best_val_ppl:
                        best_val_ppl = val_ppl
                        ckpt = os.path.join(args.output_dir, "best")
                        conditioned.save_pretrained(ckpt)
                        tokenizer.save_pretrained(ckpt)
                        torch.save(cond_provider.state_dict(), os.path.join(ckpt, "cond_encoder.pt"))
                        print(f"  -> saved new best to {ckpt}")
                        writer.add_scalar('validation/best_perplexity', best_val_ppl, global_step)
        
        train_pbar.close()
        epoch_pbar.close()

        # End-of-epoch logging
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = running_loss / len(train_loader)
        avg_tokens_per_sec = tokens_seen / epoch_time if epoch_time > 0 else 0
        
        # Log epoch metrics to TensorBoard
        writer.add_scalar('epoch/duration_seconds', epoch_time, epoch + 1)
        writer.add_scalar('epoch/average_loss', avg_epoch_loss, epoch + 1)
        writer.add_scalar('epoch/average_tokens_per_second', avg_tokens_per_sec, epoch + 1)
        
        # Save regular epoch checkpoint
        ckpt = os.path.join(args.output_dir, f"epoch-{epoch+1}")
        conditioned.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        torch.save(cond_provider.state_dict(), os.path.join(ckpt, "cond_encoder.pt"))
        
        # Save resumable checkpoint (last_epoch.pth)
        last_epoch_path = os.path.join(args.output_dir, "last_epoch.pth")
        save_checkpoint(last_epoch_path, conditioned, cond_provider, optimizer, scheduler, 
                       epoch, global_step, best_val_ppl, args)
        
        print(f"Epoch {epoch+1} completed in {epoch_time:.1f}s | avg_loss: {avg_epoch_loss:.4f} | avg_tok/s: {avg_tokens_per_sec:.0f}")
        print(f"📁 Model checkpoint: {ckpt}")
        
        # Run final evaluation at end of epoch
        if args.eval_every > 0:
            val_ppl = evaluate()
            print(f"[end-of-epoch eval] epoch {epoch+1}: val perplexity = {val_ppl:.3f}")
            # Update best_val_ppl if this is better
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl

    # Final cleanup and summary
    total_training_time = time.time() - training_start_time
    print(f"\n🎉 Training completed successfully!")
    print(f"⏱️  Total training time: {total_training_time:.1f}s ({total_training_time/3600:.2f}h)")
    print(f"📊 Final validation perplexity: {best_val_ppl:.3f}")
    print(f"📁 Best model saved to: {os.path.join(args.output_dir, 'best')}")
    print(f"💾 Resumable checkpoint: {os.path.join(args.output_dir, 'last_epoch.pth')}")
    print(f"📈 TensorBoard logs saved to: {tb_log_dir}")
    print(f"💡 View logs with: tensorboard --logdir {tb_log_dir}")
    print(f"🔄 Resume training with: --resume {os.path.join(args.output_dir, 'last_epoch.pth')}")
    
    # Close TensorBoard writer
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="HuggingFaceTB/SmolLM2-135M",
                        help="e.g. HuggingFaceTB/SmolLM2-135M or HuggingFaceTB/SmolLM-360M")
    parser.add_argument("--manifest", type=str, default=None, help="JSONL manifest with 'text' and 'kp_path' for paired training")
    parser.add_argument("--data_path", type=str, default=None, help="Fallback: folder with .txt files or a single .txt file (text-only)")
    parser.add_argument("--output_dir", type=str, default="./smollm-finetuned-xattn")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=-1, help="-1 => 3% of total steps")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--amp", action="store_true", help="Use AMP (bfloat16/float16 autocast)")
    parser.add_argument("--fp16", action="store_true", help="Use float16 instead of bfloat16 in autocast")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    # NEW: adapter controls + cond length
    parser.add_argument("--adapter_every", type=int, default=4, help="Insert an adapter every N layers (top-biased indexing).")
    parser.add_argument("--adapter_top_k", type=int, default=0, help="If >0, put adapters only on top-k layers (overrides adapter_every).")
    parser.add_argument("--cond_len", type=int, default=8, help="Number of conditioning tokens S.")
    # Conditioning model knobs
    parser.add_argument("--default_kp_frames", type=int, default=8, help="T used if no keypoints are provided.")
    parser.add_argument("--kp_embed_dim", type=int, default=256)
    parser.add_argument("--kp_spatial_layers", type=int, default=2)
    parser.add_argument("--kp_temporal_layers", type=int, default=4)
    parser.add_argument("--kp_heads", type=int, default=8)
    parser.add_argument("--kp_dropout", type=float, default=0.0)
    parser.add_argument("--kp_ff_mult", type=float, default=4.0)
    # Resume training arguments
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume from (e.g., './output/last_epoch.pth')")
    parser.add_argument("--auto_resume", action="store_true", help="Automatically resume from last_epoch.pth if it exists in output_dir")
    args = parser.parse_args()
    train(args)
