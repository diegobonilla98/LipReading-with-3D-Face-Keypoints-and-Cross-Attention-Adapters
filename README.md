## LipReading with 3D Face Keypoints and Cross‑Attention Adapters

This project trains a language model to lip‑read by conditioning on dense 3D facial keypoints extracted from video. Dense landmarks are produced with MediaPipe Face Mesh, normalized/frontalized, then encoded into a compact set of conditioning tokens that are injected into a frozen Causal LM via zero‑init gated cross‑attention adapters.

The pipeline is dataset‑agnostic (any talking‑head video with aligned transcripts works). The author used FFMTIMIT, but any similar corpus can be preprocessed into the required manifest format.

### Highlights
- **Dataset‑agnostic, MediaPipe‑based alignment and normalization** of ~478 3D face landmarks per frame
- **Face keypoint encoder**: spatial Transformer per frame → temporal Transformer over frames → Perceiver‑style resampling to S conditioning tokens
- **Frozen base Causal LM** (e.g., `HuggingFaceTB/SmolLM2-135M`) conditioned by **zero‑init gated cross‑attention adapters** added to selected layers
- **Efficient training** with AMP/bfloat16/float16, cosine schedule, gradient clipping/accumulation, checkpointing, and TensorBoard logging

---

## Repository structure (key files)
- `train.py`: End‑to‑end trainer, adapter injection wrapper, checkpoints, evaluation, TensorBoard logging
- `FaceKeypointEmbedding.py`: Dense face keypoint encoder → `[B, S, d_model]` conditioner
- `kp_dataloader.py`: JSONL manifest loader and collate for paired (keypoints, text)
- `dense_face_keypoints.py`: MediaPipe landmark extraction, frontalization via head pose, and normalization utilities
- `head_pose_estimation.py`: OpenCV PnP head‑pose solver used during frontalization

---

## Installation
Requires Python ≥ 3.10 and a recent PyTorch. Install dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # pick the right CUDA/CPU build
pip install transformers==4.* mediapipe opencv-python numpy tqdm tensorboard psutil
```

Windows notes:
- The trainer disables pinned memory on Windows to avoid event recording issues.
- Multiprocessing DataLoader workers are supported; the collate is designed to be pickle‑safe on Windows.

---

## Data preparation (MediaPipe → .npy keypoint tensors + JSONL manifest)

Training expects a JSONL manifest where each line contains the paired transcript and a path to a `.npy` file storing the per‑frame landmarks:

```json
{"text": "SPOKEN TRANSCRIPT ...", "kp_path": "path/to/sample.npy"}
```

The `.npy` must be a float32 array of shape `[T, 478, 3]` where:
- `T` is the number of frames in the utterance segment
- 478 are the MediaPipe Face Mesh landmarks (with iris/lips refinement)
- 3 are the coordinates in the project’s convention: X right, Y up, Z towards camera

### Landmark extraction and normalization
Implemented in `dense_face_keypoints.py` via `Face3DKeypointExtractor`:
- Uses MediaPipe Face Mesh with `refine_landmarks=True` to obtain ~478 landmarks per frame
- Optional **frontalization**: estimates head pose (yaw/pitch/roll) using `head_pose_estimation.solve_head_pose` (OpenCV `solvePnP` on six stable facial points), then rotates landmarks to a canonical frontal pose while preserving translation about the nose tip
- **Normalization** options (`normalize` arg):
  - `unit_sphere` (default): center at centroid and scale so max distance to centroid is 1
  - `bbox_unit`: center at centroid, scale by largest bbox side
  - `zscore`: per‑axis standardization (mean 0, std 1)

Minimal example to turn a single video into a keypoint tensor for one utterance segment:

```python
import cv2, numpy as np
from dense_face_keypoints import Face3DKeypointExtractor

def extract_video_segment_kp(video_path, start_sec, end_sec, normalize="unit_sphere"):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    start_f, end_f = int(start_sec * fps), int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    extractor = Face3DKeypointExtractor(normalize=normalize, frontalize=True)
    frames_kp = []
    for f in range(start_f, end_f):
        ok, frame = cap.read()
        if not ok:
            break
        try:
            kp = extractor.extract(frame)  # (478, 3) float32 in project coords
            frames_kp.append(kp)
        except Exception:
            # drop frames where no face is detected; optional: pad/interpolate later
            continue
    cap.release()
    if not frames_kp:
        raise RuntimeError("no frames processed")
    arr = np.stack(frames_kp, axis=0).astype(np.float32)  # [T, 478, 3]
    return arr

# Usage
# arr = extract_video_segment_kp("/path/to/video.mp4", 1.2, 3.8)
# np.save("/data/kp/sample_0001.npy", arr)
```

Then create a JSONL manifest file (one line per utterance):

```python
import json
records = [
    {"text": "THE QUICK BROWN FOX", "kp_path": "/data/kp/sample_0001.npy"},
    {"text": "JUMPS OVER THE LAZY DOG", "kp_path": "/data/kp/sample_0002.npy"},
]
with open("/data/manifest.jsonl", "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
```

Guidelines:
- Ensure each utterance segment’s transcript aligns with its frames in `kp_path`
- Prefer `unit_sphere` normalization and enable frontalization for pose‑invariant conditioning
- The trainer will pad the time dimension `T` within a batch (no need to pre‑pad offline)

---

## Data loading and collate
Implemented in `kp_dataloader.py`:
- `KeypointTextDataset`: reads a list of manifest records; on `__getitem__` loads `kp_path` (`[T, 478, 3]`) and returns `{"text", "face_kp"}`
- `collate_keypoint_batch`:
  - Tokenizes text with a HuggingFace tokenizer (no special tokens by default), pads/truncates to `--seq_len`, and optionally inserts an EOS if space remains
  - Builds `labels` with padding positions masked to `-100` for standard Causal LM training
  - Pads per‑sample keypoint sequences to the max `T` in the batch and returns `face_kp` `[B, T_max, 478, 3]` and `face_kp_mask` `[B, T_max]`
- `keypoint_collate_with_model_id` caches a tokenizer per worker process (Windows‑safe)

Manifest split helpers: `split_records(records, val_ratio)` creates train/val splits with a fixed seed for reproducibility.

---

## Face keypoint encoder (conditioning model)
Implemented in `FaceKeypointEmbedding.py` as `FaceKeypointConditionProvider` and `FaceKeypointEncoder`.

Input: `[B, T, 478, 3]` → Output: `[B, S, d_model]` where `S = --cond_len` and `d_model` matches the base LM hidden size.

Stages:
- **Per‑landmark MLP** embeds 3D coords to `kp_embed_dim` and adds a learned landmark ID embedding
- **Spatial Transformer (per frame)** processes `[1 + 478]` tokens (prepended CLS) with `spatial_layers` layers; the CLS token summarizes the frame, then a linear projects to the LM’s `d_model`
- **Temporal Transformer (over frames)** applies sinusoidal positional embeddings and `temporal_layers` of self‑attention over the frame sequence
- **Perceiver‑style resampler** cross‑attends learned latent queries to the temporal features to produce exactly `S` conditioning tokens

Numerical/shape notes:
- Keypoint normalization per frame (centroid shift and scale stabilization) is applied inside the encoder
- Temporal key padding mask is propagated through the temporal encoder and resampler

Hyperparameters exposed via `train.py` flags:
- `--cond_len`, `--kp_embed_dim`, `--kp_spatial_layers`, `--kp_temporal_layers`, `--kp_heads`, `--kp_dropout`, `--kp_ff_mult`, `--default_kp_frames`

---

## Cross‑attention adapters on a frozen LM
Defined in `train.py`:

- `ConditionedCausalLM` wraps a HuggingFace `AutoModelForCausalLM`, freezes all base weights, and replaces a subset of transformer blocks with `_BlockWithAdapter`.
- Each adapter is a **GatedCrossAttention** module:
  - LayerNorm on queries (LM hidden states) and keys/values (conditioning tokens)
  - Multi‑head cross‑attention output projection is zero‑initialized
  - A scalar **gate parameter** is initialized to ~0 (`sigmoid(-10)`) so the residual path starts as an identity mapping and learns to open gradually
- Layer placement:
  - By default, insert an adapter every `--adapter_every` layers (top‑biased indexing)
  - Or use `--adapter_top_k` to place adapters only on the top‑K layers

The LM forward is invoked as:
```python
out = conditioned(
    cond_tokens=cond_tokens,  # [B, S, d_model]
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    labels=batch["labels"],
)
loss = out.loss
```

---

## Training

Basic run (paired keypoints + text):
```bash
python train.py \
  --model_id HuggingFaceTB/SmolLM2-135M \
  --manifest /data/manifest.jsonl \
  --output_dir ./runs/smollm2_xattn \
  --batch_size 16 --eval_batch_size 16 \
  --seq_len 1024 --epochs 3 \
  --adapter_every 4 --cond_len 8 \
  --kp_embed_dim 256 --kp_spatial_layers 2 --kp_temporal_layers 4 --kp_heads 8 \
  --amp --fp16
```

Notes:
- If `--manifest` is omitted, the trainer falls back to text‑only packed‑sequence training (legacy mode)
- AMP uses bfloat16 by default; add `--fp16` to use float16
- Per‑epoch and best checkpoints are saved in `output_dir`; a resumable `last_epoch.pth` is also written
- TensorBoard logs are under `output_dir/tensorboard_logs`

Resume training:
```bash
python train.py --manifest /data/manifest.jsonl --output_dir ./runs/smollm2_xattn --resume ./runs/smollm2_xattn/last_epoch.pth
```
…or use `--auto_resume` to pick up `last_epoch.pth` automatically.

Evaluation during training:
- Validation perplexity is computed periodically (`--eval_every`) and at the end of each epoch
- The best model (lowest perplexity) is saved to `output_dir/best` (LM adapters + tokenizer + conditioning encoder weights)

System metrics:
- Logs GPU memory, CPU usage, and RAM usage to TensorBoard

---

## Inference (sketch)

To use a trained checkpoint for generation, reconstruct the wrapper and load states:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from train import ConditionedCausalLM
from FaceKeypointEmbedding import FaceKeypointConditionProvider

ckpt_dir = "./runs/smollm2_xattn/best"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(ckpt_dir)  # or the original model_id
model = ConditionedCausalLM(base).to(device)
model.load_adapters(ckpt_dir)

# Load conditioning encoder weights
cond_provider = FaceKeypointConditionProvider(d_model=model.d_model, cond_len=8).to(device)
cond_provider.load_state_dict(torch.load(f"{ckpt_dir}/cond_encoder.pt", map_location="cpu"))
model.eval(); cond_provider.eval()

# Prepare keypoints [B, T, 478, 3] and masks, then:
# cond = cond_provider(batch, device, keypoints=kp, keypoint_mask=mask)
# outputs = model.generate(input_ids=..., attention_mask=..., cond_tokens=cond, ...)
```

---

## Head pose estimation details
`head_pose_estimation.py` computes yaw/pitch/roll using OpenCV’s `solvePnP` with six stable MediaPipe indices (nose tip, chin, eye corners, mouth corners). A canonical 3D face template is used to form correspondences, and the resulting rotation matrix is converted to Euler angles. During frontalization, the rotation is inverted and applied to relative landmark coordinates (anchored at the nose tip), then mapped back to the project’s coordinate system.

---

## Tips and troubleshooting
- Ensure MediaPipe finds a face in most frames; consider dropping short gaps or interpolating
- Keep utterance segments reasonably short to control `T`; the DataLoader pads `T` within a batch
- For small GPUs, lower `--batch_size` and/or `--seq_len`, or reduce `--kp_temporal_layers`
- If gates remain near zero, consider lowering learning rate or placing adapters on top layers (`--adapter_top_k`)

---

## Acknowledgments
- MediaPipe Face Mesh (Google), OpenCV, PyTorch, HuggingFace Transformers
- Default model: `HuggingFaceTB/SmolLM2-135M`


