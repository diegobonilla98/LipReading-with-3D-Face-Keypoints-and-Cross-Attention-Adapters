import torch
import torch.nn as nn
import math


# -----------------------------
# NEW: Face keypoints -> [B,S,d_model] conditioner
# -----------------------------
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        self.dim = dim
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def _maybe_extend(self, T: int, device: torch.device, dtype: torch.dtype):
        if T <= self.pe.size(0):
            return
        # Regenerate a larger table
        new_max = int(2 ** (math.ceil(math.log2(T))))
        pe = torch.zeros(new_max, self.dim, device=device, dtype=torch.float)
        position = torch.arange(0, new_max, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2, device=device).float() * (-math.log(10000.0) / self.dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.detach().cpu(), persistent=False)

    def forward(self, x):  # x: [B, T, D]
        T = x.size(1)
        self._maybe_extend(T, x.device, x.dtype)
        return x + self.pe[:T].unsqueeze(0).to(device=x.device, dtype=x.dtype)


class PerceiverResampler(nn.Module):
    """
    Learned S latent queries that cross-attend to input tokens.
    Single cross-attn block + MLP, good enough for conditioning.
    """
    def __init__(self, d_model: int, n_heads: int, out_len: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(out_len, d_model) / math.sqrt(d_model))
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )
        # care with init
        nn.init.zeros_(self.attn.out_proj.weight)
        if self.attn.out_proj.bias is not None:
            nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x, key_padding_mask: torch.Tensor | None = None):  # x: [B, T, D]
        B, T, D = x.shape
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)  # [B, S, D]
        q = self.ln_q(lat)
        kv = self.ln_kv(x)
        # key_padding_mask expects True at positions that should be ignored
        z, _ = self.attn(q, kv, kv, need_weights=False, key_padding_mask=key_padding_mask)
        z = z + self.mlp(z)
        return z  # [B, S, D]


class FaceKeypointEncoder(nn.Module):
    """
    Spatial (landmarks) -> frame token, then temporal transformer -> tokens,
    then resample to S=cond_len with a Perceiver-like resampler.
    Input: [B, T, 478, 3]
    Output: [B, S, d_model]
    """
    def __init__(
        self,
        d_model: int,
        n_landmarks: int = 478,
        in_dim: int = 3,
        kp_embed_dim: int = 256,
        spatial_layers: int = 2,
        temporal_layers: int = 4,
        n_heads: int = 8,
        cond_len: int = 8,
        dropout: float = 0.0,
        ff_mult: float = 4.0,
        max_frames: int = 1024,
    ):
        super().__init__()
        self.n_landmarks = n_landmarks
        self.in_dim = in_dim

        # Per-landmark MLP to embed 3D coords
        self.kp_mlp = nn.Sequential(
            nn.Linear(in_dim, kp_embed_dim),
            nn.GELU(),
            nn.Linear(kp_embed_dim, kp_embed_dim),
        )
        self.landmark_emb = nn.Embedding(n_landmarks, kp_embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, kp_embed_dim))

        # Choose nhead for spatial encoder that divides kp_embed_dim
        def _divisible_heads(d: int, prefer: int) -> int:
            candidates = [h for h in range(1, min(d, prefer) + 1) if d % h == 0]
            return max(candidates) if candidates else 1

        nhead_sp = _divisible_heads(kp_embed_dim, n_heads)
        # Spatial encoder over 478 tokens (+CLS), per frame
        enc_layer_sp = nn.TransformerEncoderLayer(
            d_model=kp_embed_dim, nhead=nhead_sp,
            dim_feedforward=int(kp_embed_dim * ff_mult),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.spatial_encoder = nn.TransformerEncoder(enc_layer_sp, num_layers=spatial_layers)
        self.to_frame = nn.Linear(kp_embed_dim, d_model)   # map CLS to model dim

        # Temporal encoder over frames
        self.time_pe = SinusoidalPositionalEmbedding(d_model, max_len=max_frames)
        enc_layer_tmp = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=int(d_model * ff_mult),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(enc_layer_tmp, num_layers=temporal_layers)

        # Resampler to S tokens for cross-attn
        self.resampler = PerceiverResampler(d_model=d_model, n_heads=n_heads, out_len=cond_len, dropout=dropout)

        # Init
        nn.init.normal_(self.cls_token, std=0.02)
        # Cache landmark ids for speed
        self.register_buffer("lm_ids", torch.arange(n_landmarks).unsqueeze(0), persistent=False)

    @staticmethod
    def _normalize_per_frame(x, eps=1e-6):
        """
        x: [B, T, L, 3] -> subtract per-frame centroid, divide by per-frame std.
        Keeps scale roughly stable across faces.
        """
        mu = x.mean(dim=2, keepdim=True)          # [B,T,1,3]
        x = x - mu
        std = x.flatten(2).std(dim=-1, keepdim=True).unsqueeze(-1)  # [B,T,1,1]
        std = torch.clamp(std, min=1e-3)
        return x / (std + eps)

    def forward(self, x, frame_mask: torch.Tensor | None = None):  # x: [B, T, L=478, 3]
        B, T, L, C = x.shape
        assert L == self.n_landmarks and C == self.in_dim, "Bad keypoint shape"

        x = self._normalize_per_frame(x)

        # ---- Spatial encoder per frame
        # reshape to [B*T, L, C]
        x_bt = x.view(B * T, L, C)
        tok = self.kp_mlp(x_bt)                                   # [B*T, L, E]
        lm_ids = self.lm_ids.to(x.device).expand(B * T, -1)  # [B*T, L]
        tok = tok + self.landmark_emb(lm_ids)                      # landmark id embedding

        cls = self.cls_token.expand(B * T, -1, -1)                # [B*T, 1, E]
        tok = torch.cat([cls, tok], dim=1)                        # [B*T, 1+L, E]
        tok = self.spatial_encoder(tok)                           # [B*T, 1+L, E]
        frame_vec = tok[:, 0, :]                                  # [B*T, E]
        frame_vec = self.to_frame(frame_vec).view(B, T, -1)       # [B, T, d_model]

        # ---- Temporal encoder over frames
        h = self.time_pe(frame_vec)
        # Build key padding mask for temporal encoder if provided (True = pad)
        src_key_padding_mask = None
        if frame_mask is not None:
            assert frame_mask.shape[:2] == (B, T)
            src_key_padding_mask = ~frame_mask.to(dtype=torch.bool)
        h = self.temporal_encoder(h, src_key_padding_mask=src_key_padding_mask)  # [B, T, d_model]

        # ---- Resample to [B, S, d_model]
        cond = self.resampler(h, key_padding_mask=src_key_padding_mask)
        return cond  # [B, S, d_model]


class FaceKeypointConditionProvider(nn.Module):
    """
    Wraps FaceKeypointEncoder. Call with (batch, device, keypoints=None).
    If keypoints is None, fabricates zeros with T=default_frames so training still runs.
    """
    def __init__(
        self,
        d_model: int,
        cond_len: int,
        default_frames: int = 8,
        n_landmarks: int = 478,
        in_dim: int = 3,
        kp_embed_dim: int = 256,
        spatial_layers: int = 2,
        temporal_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.0,
        ff_mult: float = 4.0,
    ):
        super().__init__()
        self.encoder = FaceKeypointEncoder(
            d_model=d_model,
            n_landmarks=n_landmarks,
            in_dim=in_dim,
            kp_embed_dim=kp_embed_dim,
            spatial_layers=spatial_layers,
            temporal_layers=temporal_layers,
            n_heads=n_heads,
            cond_len=cond_len,
            dropout=dropout,
            ff_mult=ff_mult,
        )
        self.default_frames = default_frames
        self.n_landmarks = n_landmarks
        self.in_dim = in_dim

    @torch.no_grad()
    def _fabricate(self, batch, device):
        B = batch["input_ids"].size(0)
        return torch.zeros(B, self.default_frames, self.n_landmarks, self.in_dim, device=device)

    def forward(self, batch, device, keypoints: torch.Tensor | None = None, keypoint_mask: torch.Tensor | None = None):
        """
        keypoints: [B, T, 478, 3] (float). If None, zeros(T=default_frames) are used.
        Returns cond_tokens: [B, S, d_model]
        """
        if keypoints is None:
            keypoints = self._fabricate(batch, device)
        return self.encoder(keypoints, frame_mask=keypoint_mask)
