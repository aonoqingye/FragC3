import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CellLineContextTokenizer(nn.Module):
    """
    Convert a cell vector [B, ctx_in_dim] into Lc context tokens: [B, Lc, D].
    - Patched: allow Conv1d output length to deviate slightly; use AdaptiveAvgPool1d to align to Lc.
    """

    def __init__(self,
                 ctx_in_dim: int = 1722,
                 d_model: int = 300,
                 Lc: int = 128,
                 dropout: float = 0.0,
                 prefer_overlap: bool = False,
                 ):
        super().__init__()
        assert ctx_in_dim >= Lc and Lc >= 1, "Must satisfy ctx_in_dim >= Lc >= 1."

        self.ctx_in_dim = ctx_in_dim
        self.d_model = d_model
        self.Lc = Lc
        self.drop = nn.Dropout(dropout)

        # ---- Pre-Norm ----
        self.pre_ln = nn.LayerNorm(ctx_in_dim)

        # ---- Automatically choose kernel & stride ----
        # Minor mismatch is fine; an adaptive pooling layer will align the length later.
        k, s, p = self._choose_kernel_stride_padding(N=ctx_in_dim, Lc=Lc, prefer_overlap=prefer_overlap)

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=k, stride=s, padding=p, bias=True
        )

        # ---- Key fix: add an adaptive pooling layer ----
        # Regardless of whether Conv1d outputs 128/130/132, force it back to Lc.
        self.pool = nn.AdaptiveAvgPool1d(Lc)

        self.act = nn.GELU()
        self.post_ln = nn.LayerNorm(d_model)

        # ---- Initialization ----
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
        if self.conv.bias is not None:
            fan_in = self.conv.weight.shape[1] * self.conv.kernel_size[0]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.conv.bias, -bound, bound)

    @staticmethod
    def _choose_kernel_stride_padding(N: int, Lc: int, prefer_overlap: bool):
        """
        Compute parameters without strict asserts; best-effort to approach Lc.
        """
        s = max(1, N // Lc)
        if prefer_overlap:
            k = min(N, max(1, 2 * s))
        else:
            k = min(N, max(1, s))

            # Compute the required padding; if negative (stride too small), clamp to 0.
        # Target: (N + 2p - k) / s + 1 ≈ Lc
        # => 2p ≈ s(Lc - 1) - N + k
        numer = s * (Lc - 1) - N + k
        p = max(0, math.ceil(numer / 2))

        # Remove hard asserts and tolerate small deviations
        return k, s, p

    def forward(self, cell_vec: torch.Tensor) -> torch.Tensor:
        """
        cell_vec: [B, ctx_in_dim]
        return:   [B, Lc, d_model]
        """
        # 1. Pre-Norm
        x = self.pre_ln(cell_vec)  # [B, N]
        x = x.unsqueeze(1)  # [B, 1, N]

        # 2. Conv1d
        x = self.conv(x)  # [B, D, L_out] (L_out may differ from Lc)

        # 3. Fallback alignment (fix point)
        if x.shape[2] != self.Lc:
            x = self.pool(x)  # [B, D, Lc] force length to Lc

        x = self.act(x)
        x = self.drop(x)
        x = x.transpose(1, 2).contiguous()  # [B, Lc, D]
        x = self.post_ln(x)
        return x

class ContextLinearTokenizer(nn.Module):
    def __init__(self, ctx_in_dim, d_model, dropout=0.0):
        super().__init__()
        self.pre_ln = nn.LayerNorm(ctx_in_dim)
        self.proj = nn.Linear(ctx_in_dim, d_model)
        self.drop = nn.Dropout(dropout)
        self.post_ln = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, cell_vec):
        # [B, N] → [B, D]
        x = self.pre_ln(cell_vec)
        x = self.proj(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.post_ln(x)

        return x.unsqueeze(1)   # ✅ [B, 1, D]


class TriEinsumAttention(nn.Module):
    """
    Tri-Attention: TSDP scoring + three context–value fusion modes (multi-head).
    Q: [B, La, D], K: [B, Lb, D], V: [B, Lb, D], C_tokens: [B, Lc, D]
    attn_mask_flat: [B, La, Lb] or [B, La, Lb*Lc] (bool, True=masked).

    context–value fusion mode (cv_mode):
        - "mul": Eq.(17) Multiplicative, v_c(i,j) = v_i ⊙ c_j
        - "add": Eq.(16) Additive,      v_c(i,j) = v_i + c_j
        - "bilinear": Eq.(18) Bilinear, v_c(i,j) = (U^T v_i) ⊙ (H^T c_j)
    """
    def __init__(self, d_model: int, num_heads: int, attn_dropout: float = 0.0, cv_mode: str = "mul"):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.dh = d_model // num_heads
        self.d = d_model

        cv_mode = cv_mode.lower()
        assert cv_mode in ("mul", "add", "bilinear"), f"Unknown cv_mode: {cv_mode}"
        self.cv_mode = cv_mode
        # Only used in bilinear mode, corresponding to U and H in Eq.(18).
        if self.cv_mode == "bilinear":
            self.v_cv = nn.Linear(d_model, d_model, bias=False)
            self.c_cv = nn.Linear(d_model, d_model, bias=False)
        else:
            self.v_cv = None
            self.c_cv = None

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_dropout)

    @staticmethod
    def _split_heads(x, h):
        # [B,L,D] -> [B,H,L,Dh]
        B, L, D = x.shape
        Dh = D // h
        return x.view(B, L, h, Dh).permute(0, 2, 1, 3).contiguous()

    def forward(self, Q_in, K_in, V_in, C_tokens, attn_mask_flat=None):
        B, La, D = Q_in.shape
        Lb = K_in.size(1)
        Lc = C_tokens.size(1)

        # 1) Linear projections + split into heads
        Q = self._split_heads(self.q_proj(Q_in), self.h)     # [B,H,La,Dh]
        K = self._split_heads(self.k_proj(K_in), self.h)     # [B,H,Lb,Dh]
        V = self._split_heads(self.v_proj(V_in), self.h)     # [B,H,Lb,Dh]
        C = self._split_heads(C_tokens, self.h)              # [B,H,Lc,Dh]

        # 2) Ternary TSDP scoring: scores[b,h,q,k,c] = Σ_d Q*K*C / sqrt(dh)
        scores = torch.einsum('bhqd,bhkd,bhcd->bhqkc', Q, K, C) / math.sqrt(self.dh)  # [B,H,La,Lb,Lc]
        scores = scores.reshape(B, self.h, La, Lb * Lc)                                # [B,H,La,Lb*Lc]

        # 3) Flatten the mask over (k, c)
        if attn_mask_flat is not None:
            if attn_mask_flat.shape[-1] == Lb:
                attn_mask_flat = attn_mask_flat.repeat_interleave(Lc, dim=-1)          # [B,La,Lb*Lc]
            scores = scores.masked_fill(attn_mask_flat.unsqueeze(1), float('-inf'))    # [B,1,La,Lb*Lc]

        # 4) Safe softmax: handle rows that are entirely -inf
        all_masked = torch.isinf(scores).all(dim=-1, keepdim=True)  # [B,H,La,1]
        scores = torch.where(all_masked, torch.zeros_like(scores), scores)
        attn = torch.softmax(scores, dim=-1)                        # [B,H,La,Lb*Lc]
        attn = torch.where(all_masked, torch.zeros_like(attn), attn)

        attn = self.attn_drop(attn)

        # 5) Context–value fusion to form V^c[b,h,k,c,d]
        if self.cv_mode == "bilinear":
            # Eq.(18): v_c(i,j) = (U^T v_i) ⊙ (H^T c_j)
            V_for_cv = self._split_heads(self.v_cv(V_in), self.h)        # [B,H,Lb,Dh]
            C_for_cv = self._split_heads(self.c_cv(C_tokens), self.h)    # [B,H,Lc,Dh]
        else:
            V_for_cv = V
            C_for_cv = C

        if self.cv_mode in ("mul", "bilinear"):
            # Multiplicative / Bilinear: Hadamard product
            VC = torch.einsum('bhkd,bhcd->bhkcd', V_for_cv, C_for_cv)    # [B,H,Lb,Lc,Dh]
        elif self.cv_mode == "add":
            # Additive: v_c(i,j) = v_i + c_j
            V_exp = V_for_cv.unsqueeze(3)                                # [B,H,Lb,1,Dh]
            C_exp = C_for_cv.unsqueeze(2)                                # [B,H,1,Lc,Dh]
            VC = V_exp + C_exp                                           # [B,H,Lb,Lc,Dh]
        else:
            raise RuntimeError(f"Unsupported cv_mode at runtime: {self.cv_mode}")

        # 6) Aggregate to outputs
        attn_2d = attn.view(B, self.h, La, Lb, Lc)                       # [B,H,La,Lb,Lc]
        out = torch.einsum('bhqkc,bhkcd->bhqd', attn_2d, VC)             # [B,H,La,Dh]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, La, self.d)   # [B,La,D]
        return out, attn

class TriAddAttention(nn.Module):
    """
    Tri-Attention: TAdd scoring + Hadamard value fusion (multi-head).
    F(q, k, c) = p^T tanh(Wq q + Wk k + Wc c)
    """
    def __init__(self, d_model: int, num_heads: int, attn_dropout: float = 0.0, cv_mode: str = "mul"):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.dh = d_model // num_heads
        self.d = d_model

        cv_mode = cv_mode.lower()
        assert cv_mode in ("mul", "add", "bilinear"), f"Unknown cv_mode: {cv_mode}"
        self.cv_mode = cv_mode
        # Only used in bilinear mode, corresponding to U and H in Eq.(18).
        if self.cv_mode == "bilinear":
            self.v_cv = nn.Linear(d_model, d_model, bias=False)
            self.c_cv = nn.Linear(d_model, d_model, bias=False)
        else:
            self.v_cv = None
            self.c_cv = None

        # Project to d_model and then split into heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.c_proj = nn.Linear(d_model, d_model)

        # The "p" vector in TAdd, split per head
        self.p = nn.Parameter(torch.zeros(self.h, self.dh))
        nn.init.xavier_uniform_(self.p)

        self.attn_drop = nn.Dropout(attn_dropout)

    @staticmethod
    def _split_heads(x, h):
        # [B,L,D] -> [B,H,L,Dh]
        B, L, D = x.shape
        Dh = D // h
        return x.view(B, L, h, Dh).permute(0, 2, 1, 3).contiguous()

    def forward(self, Q_in, K_in, V_in, C_tokens, attn_mask_flat=None):
        """
        Q_in: [B, La, D]
        K_in,V_in: [B, Lb, D]
        C_tokens: [B, Lc, D]
        attn_mask_flat: [B, La, Lb] or [B, La, Lb*Lc]
        """
        B, La, D = Q_in.shape
        Lb = K_in.size(1)
        Lc = C_tokens.size(1)

        # Linear projections + split into heads
        Q = self._split_heads(self.q_proj(Q_in), self.h)       # [B,H,La,Dh]
        K = self._split_heads(self.k_proj(K_in), self.h)       # [B,H,Lb,Dh]
        V = self._split_heads(self.v_proj(V_in), self.h)       # [B,H,Lb,Dh]
        C = self._split_heads(self.c_proj(C_tokens), self.h)   # [B,H,Lc,Dh]

        # TAdd：p^T tanh(Wq q + Wk k + Wc c)
        # Broadcast by expanding dims to get [B,H,La,Lb,Lc,Dh]
        Qe = Q.unsqueeze(3).unsqueeze(4)   # [B,H,La,1,1,Dh]
        Ke = K.unsqueeze(2).unsqueeze(4)   # [B,H,1,Lb,1,Dh]
        Ce = C.unsqueeze(2).unsqueeze(3)   # [B,H,1,1,Lc,Dh]

        x = torch.tanh(Qe + Ke + Ce)       # [B,H,La,Lb,Lc,Dh]
        p = self.p.view(1, self.h, 1, 1, 1, self.dh)
        scores = (x * p).sum(-1)           # [B,H,La,Lb,Lc]
        scores = scores.view(B, self.h, La, Lb * Lc)  # [B,H,La,Lb*Lc]

        # Flatten the mask over (k, c)
        if attn_mask_flat is not None:
            if attn_mask_flat.shape[-1] == Lb:
                attn_mask_flat = attn_mask_flat.repeat_interleave(Lc, dim=-1)  # [B,La,Lb*Lc]
            scores = scores.masked_fill(attn_mask_flat.unsqueeze(1), float('-inf'))

        # Safe softmax
        all_masked = torch.isinf(scores).all(dim=-1, keepdim=True)  # [B,H,La,1]
        scores = torch.where(all_masked, torch.zeros_like(scores), scores)
        attn = torch.softmax(scores, dim=-1)                        # [B,H,La,Lb*Lc]
        attn = torch.where(all_masked, torch.zeros_like(attn), attn)

        attn = self.attn_drop(attn)

        # 5) Context–value fusion to form V^c[b,h,k,c,d]
        if self.cv_mode == "bilinear":
            # Eq.(18): v_c(i,j) = (U^T v_i) ⊙ (H^T c_j)
            V_for_cv = self._split_heads(self.v_cv(V_in), self.h)  # [B,H,Lb,Dh]
            C_for_cv = self._split_heads(self.c_cv(C_tokens), self.h)  # [B,H,Lc,Dh]
        else:
            V_for_cv = V
            C_for_cv = C

        if self.cv_mode in ("mul", "bilinear"):
            # Multiplicative / Bilinear: Hadamard product
            VC = torch.einsum('bhkd,bhcd->bhkcd', V_for_cv, C_for_cv)  # [B,H,Lb,Lc,Dh]
        elif self.cv_mode == "add":
            # Additive: v_c(i,j) = v_i + c_j
            V_exp = V_for_cv.unsqueeze(3)  # [B,H,Lb,1,Dh]
            C_exp = C_for_cv.unsqueeze(2)  # [B,H,1,Lc,Dh]
            VC = V_exp + C_exp  # [B,H,Lb,Lc,Dh]
        else:
            raise RuntimeError(f"Unsupported cv_mode at runtime: {self.cv_mode}")

        attn_2d = attn.view(B, self.h, La, Lb, Lc)   # [B,H,La,Lb,Lc]
        out = torch.einsum('bhqkc,bhkcd->bhqd', attn_2d, VC)  # [B,H,La,Dh]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, La, self.d)  # [B,La,D]
        return out, attn


class TriDotProductAttention(nn.Module):
    """
    Tri-Attention: TDP (ternary dot-product) + Hadamard value fusion.
    Same as TriEinsumAttention, but without the sqrt(dh) scaling.
    """
    def __init__(self, d_model: int, num_heads: int, attn_dropout: float = 0.0, cv_mode: str = "mul"):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.dh = d_model // num_heads
        self.d = d_model

        cv_mode = cv_mode.lower()
        assert cv_mode in ("mul", "add", "bilinear"), f"Unknown cv_mode: {cv_mode}"
        self.cv_mode = cv_mode
        # Only used in bilinear mode, corresponding to U and H in Eq.(18).
        if self.cv_mode == "bilinear":
            self.v_cv = nn.Linear(d_model, d_model, bias=False)
            self.c_cv = nn.Linear(d_model, d_model, bias=False)
        else:
            self.v_cv = None
            self.c_cv = None

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_dropout)

    @staticmethod
    def _split_heads(x, h):
        B, L, D = x.shape
        Dh = D // h
        return x.view(B, L, h, Dh).permute(0, 2, 1, 3).contiguous()

    def forward(self, Q_in, K_in, V_in, C_tokens, attn_mask_flat=None):
        B, La, D = Q_in.shape
        Lb = K_in.size(1)
        Lc = C_tokens.size(1)

        Q = self._split_heads(self.q_proj(Q_in), self.h)  # [B,H,La,Dh]
        K = self._split_heads(self.k_proj(K_in), self.h)  # [B,H,Lb,Dh]
        V = self._split_heads(self.v_proj(V_in), self.h)  # [B,H,Lb,Dh]
        C = self._split_heads(C_tokens, self.h)           # [B,H,Lc,Dh]

        # Ternary dot-product (no scaling)
        scores = torch.einsum('bhqd,bhkd,bhcd->bhqkc', Q, K, C)  # [B,H,La,Lb,Lc]
        scores = scores.reshape(B, self.h, La, Lb * Lc)          # [B,H,La,Lb*Lc]

        if attn_mask_flat is not None:
            if attn_mask_flat.shape[-1] == Lb:
                attn_mask_flat = attn_mask_flat.repeat_interleave(Lc, dim=-1)
            scores = scores.masked_fill(attn_mask_flat.unsqueeze(1), float('-inf'))

        all_masked = torch.isinf(scores).all(dim=-1, keepdim=True)
        scores = torch.where(all_masked, torch.zeros_like(scores), scores)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(all_masked, torch.zeros_like(attn), attn)

        attn = self.attn_drop(attn)

        # 5) Context–value fusion to form V^c[b,h,k,c,d]
        if self.cv_mode == "bilinear":
            # Eq.(18): v_c(i,j) = (U^T v_i) ⊙ (H^T c_j)
            V_for_cv = self._split_heads(self.v_cv(V_in), self.h)  # [B,H,Lb,Dh]
            C_for_cv = self._split_heads(self.c_cv(C_tokens), self.h)  # [B,H,Lc,Dh]
        else:
            V_for_cv = V
            C_for_cv = C

        if self.cv_mode in ("mul", "bilinear"):
            # Multiplicative / Bilinear: Hadamard product
            VC = torch.einsum('bhkd,bhcd->bhkcd', V_for_cv, C_for_cv)  # [B,H,Lb,Lc,Dh]
        elif self.cv_mode == "add":
            # Additive: v_c(i,j) = v_i + c_j
            V_exp = V_for_cv.unsqueeze(3)  # [B,H,Lb,1,Dh]
            C_exp = C_for_cv.unsqueeze(2)  # [B,H,1,Lc,Dh]
            VC = V_exp + C_exp  # [B,H,Lb,Lc,Dh]
        else:
            raise RuntimeError(f"Unsupported cv_mode at runtime: {self.cv_mode}")

        attn_2d = attn.view(B, self.h, La, Lb, Lc)
        out = torch.einsum('bhqkc,bhkcd->bhqd', attn_2d, VC)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, La, self.d)
        return out, attn


class TriTrilinearAttention(nn.Module):
    """
    Tri-Attention: Trilinear (efficient form of Eq.(14)) + Hadamard value fusion.
    Apply linear transforms to Q/K/C first, then compute the ternary dot-product.
    """
    def __init__(self, d_model: int, num_heads: int, attn_dropout: float = 0.0, cv_mode: str = "mul"):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.dh = d_model // num_heads
        self.d = d_model

        cv_mode = cv_mode.lower()
        assert cv_mode in ("mul", "add", "bilinear"), f"Unknown cv_mode: {cv_mode}"
        self.cv_mode = cv_mode
        # Only used in bilinear mode, corresponding to U and H in Eq.(18).
        if self.cv_mode == "bilinear":
            self.v_cv = nn.Linear(d_model, d_model, bias=False)
            self.c_cv = nn.Linear(d_model, d_model, bias=False)
        else:
            self.v_cv = None
            self.c_cv = None

        # Efficient Trilinear parameters W, U, H (project to the same space)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wc = nn.Linear(d_model, d_model, bias=False)

        self.v_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_dropout)

    @staticmethod
    def _split_heads(x, h):
        B, L, D = x.shape
        Dh = D // h
        return x.view(B, L, h, Dh).permute(0, 2, 1, 3).contiguous()

    def forward(self, Q_in, K_in, V_in, C_tokens, attn_mask_flat=None):
        B, La, D = Q_in.shape
        Lb = K_in.size(1)
        Lc = C_tokens.size(1)

        # Wq q, Uk k, Hc c
        Q_t = self._split_heads(self.Wq(Q_in), self.h)        # [B,H,La,Dh]
        K_t = self._split_heads(self.Wk(K_in), self.h)        # [B,H,Lb,Dh]
        C_t = self._split_heads(self.Wc(C_tokens), self.h)    # [B,H,Lc,Dh]

        V = self._split_heads(self.v_proj(V_in), self.h)      # [B,H,Lb,Dh]

        # Trilinear: <Wq q, Uk k, Hc c>
        scores = torch.einsum('bhqd,bhkd,bhcd->bhqkc', Q_t, K_t, C_t)  # [B,H,La,Lb,Lc]
        scores = scores.reshape(B, self.h, La, Lb * Lc)

        if attn_mask_flat is not None:
            if attn_mask_flat.shape[-1] == Lb:
                attn_mask_flat = attn_mask_flat.repeat_interleave(Lc, dim=-1)
            scores = scores.masked_fill(attn_mask_flat.unsqueeze(1), float('-inf'))

        all_masked = torch.isinf(scores).all(dim=-1, keepdim=True)
        scores = torch.where(all_masked, torch.zeros_like(scores), scores)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(all_masked, torch.zeros_like(attn), attn)

        attn = self.attn_drop(attn)

        # 5) Context–value fusion to form V^c[b,h,k,c,d]
        if self.cv_mode == "bilinear":
            # Eq.(18): v_c(i,j) = (U^T v_i) ⊙ (H^T c_j)
            V_for_cv = self._split_heads(self.v_cv(V_in), self.h)  # [B,H,Lb,Dh]
            C_for_cv = self._split_heads(self.c_cv(C_tokens), self.h)  # [B,H,Lc,Dh]
        else:
            V_for_cv = V
            C_for_cv = C_t

        if self.cv_mode in ("mul", "bilinear"):
            # Multiplicative / Bilinear: Hadamard product
            VC = torch.einsum('bhkd,bhcd->bhkcd', V_for_cv, C_for_cv)  # [B,H,Lb,Lc,Dh]
        elif self.cv_mode == "add":
            # Additive: v_c(i,j) = v_i + c_j
            V_exp = V_for_cv.unsqueeze(3)  # [B,H,Lb,1,Dh]
            C_exp = C_for_cv.unsqueeze(2)  # [B,H,1,Lc,Dh]
            VC = V_exp + C_exp  # [B,H,Lb,Lc,Dh]
        else:
            raise RuntimeError(f"Unsupported cv_mode at runtime: {self.cv_mode}")

        attn_2d = attn.view(B, self.h, La, Lb, Lc)
        out = torch.einsum('bhqkc,bhkcd->bhqd', attn_2d, VC)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, La, self.d)
        return out, attn
