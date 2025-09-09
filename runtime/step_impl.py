from typing import List, Tuple, Optional, Dict, Any
import torch
from torch import Tensor
from dataclasses import dataclass


@dataclass
class DecodeRequest:
    req_id: int
    bos_id: int
    eos_id: int
    max_len: int
    enc: "EncState"
    meta: Dict[str, Any]
    sid: Optional[int]
    tokens: List[int]
    finished: bool
    step: int


@dataclass
class EncState:
    extra: Dict[str, Any]


class Stepper:
    """
    连续前缀 + 原地写回（零拷贝）：
    - get_block_layer(layer_idx, B) 直接给 [B, T, H, D] 视图
    - flash_attn_with_kvcache 在该视图上就地写入新 K/V
    - 每层 bump_lengths 同步长度
    """
    def __init__(self, decoder: "TransformerDecoder", kv, device: Optional[str] = None):
        self.dec = decoder
        self.kv = kv
        self.device = (
            device 
            or getattr(getattr(kv, "spec", None), "device", None) 
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.attn_dtype = decoder.attn_dtype
        self.n_layers = decoder.n_layers
        self.n_head = decoder.n_head
        self.d_model = decoder.d_model
        self.d_k = decoder.d_k
        self._packed_cross: Dict[int, Dict[str, Any]] = {}  # per-layer 打包缓存
        self._rotary = None
        self._cap_cached = None
        self._fn_bump_lengths = self._pick(
            ["bump_lengths", "increment_lengths", "advance_lengths", "incr_lengths", "bump"]
        )

    def _get_packed_cross(self, layer_idx: int, batch_reqs: List["DecodeRequest"], device: torch.device):
        """
        返回（k_cat, v_cat, cu_k, max_seqlen_k），若本步批次未变则复用缓存。
        缓存键：按当前批次的 sid 顺序组合的 tuple。
        """
        # 以 sid 顺序作为稳定 key（也可用 req_id，但 sid 顺序更贴近 varlen 拼接需求）
        # order_key = tuple(int(r.sid) for r in batch_reqs)
        order_key = tuple(int(r.req_id) for r in batch_reqs)
        cache = self._packed_cross.get(layer_idx)
        if cache is not None and cache.get("order_key") == order_key:
            return cache["k_cat"], cache["v_cat"], cache["cu_k"], cache["max_k"]
        
        # 重建一次（只有当批次变动或顺序变化时才会走到这里）
        k_parts, v_parts, lens = [], [], []
        for r in batch_reqs:
            ext = r.enc.extra
            k_i = ext["cross_k_unpad_layers"][layer_idx]  # [L_i, H, D] (已是 attn_dtype, contiguous)
            v_i = ext["cross_v_unpad_layers"][layer_idx]
            k_parts.append(k_i)
            v_parts.append(v_i)
            lens.append(k_i.size(0))
        
        if lens:
            # 仅此处做一次 cat；后续多步复用
            k_cat = torch.cat(k_parts, dim=0)  # [(ΣL_i), H, D]
            v_cat = torch.cat(v_parts, dim=0)
            lens_t = torch.tensor(lens, device=device, dtype=torch.int32)
            cu_k = torch.zeros(len(lens) + 1, device=device, dtype=torch.int32)
            cu_k[1:] = torch.cumsum(lens_t, dim=0)
            max_k = int(lens_t.max().item())
        else:
            k_cat = torch.empty(0, self.n_head, self.d_k, device=device, dtype=self.attn_dtype)
            v_cat = torch.empty_like(k_cat)
            cu_k = torch.zeros(1, device=device, dtype=torch.int32)
            max_k = 0
        
        self._packed_cross[layer_idx] = {
            "order_key": order_key,
            "k_cat": k_cat,
            "v_cat": v_cat,
            "cu_k": cu_k,
            "max_k": max_k,
        }
        return k_cat, v_cat, cu_k, max_k

    def _pick(self, names):
        for n in names:
            fn = getattr(self.kv, n, None)
            if callable(fn):
                return fn
        
        def _missing(*args, **kwargs):
            raise AttributeError(f"KV manager missing any of: {names}")
        return _missing

    def _maybe_init_rotary(self, cap: int):
        if self._rotary is None or self._cap_cached != cap:
            cos, sin = self.dec.get_rotary_encoding(cap, device=self.device, dtype=self.attn_dtype)
            self._rotary = (cos, sin)
            self._cap_cached = cap

    @torch.inference_mode()
    def __call__(
        self,
        y_prev: Tensor,  # [B]
        batch_reqs: List[DecodeRequest],
        _K_ignored: Tensor,
        _V_ignored: Tensor,
        _L_ignored: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        B = y_prev.size(0)
        device = self.device
        
        # 预生成 rotary
        cap = getattr(getattr(self.kv, "spec", None), "cap", None)
        if cap is not None:
            self._maybe_init_rotary(cap)
        
        sids = torch.tensor([r.sid for r in batch_reqs], device=device, dtype=torch.long)
        fast_expected = torch.arange(B, device=device, dtype=torch.long)
        can_fast = hasattr(self.kv, "get_block_layer") and torch.equal(sids, fast_expected)
        
        # 第0层长度（用于位置）
        try:
            if can_fast:
                K0, V0, L0 = self.kv.get_block_layer(0, B)  # [B,T,H,D], [B]
            else:
                K0, V0, L0 = self.kv.get_batch_fixedcap_layer(0, sids)  # [B,T,H,D], [B]
        except AttributeError:
            K0, V0, L0 = self.kv.get_batch_fixedcap(sids, layer_idx=0)
        
        if cap is None:
            self._maybe_init_rotary(K0.size(1))  # T/CAP
        
        # 位置索引
        pos_ids = L0.to(dtype=torch.long)  # [B]
        pe_buf: Tensor = self.dec.positional_encoding.pe.to(device)  # [1, max_len, d_model]
        pos_ids = pos_ids.clamp(min=0, max=pe_buf.size(1) - 1)
        
        # token embedding + abs PE
        y_prev = y_prev.to(device)
        emb = self.dec.tgt_word_emb(y_prev) * (self.d_model ** 0.5)  # [B, d_model]
        pos_emb = pe_buf.index_select(1, pos_ids).squeeze(0)  # [B, d_model]
        x = self.dec.dropout((emb + pos_emb).unsqueeze(1))  # [B,1,d_model]
        
        rotary_cos, rotary_sin = self._rotary  # [cap, d_k]
        
        # 逐层
        for layer_idx, layer in enumerate(self.dec.layer_stack):
            if can_fast:
                Kc, Vc, Lc = self.kv.get_block_layer(layer_idx, B)  # [B,T,H,D], [B]
            else:
                Kc, Vc, Lc = self.kv.get_batch_fixedcap_layer(layer_idx, sids)  # [B,T,H,D], [B]
            
            # q/k/v
            x_norm = layer.self_attn_norm(x)
            q = layer.self_attn.w_qs(x_norm).to(self.attn_dtype).view(B, 1, self.n_head, self.d_k)
            k = layer.self_attn.w_ks(x_norm).to(self.attn_dtype).view(B, 1, self.n_head, self.d_k)
            v = layer.self_attn.w_vs(x_norm).to(self.attn_dtype).view(B, 1, self.n_head, self.d_k)
            
            # 直接传 [B,T,H,D]（与内核一致，且是池子的原地视图）
            out = _flash_attn_with_kvcache(
                q=q,
                k_cache=Kc,
                v_cache=Vc,
                k=k,
                v=v,
                rotary_cos=rotary_cos,
                rotary_sin=rotary_sin,
                cache_seqlens=Lc,  # int32
                softmax_scale=layer.self_attn.softmax_scale,
                causal=True,
            )
            
            out = out.to(layer.self_attn.fc.weight.dtype).view(B, 1, self.d_model)
            out = layer.self_attn.fc(out)
            out = layer.self_attn.dropout(out)
            x = x + out
            
            # 同步长度
            self._fn_bump_lengths(layer_idx, fast_expected if can_fast else sids, 1)
            
            # cross-attn varlen（与原实现一致）
            # 一次性/按需 打包 cross-attn K/V；批次/顺序不变则复用，避免每步 cat
            enc_k_cat, enc_v_cat, cu_k, max_seqlen_k = self._get_packed_cross(layer_idx, batch_reqs, device)
            
            # 每步的 cu_q 仍然简单构造（查询长度恒为 1）
            cu_q = torch.arange(0, B + 1, device=device, dtype=torch.int32)
            
            x_cross = layer.cross_attn_norm(x)
            x_cross = layer.cross_attn.varlen_cross_attention(
                q=x_cross,
                k_unpad=enc_k_cat,
                v_unpad=enc_v_cat,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=1,
                max_seqlen_k=max_seqlen_k,
                effective_batch_size=B,
            )
            x = x + x_cross
            x = layer.ffn_step(x)
        
        x = self.dec.layer_norm_out(x)
        logits = self.dec.tgt_word_prj(x[:, 0])  # [B, Vocab]
        return logits, torch.empty(0, device=device), torch.empty(0, device=device)


# ---- FlashAttention with kvcache 薄封装 ----
def _flash_attn_with_kvcache(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    k: Tensor,
    v: Tensor,
    rotary_cos: Tensor,
    rotary_sin: Tensor,
    cache_seqlens: Tensor,
    softmax_scale: float,
    causal: bool = True,
) -> Tensor:
    from flash_attn import flash_attn_with_kvcache as _fa_kvcache
    out = _fa_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k,
        v=v,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        cache_seqlens=cache_seqlens,
        softmax_scale=softmax_scale,
        causal=causal,
        rotary_interleaved=True,
    )
    return out


# ---- 编码侧打包（每层 unpadded K/V） ----
@torch.inference_mode()
def build_enc_state_for_request(
    decoder: "TransformerDecoder",
    enc_outputs: torch.Tensor,  # [1, Ti, d_model]
    src_mask: torch.Tensor,     # [1, Ti] 或 [1, 1, Ti]
) -> EncState:
    assert enc_outputs.dim() == 3 and enc_outputs.size(0) == 1, \
        f"enc_outputs 期望 [1, Ti, d_model]，得到 {tuple(enc_outputs.shape)}"
    
    attn_dtype = decoder.attn_dtype
    H, D = decoder.n_head, decoder.d_k
    
    if src_mask.dim() == 3:
        src_mask = src_mask[:, 0, :]
    src_mask = src_mask.to(torch.bool)
    
    Ti = enc_outputs.size(1)
    L = int(src_mask[0].sum().item()) if Ti > 0 else 0
    xs = enc_outputs[:, :L, :]
    
    cross_k_unpad_layers: List[torch.Tensor] = []
    cross_v_unpad_layers: List[torch.Tensor] = []
    
    for layer in decoder.layer_stack:
        k_proj = layer.cross_attn.w_ks(xs)  # [1, L, H*D]
        v_proj = layer.cross_attn.w_vs(xs)  # [1, L, H*D]
        
        k_unpad = k_proj.view(1, L, H, D).squeeze(0).to(attn_dtype).contiguous()  # [L,H,D]
        v_unpad = v_proj.view(1, L, H, D).squeeze(0).to(attn_dtype).contiguous()
        
        cross_k_unpad_layers.append(k_unpad)
        cross_v_unpad_layers.append(v_unpad)
    
    extra = {
        "cross_k_unpad_layers": cross_k_unpad_layers,
        "cross_v_unpad_layers": cross_v_unpad_layers,
        "max_seqlen_k": L,
    }
    return EncState(extra=extra)