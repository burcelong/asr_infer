# ===== runtime/kv_adapter.py =====
from typing import Optional, Tuple
import torch
from dataclasses import dataclass

# 仅复用 KVSpec 的字段定义；也可直接在此重写 KVSpec
@dataclass
class KVSpec:
    slots: int
    n_heads: int
    cap: int
    head_dim: int
    dtype: torch.dtype = torch.float16
    device: str = "cuda"

class LayeredKVCacheManager:
    """
    多层 KV 池（连续前缀 + 原地写回）：
      - 每层各自分配：k_pool/v_pool 形状 [SLOTS, CAP, H, D] ；与 flash_attn_with_kvcache 要求的 [B, T, H, D] 完全一致
      - 活跃序列固定占行 0..B-1；新请求 -> 行 B；完成 -> 与最后一行 swap 填洞
      - get_block_layer(layer_idx, B) 返回 [B, CAP, H, D]、[B] 的**视图（非拷贝）**
      - 内核直接在此视图上写回；不做 permute、不做 contiguous
    """
    def __init__(self, spec: KVSpec, n_layers: int):
        self.spec = spec
        self.n_layers = n_layers
        S, C, H, D = spec.slots, spec.cap, spec.n_heads, spec.head_dim
        dev, dt = spec.device, spec.dtype

        self.k_pools = [torch.empty(S, C, H, D, device=dev, dtype=dt) for _ in range(n_layers)]
        self.v_pools = [torch.empty(S, C, H, D, device=dev, dtype=dt) for _ in range(n_layers)]
        self.lengths  = [torch.zeros(S, dtype=torch.int32, device=dev) for _ in range(n_layers)]

        self.active_count = 0
        self.max_slots = S

    # ---------- 连续前缀分配/回收 ----------
    @torch.no_grad()
    def alloc(self) -> int:
        if self.active_count >= self.max_slots:
            raise RuntimeError("LayeredKVCacheManager: no free slots")
        sid = self.active_count
        for L in self.lengths:
            L[sid] = 0
        self.active_count += 1
        return sid

    @torch.no_grad()
    def free(self, sid: int) -> Optional[Tuple[int, int]]:
        """
        释放一行；若 sid 不是最后一行，把最后一行 swap 到 sid。
        返回:
          - None: sid 已是最后一行（无交换）
          - (moved_sid, dst_sid): 原最后一行 moved_sid 被移到 dst_sid(=sid)
        """
        last = self.active_count - 1
        if sid < 0 or sid > last:
            raise RuntimeError(f"free invalid sid={sid}, active_count={self.active_count}")
        moved = None
        if sid != last:
            for l in range(self.n_layers):
                self.k_pools[l][[sid, last]] = self.k_pools[l][[last, sid]]
                self.v_pools[l][[sid, last]] = self.v_pools[l][[last, sid]]
                self.lengths[l][[sid, last]] = self.lengths[l][[last, sid]]
            moved = (last, sid)
        for l in range(self.n_layers):
            self.lengths[l][last] = 0
        self.active_count -= 1
        return moved

    # ---------- 快路径：块视图（零拷贝） ----------
    @torch.no_grad()
    def get_block_layer(self, layer_idx: int, B: int):
        if B > self.active_count:
            raise RuntimeError(f"get_block_layer B={B} > active_count={self.active_count}")
        K = self.k_pools[layer_idx].narrow(0, 0, B)  # [B, CAP, H, D] 视图
        V = self.v_pools[layer_idx].narrow(0, 0, B)
        L = self.lengths[layer_idx].narrow(0, 0, B)  # [B] 视图
        return K, V, L

    # ---------- 慢路径：任意顺序的选择（小拷贝） ----------
    @torch.no_grad()
    def get_batch_fixedcap_layer(self, layer_idx: int, active_sids: torch.Tensor):
        K = self.k_pools[layer_idx].index_select(0, active_sids)  # [B, CAP, H, D]
        V = self.v_pools[layer_idx].index_select(0, active_sids)
        L = self.lengths[layer_idx].index_select(0, active_sids)
        return K, V, L

    @torch.no_grad()
    def get_batch_fixedcap(self, active_sids: torch.Tensor, layer_idx: int = 0):
        return self.get_batch_fixedcap_layer(layer_idx, active_sids)

    # ---------- 长度自增 ----------
    @torch.no_grad()
    def bump_lengths(self, layer_idx: int, sids: torch.Tensor, delta: int = 1):
        # 写长度很轻，直接小循环最稳
        L = self.lengths[layer_idx]
        for sid in sids.tolist():
            L[sid] += int(delta)
