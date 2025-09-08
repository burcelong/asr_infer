# ===== kv_cache_manager.py =====
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch

@dataclass
class KVSpec:
    slots: int          # 同时活跃的最大序列数（SLOTS）
    n_heads: int        # 注意力头数 H
    cap: int            # 每序列最大缓存长度（CAP），可按你模型的 max_decode_len 设得足够大
    head_dim: int       # 每头维度 D
    dtype: torch.dtype = torch.float16
    device: str = "cuda"

class KVCacheManager:
    """
    固定形状的自注意力 KV 池：
      - big_self_k/v 的底层形状固定为 [SLOTS, H, CAP, D]
      - 用 lengths[sid] 记录每条序列当前有效长度
      - 通过 active_sids (逻辑 batch) 在每步聚合出 [B, H, CAP, D] 的视图喂给内核
    """
    def __init__(self, spec: KVSpec):
        self.spec = spec
        S, H, C, D = spec.slots, spec.n_heads, spec.cap, spec.head_dim
        self.k_pool = torch.empty(S, H, C, D, dtype=spec.dtype, device=spec.device)
        self.v_pool = torch.empty_like(self.k_pool)
        self.lengths = torch.zeros(S, dtype=torch.int32, device=spec.device)  # 每序列已用长度
        # 槽位管理（你也可以外部管理，这里给最简单实现）
        self.free_slots = list(range(S))
        self.alive = set()  # 当前占用的 sid

    # -------- 槽位分配/回收 --------
    def alloc(self) -> int:
        if not self.free_slots:
            raise RuntimeError("KVCacheManager: no free slots")
        sid = self.free_slots.pop()
        self.lengths[sid] = 0
        self.alive.add(sid)
        return sid

    def free(self, sid: int):
        self.lengths[sid] = 0
        self.alive.discard(sid)
        self.free_slots.append(sid)

    # -------- 读：聚合当前活跃 batch 的 K/V 与长度 --------
    @torch.no_grad()
    def get_batch_fixedcap(self, active_sids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回：
          K_step: [B, H, CAP, D]
          V_step: [B, H, CAP, D]
          L_step: [B]  int32，有效长度
        注意：底层池形状不变，这里只是按行选择（必要时 contiguous）
        """
        K = self.k_pool.index_select(0, active_sids)  # 视图/小拷贝
        V = self.v_pool.index_select(0, active_sids)
        L = self.lengths.index_select(0, active_sids)
        # 某些 kernel 需要 contiguous
        return K.contiguous(), V.contiguous(), L

    # （可选）若你的内核是 varlen + cu_seqlens，可以提供 flatten + cu_seqlens 的打包接口
    @torch.no_grad()
    def get_varlen_flatten(self, active_sids: torch.Tensor):
        """
        把每行的前 L_i 段拼成一条扁平序列（仅示例，易读性优先）
        返回：
          K_flat: [sum(L_i), H, D]
          V_flat: [sum(L_i), H, D]
          cu_seqlens: [B+1]  int32, 以 token 计数的前缀和
        """
        L = self.lengths.index_select(0, active_sids)  # [B]
        B = active_sids.numel()
        parts_K, parts_V = [], []
        for i in range(B):
            sid = int(active_sids[i].item())
            Li = int(L[i].item())
            if Li > 0:
                parts_K.append(self.k_pool[sid, :, :Li, :])  # [H, Li, D]
                parts_V.append(self.v_pool[sid, :, :Li, :])
        if parts_K:
            K_cat = torch.cat([p.permute(1,0,2).contiguous() for p in parts_K], dim=0)  # [sum(L), H, D]
            V_cat = torch.cat([p.permute(1,0,2).contiguous() for p in parts_V], dim=0)
        else:
            K_cat = torch.empty(0, self.spec.n_heads, self.spec.head_dim, dtype=self.spec.dtype, device=self.spec.device)
            V_cat = torch.empty_like(K_cat)
        # cu_seqlens 以 token 为单位累加（每序列 L_i）
        cu = torch.zeros(B+1, dtype=torch.int32, device=self.spec.device)
        if B > 0:
            cu[1:] = torch.cumsum(L, dim=0)
        return K_cat, V_cat, cu, L

    # -------- 写：把新 token 的 k_t/v_t 追加到各自序列末尾 --------
    @torch.no_grad()
    def append_one(self, sid: int, k_t: torch.Tensor, v_t: torch.Tensor):
        """
        k_t/v_t: [H, 1, D] 或 [H, D]（会自动 unsqueeze）
        """
        H, C, D = self.spec.n_heads, self.spec.cap, self.spec.head_dim
        L = int(self.lengths[sid].item())
        if L >= C:
            raise RuntimeError(f"KV cap overflow: sid={sid}, L={L}, CAP={C}")
        if k_t.dim() == 2: k_t = k_t.unsqueeze(1)
        if v_t.dim() == 2: v_t = v_t.unsqueeze(1)
        # 写入第 L 个位置
        self.k_pool[sid, :, L:L+1, :].copy_(k_t)
        self.v_pool[sid, :, L:L+1, :].copy_(v_t)
        self.lengths[sid] = L + 1

    @torch.no_grad()
    def append_batch(self, active_sids: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor):
        """
        批量追加：
          active_sids: [B]
          k_new/v_new: [B, H, 1, D] （或 [B, H, D]）
        为了简洁，使用小循环写入；写入是显存带宽瓶颈，循环开销很小
        """
        if k_new.dim() == 3: k_new = k_new.unsqueeze(2)
        if v_new.dim() == 3: v_new = v_new.unsqueeze(2)
        B = active_sids.numel()
        for i in range(B):
            self.append_one(int(active_sids[i].item()),
                            k_new[i], v_new[i])
