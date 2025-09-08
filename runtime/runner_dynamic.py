# fireredasr/runtime/runner_dynamic.py
import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch

@dataclass
class EncState:
    enc_k: Optional[torch.Tensor] = None
    enc_v: Optional[torch.Tensor] = None
    cu_seqlens: Optional[torch.Tensor] = None
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DecodeRequest:
    req_id: int
    bos_id: int
    eos_id: int
    max_len: int
    enc: EncState
    meta: Dict[str, Any] = field(default_factory=dict)
    sid: Optional[int] = None
    tokens: List[int] = field(default_factory=list)
    finished: bool = False
    step: int = 0
    future: asyncio.Future = field(default_factory=asyncio.Future)

class TokenRunner:
    def __init__(
        self,
        *,
        kv,
        step_fn: Callable[
            [torch.Tensor, List[DecodeRequest], torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        device: Optional[str] = None,
        bos_id: int = 1,
        eos_id: int = 2,
        max_active: Optional[int] = None,
        loop_hz: int = 0,
        join_new_on_next_step: bool = True,
        sample_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        on_finish: Optional[Callable[[DecodeRequest], None]] = None,
    ):
        self.kv = kv
        self.step_fn = step_fn
        self.device = (
            device
            or getattr(getattr(kv, "spec", None), "device", None)
            or getattr(kv, "device", None)
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_active = max_active or getattr(getattr(kv, "spec", None), "slots", 1024)
        self.loop_hz = loop_hz
        self.join_new_on_next_step = join_new_on_next_step
        self.sample_fn = sample_fn
        self.on_finish = on_finish

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._req_id_gen = 0
        self._queue: "asyncio.Queue[DecodeRequest]" = asyncio.Queue()
        self._active: Dict[int, DecodeRequest] = {}  # sid -> req
        self._sid2req: Dict[int, DecodeRequest] = {}  # <—— 新增：维护映射，用于 swap 修正
        self._staged: List[DecodeRequest] = []
        self._work_event = asyncio.Event()

        # 适配不同命名
        self._fn_alloc = self._pick(["alloc", "allocate", "acquire_slot"])
        self._fn_free = self._pick(["free", "release", "free_slot"])
        self._fn_get_batch = self._pick(["get_batch_fixedcap", "get_batch", "gather_fixedcap", "gather_batch"])

        # Stepper 已在内核中原地写 KV；Runner 不需要 append。
        self._fn_append_batch = lambda *a, **k: None

    def _pick(self, names: List[str]):
        for n in names:
            fn = getattr(self.kv, n, None)
            if callable(fn):
                return fn
        raise AttributeError(f"KV manager missing any of: {names}")

    async def submit(self, *, enc: EncState, max_len: int, bos_id: Optional[int] = None,
                     eos_id: Optional[int] = None, sid: Optional[int] = None, **meta) -> asyncio.Future:
        self._req_id_gen += 1
        req = DecodeRequest(
            req_id=self._req_id_gen,
            bos_id=bos_id if bos_id is not None else self.bos_id,
            eos_id=eos_id if eos_id is not None else self.eos_id,
            max_len=max_len,
            enc=enc,
            meta=meta,
        )
        if sid is not None:
            req.sid = sid
        await self._queue.put(req)
        self._work_event.set()
        return req.future

    def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._main_loop())

    async def stop(self):
        self._running = False
        if self._task:
            await self._task
            self._task = None

    # —— 关键：释放 + 处理 swap 并修复 sid —— #
    def _free_slot_and_patch_sid(self, freed_sid: int):
        """
        释放 freed_sid ；若 KV 把“最后一行” swap 到 freed_sid，
        把那条请求的 sid 改成 freed_sid，并修正映射。
        """
        # 移除完成请求的映射
        self._sid2req.pop(freed_sid, None)

        moved = self._fn_free(freed_sid)
        # free() 语义：
        #   - None：freed_sid 原本就是最后一行，无需修复
        #   - (moved_sid, dst_sid)：原最后一行 moved_sid 被换到 dst_sid(=freed_sid)
        if moved is not None:
            try:
                moved_sid, dst_sid = moved
            except Exception:
                # 如果你的 kv.free 直接返回 tuple，不带命名，仍然按 tuple 解
                moved_sid, dst_sid = moved[0], moved[1]

            moved_req = self._sid2req.pop(moved_sid, None)
            if moved_req is not None:
                # 更新 moved_req 的 sid 并重建映射/活动字典
                moved_req.sid = dst_sid
                self._sid2req[dst_sid] = moved_req
                # 同步 _active：把 key 从 moved_sid 改成 dst_sid
                if moved_sid in self._active:
                    self._active.pop(moved_sid, None)
                    self._active[dst_sid] = moved_req

    async def _main_loop(self):
        tick_sleep = (1.0 / self.loop_hz) if self.loop_hz > 0 else 0.0
        while self._running:
            await self._drain_new_reqs()

            # 把 staged 请求实际分配 slot（连续前缀策略：alloc 给 0..B-1）
            if self._staged:
                # 尽量一次性吃完 staged（受 slots 限制）
                new_list: List[DecodeRequest] = []
                for r in self._staged:
                    if r.sid is None:
                        try:
                            r.sid = self._fn_alloc()
                        except Exception:
                            # 没槽位，留到下回合
                            continue
                    self._active[r.sid] = r
                    self._sid2req[r.sid] = r  # 记录映射（用于 swap 修复）
                    new_list.append(r)
                # 清理已成功入场的
                self._staged = [r for r in self._staged if r not in new_list]

            # 取仍活跃的请求
            active_items = [(sid, r) for sid, r in self._active.items()
                            if (not r.finished) and (r.step < r.max_len)]
            if not active_items:
                # 纯事件驱动；无积压则无限等，有积压给极短窗口
                backlog = self._queue.qsize() + len(self._staged)
                self._work_event.clear()
                if backlog == 0:
                    await self._work_event.wait()
                else:
                    win = min(0.005, 0.002 * max(1, backlog))  # 2~8ms 自适应窗口
                    try:
                        await asyncio.wait_for(self._work_event.wait(), timeout=win)
                    except asyncio.TimeoutError:
                        pass
                # 只让出一次调度即可，别再固定 tick
                await asyncio.sleep(0)
                continue

            # —— 关键：按 sid 排序，保证 0..B-1 的连续前缀，Stepper 才能走 get_block_layer 快路径 —— #
            active_items.sort(key=lambda x: x[0])
            sids_list = [sid for sid, _ in active_items]
            batch_reqs = [r for _, r in active_items]
            B = len(batch_reqs)

            sids = torch.tensor(sids_list, device=self.device, dtype=torch.long)

            # Runner 侧的 K/V/L 只是为了兼容接口；Stepper 会自行从 kv 读取快路径视图
            try:
                K, V, L = self._fn_get_batch(sids)
            except TypeError:
                # 某些实现可能需要 layer_idx 参数；此处仅占位
                K = torch.empty(0, device=self.device)
                V = torch.empty(0, device=self.device)
                L = torch.empty(0, device=self.device, dtype=torch.int32)

            # 上一步 token（无则 BOS）
            y_prev = torch.tensor(
                [(r.tokens[-1] if r.tokens else r.bos_id) for r in batch_reqs],
                device=self.device, dtype=torch.long
            )

            # 推一步
            try:
                logits, k_t, v_t = self.step_fn(y_prev, batch_reqs, K, V, L)
            except Exception as e:
                # 出错：把本批全部异常返回，并释放它们占用的 slot（含 swap 修复）
                for sid, r in active_items:
                    if not r.future.done():
                        r.future.set_exception(e)
                    r.finished = True
                    try:
                        self._free_slot_and_patch_sid(sid)
                    except Exception:
                        pass
                # 清理 active 表（释放时已移动/移除）
                for sid, _ in active_items:
                    self._active.pop(sid, None)
                continue

            # 采样/贪心
            next_tokens = self.sample_fn(logits) if self.sample_fn else torch.argmax(logits, dim=-1)

            # Stepper 已原地写回 KV；Runner 不再 append
            # self._fn_append_batch(sids, k_t, v_t)

            # 写入 token、判终止
            next_list = next_tokens.tolist()
            finished_sids: List[int] = []
            for i, r in enumerate(batch_reqs):
                tok = int(next_list[i])
                r.tokens.append(tok)
                r.step += 1
                if tok == r.eos_id or r.step >= r.max_len:
                    r.finished = True
                    finished_sids.append(int(sids_list[i]))

            # 收尾：完成 → 释放（带 swap 修复）→ 返回结果
            for sid in finished_sids:
                req = self._active.get(sid)
                if req is None:
                    continue
                # 先把结果放回去（避免被 swap 时找不到）
                if not req.future.done():
                    req.future.set_result({"req_id": req.req_id, "tokens": req.tokens, "meta": req.meta})
                if self.on_finish is not None:
                    try:
                        self.on_finish(req)
                    except Exception:
                        pass
                # 释放并处理 swap
                try:
                    self._free_slot_and_patch_sid(sid)
                except Exception:
                    pass
                # 从 active 移除（注意：如果最后一行被换到这里，我们在 _free_slot_and_patch_sid 里已经把映射修复了）
                self._active.pop(sid, None)
                req.sid = None  # 标记该请求不再占槽

            await asyncio.sleep(0)

    async def _drain_new_reqs(self):
        n_can_take = self.max_active - (len(self._active) + len(self._staged))
        taken = 0
        while taken < n_can_take and not self._queue.empty():
            req: DecodeRequest = await self._queue.get()
            if not self.join_new_on_next_step:
                if req.sid is None:
                    try:
                        req.sid = self._fn_alloc()
                    except Exception:
                        # 分不到槽位，仍放回队列尾部
                        await self._queue.put(req)
                        break
                self._active[req.sid] = req
                self._sid2req[req.sid] = req   # <—— 也要登记映射
            else:
                self._staged.append(req)
            taken += 1
