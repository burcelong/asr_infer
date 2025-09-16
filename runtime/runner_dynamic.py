import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import time

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
    """
    令牌级动态批处理运行器
    修复点：
      - 释放顺序：先 pop 再 free，再修补移动行
      - 规范化 KV.free 返回
      - 多完成降序释放
      - 容量计算排除 finished
      - 一致性自检 + 幽灵清理（开发期可关）
    """
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
        self._active: Dict[int, DecodeRequest] = {}
        self._sid2req: Dict[int, DecodeRequest] = {}
        self._staged: List[DecodeRequest] = []
        self._work_event = asyncio.Event()

        self._fn_alloc = self._pick(["alloc", "allocate", "acquire_slot"])
        self._fn_free = self._pick(["free", "release", "free_slot"])
        self._fn_get_batch = self._pick(["get_batch_fixedcap", "get_batch", "gather_fixedcap", "gather_batch"])
        self._fn_append_batch = lambda *a, **k: None

        self._invariant_check = True  # 开发期可设为 False 关闭一致性检查

    def _pick(self, names: List[str]) -> Callable:
        for name in names:
            method = getattr(self.kv, name, None)
            if callable(method):
                return method
        raise AttributeError(f"KV管理器缺少以下任一方法: {names}")

    async def submit(
        self,
        *,
        enc: EncState,
        max_len: int,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        sid: Optional[int] = None,
        **meta
    ) -> asyncio.Future:
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

    # --------- 规范化 free() 返回 ---------
    @staticmethod
    def _normalize_moved(freed_sid: int, moved) -> Optional[Tuple[int, int]]:
        """
        归一化为 (moved_sid, dst_sid=freed_sid)。
        支持 (moved, dst) 或 (dst, moved)；None/False 视为无移动。
        """
        if not moved:
            return None
        try:
            a, b = moved
        except Exception:
            return None
        if a == freed_sid and b != freed_sid:
            return (b, freed_sid)   # (dst, moved) -> (moved, dst)
        if b == freed_sid and a != freed_sid:
            return (a, freed_sid)   # (moved, dst)
        # 其它返回形态不可信，忽略修补
        return None

    def _free_slot_and_patch_sid(self, freed_sid: int):
        """
        原子释放 + 修补：
          1) 先把 freed_sid 从 _active/_sid2req 移除
          2) 调 KV.free(freed_sid)
          3) 若有移动，把 moved_sid 的请求修正到 dst_sid=freed_sid
        """
        # 1) 先移除“完成的旧行”，避免误删搬来的
        self._active.pop(freed_sid, None)
        self._sid2req.pop(freed_sid, None)

        # 2) 调 free，可能触发行交换（尾部补洞）
        moved_raw = self._fn_free(freed_sid)
        norm = self._normalize_moved(freed_sid, moved_raw)
        if not norm:
            return
        moved_sid, dst_sid = norm  # dst_sid 必为 freed_sid

        # 3) 修补 moved -> dst
        moved_req = self._sid2req.pop(moved_sid, None)
        if moved_req is not None:
            if moved_sid in self._active:
                self._active.pop(moved_sid, None)
            moved_req.sid = dst_sid
            self._sid2req[dst_sid] = moved_req
            self._active[dst_sid] = moved_req

    def _check_invariants(self):
        """开发期：轻量一致性检查 + finished 幽灵清理"""
        if not self._invariant_check:
            return
        for sid, r in list(self._active.items()):
            if r.sid != sid:
                print(f"[runner][WARN] active-key/sid mismatch: key={sid}, req.sid={r.sid}, req_id={r.req_id}")
        dangling = set(self._active.keys()) - set(self._sid2req.keys())
        if dangling:
            print(f"[runner][WARN] active keys not all in sid2req: {sorted(dangling)}")
        # 清理 finished 幽灵占位
        if any(r.finished for r in self._active.values()):
            self._active = {sid: r for sid, r in self._active.items() if not r.finished}

    async def _main_loop(self):
        tick_sleep = (1.0 / self.loop_hz) if self.loop_hz > 0 else 0.0

        while self._running:
            # 1) 吸收新请求
            await self._drain_new_reqs()

            # 2) 分配槽位（把 staged 推入 active）
            if self._staged:
                try:
                    qsz = self._queue.qsize()
                except Exception:
                    qsz = -1
                print(f"[runner] staged_before={len(self._staged)} active={len(self._active)} "
                      f"queue={qsz} slots_max={self.max_active}")

                new_list: List[DecodeRequest] = []
                for req in list(self._staged):
                    if req.sid is None:
                        try:
                            sid = self._fn_alloc()
                            req.sid = sid
                            print(f"[runner] alloc sid={sid} for req={req.req_id} "
                                  f"(file={req.meta.get('filename','?')})")
                        except Exception as e:
                            print(f"[runner] NO SLOT for req={req.req_id} "
                                  f"staged={len(self._staged)} active={len(self._active)} "
                                  f"err={type(e).__name__}:{e}")
                            continue
                    self._active[req.sid] = req
                    self._sid2req[req.sid] = req
                    new_list.append(req)

                self._staged = [req for req in self._staged if req not in new_list]
                print(f"[runner] staged_after={len(self._staged)} active={len(self._active)} "
                      f"newly_activated={len(new_list)}")

            # 3) 组 batch（只挑未完成）
            active_items = [
                (sid, req)
                for sid, req in self._active.items()
                if (not req.finished) and req.step < req.max_len
            ]

            if not active_items:
                backlog = self._queue.qsize() + len(self._staged)
                self._work_event.clear()
                if backlog == 0:
                    await self._work_event.wait()
                else:
                    win = min(0.005, 0.002 * max(1, backlog))
                    try:
                        await asyncio.wait_for(self._work_event.wait(), timeout=win)
                    except asyncio.TimeoutError:
                        pass
                await asyncio.sleep(0)
                continue

            # 4) 排序执行一步
            active_items.sort(key=lambda x: x[0])
            sids_list = [sid for sid, _ in active_items]
            batch_reqs = [req for _, req in active_items]
            print(f"[runner] STEP start sids={sids_list} reqs={[r.req_id for r in batch_reqs]} "
                  f"steps={[r.step for r in batch_reqs]}")
            t0 = time.perf_counter()

            sids = torch.tensor(sids_list, device=self.device, dtype=torch.long)
            try:
                K, V, L = self._fn_get_batch(sids)
            except TypeError:
                K = torch.empty(0, device=self.device)
                V = torch.empty(0, device=self.device)
                L = torch.empty(0, device=self.device, dtype=torch.int32)

            y_prev = torch.tensor(
                [(req.tokens[-1] if req.tokens else req.bos_id) for req in batch_reqs],
                device=self.device, dtype=torch.long
            )

            try:
                logits, k_t, v_t = self.step_fn(y_prev, batch_reqs, K, V, L)
                t1 = time.perf_counter()
                print(f"[runner] STEP done dt={(t1 - t0)*1000:.1f}ms")
            except Exception as e:
                # 本步失败：给本轮 active 设置异常并释放
                for sid, req in active_items:
                    if not req.future.done():
                        req.future.set_exception(e)
                    req.finished = True
                    try:
                        self._free_slot_and_patch_sid(sid)
                    except Exception:
                        pass
                # 下一轮继续
                self._work_event.set()
                continue

            # 5) 采样 + 终止判定
            next_tokens = self.sample_fn(logits) if self.sample_fn else torch.argmax(logits, dim=-1)
            next_list = next_tokens.tolist()

            finished_sids: List[int] = []
            for i, req in enumerate(batch_reqs):
                tok = int(next_list[i])
                print(f"[runner] tok req={req.req_id} step={req.step+1} tok={tok} eos={tok==req.eos_id}")
                req.step += 1
                if tok == req.eos_id:
                    req.finished = True
                    finished_sids.append(int(sids_list[i]))
                else:
                    req.tokens.append(tok)
                    if req.step >= req.max_len:
                        req.finished = True
                        finished_sids.append(int(sids_list[i]))

            # 6) 按 sid 降序释放，减少交换互扰
            for sid in sorted(finished_sids, reverse=True):
                req = self._active.get(sid)
                if req is None:
                    continue
                if not req.future.done():
                    req.future.set_result({"req_id": req.req_id, "tokens": req.tokens, "meta": req.meta})
                if self.on_finish is not None:
                    try:
                        self.on_finish(req)
                    except Exception:
                        pass
                try:
                    print(f"[runner] FIN req={req.req_id} sid={sid} len={len(req.tokens)}")
                    self._free_slot_and_patch_sid(sid)
                    print(f"[runner] ACTIVE after free: {list(self._active.keys())}")
                except Exception:
                    pass
                # 不要再额外 pop(sid)，上面已处理
                print(f"[runner] ACTIVE after pop({sid}): {list(self._active.keys())}")
                req.sid = None

            # 7) 开发期一致性检查 + 小让步
            self._check_invariants()
            await asyncio.sleep(0)

    async def _drain_new_reqs(self):
        """从请求队列中获取新请求并进行初步处理"""
        # 只统计未完成活跃项，防止幽灵占位导致容量为负
        live_active = sum(1 for r in self._active.values() if not r.finished)
        n_can_take = self.max_active - (live_active + len(self._staged))
        taken = 0

        while taken < n_can_take and not self._queue.empty():
            req: DecodeRequest = await self._queue.get()
            if not self.join_new_on_next_step:
                if req.sid is None:
                    try:
                        req.sid = self._fn_alloc()
                    except Exception:
                        await self._queue.put(req)
                        break
                self._active[req.sid] = req
                self._sid2req[req.sid] = req
            else:
                self._staged.append(req)
            taken += 1
