import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


@dataclass
class EncState:
    """编码器状态数据类，存储编码过程中的关键信息"""
    enc_k: Optional[torch.Tensor] = None
    enc_v: Optional[torch.Tensor] = None
    cu_seqlens: Optional[torch.Tensor] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecodeRequest:
    """解码请求数据类，包含单个解码任务的所有信息"""
    req_id: int
    bos_id: int
    eos_id: int
    max_len: int
    enc: EncState
    meta: Dict[str, Any] = field(default_factory=dict)
    sid: Optional[int] = None  # 槽位ID
    tokens: List[int] = field(default_factory=list)  # 已生成的token列表
    finished: bool = False  # 是否完成解码
    step: int = 0  # 当前解码步数
    future: asyncio.Future = field(default_factory=asyncio.Future)  # 用于异步结果返回


class TokenRunner:
    """
    令牌级动态批处理运行器，负责管理解码请求队列、分配资源并协调解码过程
    支持连续批处理和KV缓存管理
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
        
        # 设备配置
        self.device = (
            device 
            or getattr(getattr(kv, "spec", None), "device", None) 
            or getattr(kv, "device", None) 
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # 解码参数
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_active = max_active or getattr(getattr(kv, "spec", None), "slots", 1024)
        self.loop_hz = loop_hz
        self.join_new_on_next_step = join_new_on_next_step
        self.sample_fn = sample_fn
        self.on_finish = on_finish
        
        # 运行状态管理
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._req_id_gen = 0  # 请求ID生成器
        self._queue: "asyncio.Queue[DecodeRequest]" = asyncio.Queue()  # 请求队列
        self._active: Dict[int, DecodeRequest] = {}  # 活跃请求 {sid: 请求}
        self._sid2req: Dict[int, DecodeRequest] = {}  # 槽位ID到请求的映射
        self._staged: List[DecodeRequest] = []  # 待处理的请求
        self._work_event = asyncio.Event()  # 工作事件触发器
        
        # KV缓存管理器接口适配
        self._fn_alloc = self._pick(["alloc", "allocate", "acquire_slot"])
        self._fn_free = self._pick(["free", "release", "free_slot"])
        self._fn_get_batch = self._pick(["get_batch_fixedcap", "get_batch", "gather_fixedcap", "gather_batch"])
        self._fn_append_batch = lambda *a, **k: None  # Stepper已处理KV写入

    def _pick(self, names: List[str]) -> Callable:
        """从KV管理器中选择可用的接口方法"""
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
        sid: Optional[int] = None,** meta
    ) -> asyncio.Future:
        """提交新的解码请求并返回Future对象用于获取结果"""
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
        """启动运行器的主循环"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._main_loop())

    async def stop(self):
        """停止运行器并清理资源"""
        self._running = False
        if self._task:
            await self._task
            self._task = None

    def _free_slot_and_patch_sid(self, freed_sid: int):
        """
        释放指定槽位并处理KV缓存中的交换修复
        如果最后一行被交换到释放的槽位，更新相关请求的映射关系
        """
        # 移除完成请求的映射
        self._sid2req.pop(freed_sid, None)
        
        # 释放槽位并获取可能的交换信息
        moved = self._fn_free(freed_sid)
        
        # 处理交换情况：原最后一行被移动到释放的槽位
        if moved is not None:
            try:
                moved_sid, dst_sid = moved
            except Exception:
                # 兼容直接返回元组的情况
                moved_sid, dst_sid = moved[0], moved[1]
                
            # 更新被移动请求的槽位ID和映射关系
            moved_req = self._sid2req.pop(moved_sid, None)
            if moved_req is not None:
                moved_req.sid = dst_sid
                self._sid2req[dst_sid] = moved_req
                
                # 同步活跃请求字典
                if moved_sid in self._active:
                    self._active.pop(moved_sid, None)
                    self._active[dst_sid] = moved_req

    async def _main_loop(self):
        """运行器主循环，处理请求队列和批处理解码"""
        tick_sleep = (1.0 / self.loop_hz) if self.loop_hz > 0 else 0.0
        
        while self._running:
            # 处理新请求
            await self._drain_new_reqs()
            
            # 为待处理请求分配槽位
            if self._staged:
                new_list: List[DecodeRequest] = []
                for req in self._staged:
                    if req.sid is None:
                        try:
                            req.sid = self._fn_alloc()
                        except Exception:
                            # 没有可用槽位，留到下一轮处理
                            continue
                    
                    self._active[req.sid] = req
                    self._sid2req[req.sid] = req
                    new_list.append(req)
                
                # 保留未成功分配的请求
                self._staged = [req for req in self._staged if req not in new_list]
            
            # 获取仍活跃的请求
            active_items = [
                (sid, req) 
                for sid, req in self._active.items() 
                if not req.finished and req.step < req.max_len
            ]
            
            # 没有活跃请求时的处理逻辑
            if not active_items:
                backlog = self._queue.qsize() + len(self._staged)
                self._work_event.clear()
                
                if backlog == 0:
                    # 无积压请求，等待新请求
                    await self._work_event.wait()
                else:
                    # 有积压请求，使用自适应等待窗口
                    win = min(0.005, 0.002 * max(1, backlog))
                    try:
                        await asyncio.wait_for(self._work_event.wait(), timeout=win)
                    except asyncio.TimeoutError:
                        pass
                
                await asyncio.sleep(0)
                continue
            
            # 按槽位ID排序，确保连续前缀以优化性能
            active_items.sort(key=lambda x: x[0])
            sids_list = [sid for sid, _ in active_items]
            batch_reqs = [req for _, req in active_items]
            batch_size = len(batch_reqs)
            
            # 准备槽位ID张量
            sids = torch.tensor(sids_list, device=self.device, dtype=torch.long)
            
            # 获取KV缓存批次数据
            try:
                K, V, L = self._fn_get_batch(sids)
            except TypeError:
                # 兼容需要layer_idx参数的实现
                K = torch.empty(0, device=self.device)
                V = torch.empty(0, device=self.device)
                L = torch.empty(0, device=self.device, dtype=torch.int32)
            
            # 准备上一步的token（首次使用BOS）
            y_prev = torch.tensor(
                [(req.tokens[-1] if req.tokens else req.bos_id) for req in batch_reqs],
                device=self.device,
                dtype=torch.long
            )
            
            # 执行解码步骤
            try:
                logits, k_t, v_t = self.step_fn(y_prev, batch_reqs, K, V, L)
            except Exception as e:
                # 处理解码错误：标记所有请求为完成并返回异常
                for sid, req in active_items:
                    if not req.future.done():
                        req.future.set_exception(e)
                    req.finished = True
                    try:
                        self._free_slot_and_patch_sid(sid)
                    except Exception:
                        pass
                
                # 清理活跃请求
                for sid, _ in active_items:
                    self._active.pop(sid, None)
                
                continue
            
            # 采样生成下一个token
            next_tokens = self.sample_fn(logits) if self.sample_fn else torch.argmax(logits, dim=-1)
            
            # 处理生成的token并检查终止条件
            next_list = next_tokens.tolist()
            finished_sids: List[int] = []
            
            for i, req in enumerate(batch_reqs):
                tok = int(next_list[i])
                req.step += 1
                
                # 检查是否到达终止条件
                if tok == req.eos_id:
                    req.finished = True
                    finished_sids.append(int(sids_list[i]))
                else:
                    req.tokens.append(tok)
                    # 检查是否达到最大长度
                    if req.step >= req.max_len:
                        req.finished = True
                        finished_sids.append(int(sids_list[i]))
            
            # 处理完成的请求
            for sid in finished_sids:
                req = self._active.get(sid)
                if req is None:
                    continue
                
                # 返回结果
                if not req.future.done():
                    req.future.set_result({
                        "req_id": req.req_id,
                        "tokens": req.tokens,
                        "meta": req.meta
                    })
                
                # 调用完成回调
                if self.on_finish is not None:
                    try:
                        self.on_finish(req)
                    except Exception:
                        pass
                
                # 释放槽位并处理交换
                try:
                    self._free_slot_and_patch_sid(sid)
                except Exception:
                    pass
                
                # 从活跃请求中移除
                self._active.pop(sid, None)
                req.sid = None  # 标记不再占用槽位
            
            await asyncio.sleep(0)

    async def _drain_new_reqs(self):
        """从请求队列中获取新请求并进行初步处理"""
        # 计算可接收的新请求数量
        n_can_take = self.max_active - (len(self._active) + len(self._staged))
        taken = 0
        
        while taken < n_can_take and not self._queue.empty():
            req: DecodeRequest = await self._queue.get()
            
            if not self.join_new_on_next_step:
                # 立即分配槽位
                if req.sid is None:
                    try:
                        req.sid = self._fn_alloc()
                    except Exception:
                        # 分配失败，放回队列
                        await self._queue.put(req)
                        break
                
                self._active[req.sid] = req
                self._sid2req[req.sid] = req
            else:
                # 暂存到下一批处理
                self._staged.append(req)
            
            taken += 1