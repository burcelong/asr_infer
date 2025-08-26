import torch
import argparse
import os
import traceback
import threading
import time
import asyncio
import gc
import wave  # 用于计算语音时长
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from fireredasr.models.fireredasr import FireRedAsr
import tempfile
import uuid
from dataclasses import dataclass, field

# ========================= 配置 =========================
MODEL_PATH = os.getenv("FIRERED_MODEL_PATH", "/root/autodl-tmp/tenxun/FireRedASR-AED-L")

if not os.path.exists(MODEL_PATH):
    print(f"错误: 模型路径不存在 - {MODEL_PATH}")
    exit(1)

# 添加安全全局对象
torch.serialization.add_safe_globals([argparse.Namespace])

# 每多少个 batch 做一次周期性 empty_cache
EMPTY_CACHE_PERIOD = 50


@dataclass
class BatchRequest:
    """单个请求的数据结构（保留语音时长用于结果返回）"""
    request_id: str
    filename: str
    temp_path: str
    config: Dict[str, Any]
    result: Dict[str, Any]  # 存储处理结果
    event: asyncio.Event  # 异步事件
    start_time: float  # 计时用
    enqueue_time: float  # 入队时间
    audio_duration: float  # 语音时长（秒，仅用于结果展示）


@dataclass
class ModelInstance:
    """
    单模型实例类：
    所有语音请求统一进入该实例的队列，条件变量驱动批处理
    """
    instance_id: int
    name: str  # 队列名称（用于日志区分）
    model: Any = None
    queue: List[BatchRequest] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    cond: threading.Condition = field(default_factory=lambda: threading.Condition(threading.Lock()))
    is_processing: bool = False
    process_lock: threading.Lock = field(default_factory=threading.Lock)
    batch_thread: threading.Thread = None
    batch_size: int = 32
    batch_timeout: float = 0.01 
    max_queue_size: int = 400 
    batch_counter: int = 0  

    def start(self):
        """启动批处理线程"""
        self.batch_thread = threading.Thread(
            target=batch_processor, args=(self,), daemon=True
        )
        self.batch_thread.start()


class ASRRequest(BaseModel):
    """ASR请求参数模型（保持原配置项不变）"""
    beam_size: int = 2
    nbest: int = 1
    decode_max_len: int = 0
    softmax_smoothing: float = 1
    aed_length_penalty: float = 0.6
    eos_penalty: float = 1.0
    use_gpu: int = 1  # 0=CPU，1=GPU


# 全局唯一模型实例（替代原长短两个实例）
asr_instance: Optional[ModelInstance] = None

# 创建FastAPI应用（更新描述为单队列服务）
app = FastAPI(
    title="单队列ASR服务",
    description="所有语音请求统一进入单个队列，由单个模型实例批处理"
)


def get_audio_duration(file_path: str) -> float:
    """计算语音文件时长（秒，保留用于结果返回）"""
    try:
        with wave.open(file_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate == 0:
                return 0.0
            return frames / float(rate)
    except Exception as e:
        print(f"计算语音时长失败: {e}")
        return 0.0  # 异常时返回0.0，不影响队列处理


def safe_results_index(results, i):
    """安全获取模型返回的第 i 条结果（防止索引越界）"""
    if results is None or i >= len(results):
        return {"uttid": "", "text": "", "nbest": []}
    item = results[i]
    return item[0] if isinstance(item, list) and len(item) > 0 else item


def run_inference_with_oom_recovery(instance: ModelInstance, batch_uttid, batch_wav_path, config, take_count):
    """
    推理执行（带OOM恢复逻辑）：
    - 首次按当前批次量推
    - 若OOM则清空缓存并逐步降批重试（最多2次降批）
    """
    try_sizes = [take_count, max(1, take_count // 2), max(1, take_count // 4)]
    last_err = None
    for sz in try_sizes:
        try:
            # 裁剪批次量（按当前重试尺寸）
            uttid = batch_uttid[:sz]
            paths = batch_wav_path[:sz]
            with torch.no_grad():  # 禁用梯度计算，减少显存占用
                return instance.model.transcribe(uttid, paths, config)
        except RuntimeError as e:
            msg = str(e)
            last_err = e
            if "out of memory" in msg.lower():
                print(f"[WARN][{instance.name}] CUDA OOM，清空缓存并降批重试（当前batch={sz}）")
                torch.cuda.empty_cache()
                time.sleep(0.01)  # 短暂等待缓存清理完成
                continue
            else:
                raise  # 非OOM错误直接抛出
    # 所有重试失败时抛出最后一次错误
    raise last_err if last_err else RuntimeError("未知推理错误")


def batch_processor(instance: ModelInstance):
    """批处理线程（处理实例唯一队列）"""
    while True:
        # 等待批处理触发条件（满批/超时/队列非空）
        with instance.cond:
            start_wait = time.time()
            while True:
                qlen = len(instance.queue)
                if qlen >= instance.batch_size:
                    break  # 满批触发
                # 计算剩余等待时间（未超时且队列非空则继续等）
                remaining = instance.batch_timeout - (time.time() - start_wait)
                if qlen > 0 and remaining <= 0:
                    break  # 超时触发（有任务时）
                # 队列空则等待通知，否则等待剩余时间
                timeout = remaining if qlen > 0 else instance.batch_timeout
                instance.cond.wait(timeout=timeout)

        # 占用处理锁（防止同一实例并行推理）
        with instance.process_lock:
            if instance.is_processing:
                print(f"[DEBUG][{instance.name}] 实例忙碌，跳过本次调度")
                continue
            instance.is_processing = True

        # 从队列取出当前批次任务
        current_batch: List[BatchRequest] = []
        try:
            with instance.lock:
                if instance.queue:
                    take_count = min(instance.batch_size, len(instance.queue))
                    current_batch = instance.queue[:take_count]
                    instance.queue = instance.queue[take_count:]
                    print(f"[DEBUG][{instance.name}] 取出{len(current_batch)}个任务（剩余：{len(instance.queue)}）")
        except Exception as e:
            print(f"[ERROR][{instance.name}] 取出任务失败：{e}")
            with instance.process_lock:
                instance.is_processing = False
            continue

        if not current_batch:
            with instance.process_lock:
                instance.is_processing = False
            continue

        # 执行批量推理
        results = None
        try:
            infer_start = time.time()
            # 提取批次内的关键信息
            batch_uttid = [req.request_id for req in current_batch]
            batch_wav_path = [req.temp_path for req in current_batch]
            config = current_batch[0].config  # 同批次使用相同配置

            # 带OOM恢复的推理
            results = run_inference_with_oom_recovery(
                instance=instance,
                batch_uttid=batch_uttid,
                batch_wav_path=batch_wav_path,
                config=config,
                take_count=len(current_batch),
            )

            # 计算推理耗时并映射结果到每个请求
            infer_duration = time.time() - infer_start
            print(f"[INFO][{instance.name}] 批处理完成，耗时{infer_duration:.3f}秒，batch={len(current_batch)}")
            for i, req in enumerate(current_batch):
                res = safe_results_index(results, i)
                total_process_time = time.time() - req.start_time
                # 填充结果（保留语音时长用于展示）
                req.result = {
                    "filename": req.filename,
                    "results": res,
                    "status": "success",
                    "process_time": f"{total_process_time:.4f}秒",
                    "instance_id": instance.instance_id,
                    "batch_size": len(current_batch),
                    "queue_type": instance.name,
                    "audio_duration": f"{req.audio_duration:.2f}秒",
                }
                req.event.set()  # 唤醒等待的请求
                print(f"[DEBUG][{instance.name}] 任务{req.request_id}完成，总耗时{total_process_time:.3f}秒")

        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR][{instance.name}] 批处理出错：{error_msg}")
            # 错误时给每个请求填充错误信息
            for req in current_batch:
                req.result = {
                    "error": error_msg,
                    "status": "error",
                    "batch_size": len(current_batch),
                    "queue_type": instance.name,
                }
                req.event.set()
        finally:
            # 清理临时文件（防止磁盘占用）
            for req in current_batch:
                try:
                    if os.path.exists(req.temp_path):
                        os.unlink(req.temp_path)
                except Exception as e:
                    print(f"[WARN][{instance.name}] 清理临时文件失败: {e}")

            # 内存回收（减少内存泄漏风险）
            del current_batch, results
            gc.collect()

            # 周期性显存清理（按配置周期执行）
            instance.batch_counter += 1
            if instance.batch_counter % EMPTY_CACHE_PERIOD == 0:
                print(f"[DEBUG][{instance.name}] 周期性显存清理（每{EMPTY_CACHE_PERIOD}批）")
                torch.cuda.empty_cache()

            # 释放实例忙碌标记
            with instance.process_lock:
                instance.is_processing = False


def load_model():
    """加载单个模型实例（替代原双实例加载）"""
    global asr_instance
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"正在加载模型实例（设备：{device}）...")

    try:
        # 初始化单模型实例（沿用原短语音实例配置，可根据需求调整）
        asr_instance = ModelInstance(
            instance_id=0,
            name="single_queue"
        )
        # 加载模型权重
        print(f"加载模型到实例 {asr_instance.instance_id} ({asr_instance.name}) ...")
        asr_instance.model = FireRedAsr.from_pretrained("aed", MODEL_PATH, device=device)

        # 初始显存清理
        torch.cuda.empty_cache()
        print(f"[DEBUG] 模型加载完成，初始GPU显存清理完成")

        # 启动批处理线程
        asr_instance.start()
        print("单模型实例加载完成！")

    except Exception as e:
        print(f"模型加载失败: {e}")
        traceback.print_exc()
        exit(1)


@app.post("/asr/single", response_model=Dict[str, Any])
async def single_file_asr(
    audio_file: UploadFile = File(...),
    config: Optional[ASRRequest] = None
):
    """异步接口：所有语音请求统一进入单队列处理"""
    # 检查模型是否加载完成
    if not asr_instance:
        raise HTTPException(status_code=503, detail="模型未加载完成，服务暂不可用")

    request_id = str(uuid.uuid4())  # 生成唯一请求ID
    config = config or ASRRequest()  # 默认配置
    start_time = time.time()
    temp_path = None  # 临时文件路径

    try:
        # 1. 保存上传文件到临时目录（.wav格式）
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            temp_path = f.name
            f.write(await audio_file.read())  # 异步读取上传文件

        # 2. 计算语音时长（仅用于结果返回，不影响队列）
        audio_duration = get_audio_duration(temp_path)

        # 3. 队列容量检查（防止队列溢出）
        with asr_instance.lock:
            current_queue_len = len(asr_instance.queue)
        if current_queue_len >= asr_instance.max_queue_size:
            # 队列满时清理临时文件并返回错误
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return JSONResponse(
                content={
                    "status": "error",
                    "error": f"服务繁忙（队列已达上限：{current_queue_len}/{asr_instance.max_queue_size}），请稍后重试",
                    "queue_type": asr_instance.name,
                },
                status_code=503,
            )

        # 4. 构造请求对象并加入队列
        event = asyncio.Event()  # 用于异步等待结果
        enqueue_time = time.time()
        batch_req = BatchRequest(
            request_id=request_id,
            filename=audio_file.filename,
            temp_path=temp_path,
            config=config.model_dump(),  # 转换为字典格式
            result={},
            event=event,
            start_time=start_time,
            enqueue_time=enqueue_time,
            audio_duration=audio_duration,
        )

        # 加入队列并唤醒批处理线程
        with asr_instance.cond:
            asr_instance.queue.append(batch_req)
            new_queue_len = len(asr_instance.queue)
            print(f"[DEBUG] 请求{request_id}入队（语音时长：{audio_duration:.2f}秒，队列长度：{new_queue_len}）")
            asr_instance.cond.notify()  # 唤醒等待的批处理线程

        # 5. 等待结果（超时时间：60秒，可根据需求调整）
        timeout = 60  # 单个请求超时时间
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            # 超时后从队列中移除该请求（防止僵尸任务）
            with asr_instance.cond:
                asr_instance.queue = [req for req in asr_instance.queue if req.request_id != request_id]
                asr_instance.cond.notify_all()  # 重新唤醒线程处理剩余任务
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return JSONResponse(
                content={
                    "status": "error",
                    "error": f"请求超时（已超过{timeout}秒）",
                    "queue_type": asr_instance.name,
                },
                status_code=504,
            )

        # 6. 返回处理结果（成功/失败）
        if batch_req.result.get("status") == "success":
            return batch_req.result
        else:
            return JSONResponse(content=batch_req.result, status_code=500)

    except Exception as e:
        # 异常兜底：清理临时文件并返回错误
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        error_msg = str(e)
        print(f"[ERROR] 请求{request_id}处理出错：{error_msg}")
        return JSONResponse(
            content={"error": error_msg, "status": "error", "request_id": request_id},
            status_code=500
        )


@app.get("/healthz")
def health_check():
    """服务健康检查接口（返回队列状态）"""
    if not asr_instance:
        return JSONResponse(content={"status": "not_ready", "reason": "模型未加载"}, status_code=503)
    return {
        "status": "ok",
        "queue_length": len(asr_instance.queue),  # 当前队列长度
        "max_queue_size": asr_instance.max_queue_size,  # 队列最大容量
        "instance_name": asr_instance.name,  # 实例名称
        "batch_size": asr_instance.batch_size  # 批处理尺寸
    }


if __name__ == '__main__':
    # 1. 加载模型（启动服务前必须完成）
    load_model()

    # 2. 启动FastAPI服务（单进程，避免多进程模型重复加载）
    import uvicorn
    print("启动单队列ASR服务...")
    uvicorn.run(
        app,
        host='0.0.0.0',  # 允许外部访问
        port=8023,        # 服务端口（可根据需求调整）
        workers=1,        # 单进程：避免多进程导致模型重复加载和显存溢出
        reload=False,     # 生产环境禁用自动重载
        timeout_keep_alive=120  # 连接保持超时
    )