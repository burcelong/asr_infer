import torch
import argparse  # 用于命令行参数解析
import os
import traceback
import threading
import time
import asyncio
import gc
import wave
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from fireredasr.models.fireredasr import FireRedAsr
import tempfile
import uuid
from dataclasses import dataclass, field

# ========================= 命令行参数解析（核心新增） =========================
def parse_args():
    """解析命令行参数：支持指定模型路径和服务端口"""
    parser = argparse.ArgumentParser(description="单队列ASR服务 - 支持命令行配置模型路径和端口")
    # 模型路径参数：--model-path 或 -m，优先级：命令行 > 环境变量 > 默认值
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        default=os.getenv("FIRERED_MODEL_PATH"),
    )
    # 服务端口参数：--port 或 -p，默认8023
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8023,
    )
    # 可选：添加主机参数（默认0.0.0.0，方便外部访问）
    parser.add_argument(
        "--host", "-H",
        type=str,
        default="0.0.0.0",
        help="服务监听主机（默认：0.0.0.0，允许外部网络访问）"
    )
    return parser.parse_args()

# 解析命令行参数（全局生效）
args = parse_args()

# ========================= 配置（基于命令行参数更新） =========================
# 模型路径：命令行参数 > 环境变量 > 默认值（已在argparse中处理）
MODEL_PATH = args.model_path

# 验证模型路径有效性
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
    """单模型实例类：所有语音请求统一进入该实例的队列"""
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


# 全局唯一模型实例
asr_instance: Optional[ModelInstance] = None

# 创建FastAPI应用
app = FastAPI(
    title="单队列ASR服务（命令行配置版）",
    description=f"支持命令行指定模型路径和端口，所有语音请求统一批处理\n当前模型路径：{MODEL_PATH}\n服务端口：{args.port}"
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
    """推理执行（带OOM恢复逻辑）"""
    try_sizes = [take_count, max(1, take_count // 2), max(1, take_count // 4)]
    last_err = None
    for sz in try_sizes:
        try:
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
                time.sleep(0.01)
                continue
            else:
                raise
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
                remaining = instance.batch_timeout - (time.time() - start_wait)
                if qlen > 0 and remaining <= 0:
                    break  # 超时触发（有任务时）
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

            # 映射结果并唤醒请求
            infer_duration = time.time() - infer_start
            print(f"[INFO][{instance.name}] 批处理完成，耗时{infer_duration:.3f}秒，batch={len(current_batch)}")
            for i, req in enumerate(current_batch):
                res = safe_results_index(results, i)
                total_process_time = time.time() - req.start_time
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
                req.event.set()
                print(f"[DEBUG][{instance.name}] 任务{req.request_id}完成，总耗时{total_process_time:.3f}秒")

        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR][{instance.name}] 批处理出错：{error_msg}")
            for req in current_batch:
                req.result = {
                    "error": error_msg,
                    "status": "error",
                    "batch_size": len(current_batch),
                    "queue_type": instance.name,
                }
                req.event.set()
        finally:
            # 清理临时文件
            for req in current_batch:
                try:
                    if os.path.exists(req.temp_path):
                        os.unlink(req.temp_path)
                except Exception as e:
                    print(f"[WARN][{instance.name}] 清理临时文件失败: {e}")

            # 内存回收
            del current_batch, results
            gc.collect()

            # 周期性显存清理
            instance.batch_counter += 1
            if instance.batch_counter % EMPTY_CACHE_PERIOD == 0:
                print(f"[DEBUG][{instance.name}] 周期性显存清理（每{EMPTY_CACHE_PERIOD}批）")
                torch.cuda.empty_cache()

            # 释放实例忙碌标记
            with instance.process_lock:
                instance.is_processing = False


def load_model():
    """加载单个模型实例（使用命令行指定的模型路径）"""
    global asr_instance
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"正在加载模型实例（设备：{device}，模型路径：{MODEL_PATH}）...")

    try:
        # 初始化单模型实例
        asr_instance = ModelInstance(
            instance_id=0,
            name="single_queue"
        )
        # 加载模型（使用命令行参数指定的路径）
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
    if not asr_instance:
        raise HTTPException(status_code=503, detail="模型未加载完成，服务暂不可用")

    request_id = str(uuid.uuid4())
    config = config or ASRRequest()
    start_time = time.time()
    temp_path = None

    try:
        # 1. 保存上传文件到临时目录
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            temp_path = f.name
            f.write(await audio_file.read())

        # 2. 计算语音时长
        audio_duration = get_audio_duration(temp_path)

        # 3. 队列容量检查
        with asr_instance.lock:
            current_queue_len = len(asr_instance.queue)
        if current_queue_len >= asr_instance.max_queue_size:
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

        # 4. 构造请求并加入队列
        event = asyncio.Event()
        enqueue_time = time.time()
        batch_req = BatchRequest(
            request_id=request_id,
            filename=audio_file.filename,
            temp_path=temp_path,
            config=config.model_dump(),
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
            asr_instance.cond.notify()

        # 5. 异步等待结果（超时60秒）
        timeout = 60
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            # 超时移除任务
            with asr_instance.cond:
                asr_instance.queue = [req for req in asr_instance.queue if req.request_id != request_id]
                asr_instance.cond.notify_all()
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

        # 6. 返回结果
        if batch_req.result.get("status") == "success":
            return batch_req.result
        else:
            return JSONResponse(content=batch_req.result, status_code=500)

    except Exception as e:
        # 异常兜底清理
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
    """服务健康检查接口（返回当前配置和队列状态）"""
    if not asr_instance:
        return JSONResponse(content={"status": "not_ready", "reason": "模型未加载"}, status_code=503)
    return {
        "status": "ok",
        "current_config": {
            "model_path": MODEL_PATH,
            "service_port": args.port,
            "service_host": args.host,
            "batch_size": asr_instance.batch_size,
            "max_queue_size": asr_instance.max_queue_size
        },
        "queue_status": {
            "current_length": len(asr_instance.queue),
            "instance_name": asr_instance.name
        }
    }


if __name__ == '__main__':
    # 1. 加载模型（使用命令行指定的路径）
    load_model()

    # 2. 启动FastAPI服务（使用命令行指定的主机和端口）
    import uvicorn
    print(f"启动单队列ASR服务...\n监听地址：{args.host}:{args.port}\n模型路径：{MODEL_PATH}")
    uvicorn.run(
        app,
        host=args.host,       # 命令行指定的主机（默认0.0.0.0）
        port=args.port,       # 命令行指定的端口（默认8023）
        workers=1,            # 单进程：避免多进程模型重复加载
        reload=False,         # 生产环境禁用自动重载
        timeout_keep_alive=120
    )