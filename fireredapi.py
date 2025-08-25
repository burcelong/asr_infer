import torch
import argparse
import os
import traceback
import threading
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from fireredasr.models.fireredasr import FireRedAsr
import tempfile

# 解析命令行参数
parser = argparse.ArgumentParser(description='FireRedASR API服务')
parser.add_argument('--model-path', '-m', required=True, help='模型文件路径')
parser.add_argument('--port', '-p', type=int, default=8025, help='服务端口，默认8025')
parser.add_argument('--host', '-H', default='0.0.0.0', help='服务主机地址，默认0.0.0.0')
args = parser.parse_args()

# 模型路径配置（从命令行参数获取）
model_path = args.model_path

# 验证路径是否存在
if not os.path.exists(model_path):
    print(f"错误: 模型路径不存在 - {model_path}")
    exit(1)

# 添加安全全局对象以解决PyTorch 2.6的加载问题
torch.serialization.add_safe_globals([argparse.Namespace])

# 模型加载与线程锁
model = None
model_lock = threading.Lock()  # 线程锁，确保并发安全

# 请求计数器
request_counter = 0
counter_lock = threading.Lock()
app_start_time = time.time()

# 请求模型
class ASRRequest(BaseModel):
    beam_size: int = 5
    nbest: int = 1
    decode_max_len: int = 0
    softmax_smoothing: float = 1.0
    aed_length_penalty: float = 0.6
    eos_penalty: float = 1.0
    use_gpu: int = 1  # 0表示使用CPU，1表示使用GPU

# 创建FastAPI应用
app = FastAPI(title="FireRedASR API", description="基于FireRedASR的语音识别API服务")

# 加载模型
def load_model():
    global model
    print(f"正在从 {model_path} 加载模型...")
    try:
        with model_lock:
            model = FireRedAsr.from_pretrained("aed", model_path)
        print("模型加载完成！")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        exit(1)

# 单文件语音识别接口
@app.post("/asr/single", response_model=Dict[str, Any])
async def single_file_asr(audio_file: UploadFile = File(...), config: Optional[ASRRequest] = None):
    global request_counter
    
    # 检查模型是否加载
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载，请稍后再试")
    
    # 请求计数
    with counter_lock:
        request_counter += 1
        current_count = request_counter
    
    try:
        start_time = time.time()
        # 使用默认配置
        if config is None:
            config = ASRRequest()
        
        # 创建临时文件保存上传的音频
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = temp_file.name
            contents = await audio_file.read()
            temp_file.write(contents)
        
        # 执行语音识别
        batch_uttid = [audio_file.filename]
        batch_wav_path = [temp_path]
        
        with model_lock:
            results = model.transcribe(
                batch_uttid,
                batch_wav_path,
                {
                    "use_gpu": config.use_gpu,
                    "beam_size": config.beam_size,
                    "nbest": config.nbest,
                    "decode_max_len": config.decode_max_len,
                    "softmax_smoothing": config.softmax_smoothing,
                    "aed_length_penalty": config.aed_length_penalty,
                    "eos_penalty": config.eos_penalty
                }
            )
        
        process_time = time.time() - start_time
        return {
            "filename": audio_file.filename,
            "results": results,
            "status": "success",
            "request_id": current_count,
            "process_time": f"{process_time:.4f}秒"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "request_id": current_count,
            "traceback": traceback.format_exc(),
            "status": "error"
        }, 500
    finally:
        # 清理临时文件
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        # 减少请求计数
        with counter_lock:
            request_counter -= 1

# 批量语音识别接口
@app.post("/asr/batch", response_model=Dict[str, Any])
async def batch_file_asr(audio_files: List[UploadFile] = File(...), config: Optional[ASRRequest] = None):
    global request_counter
    
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载，请稍后再试")
    
    with counter_lock:
        request_counter += 1
        current_count = request_counter
    
    try:
        start_time = time.time()
        if config is None:
            config = ASRRequest()
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        batch_uttid = []
        batch_wav_path = []
        
        # 保存所有上传的音频文件
        for audio_file in audio_files:
            temp_path = os.path.join(temp_dir, audio_file.filename)
            with open(temp_path, "wb") as f:
                f.write(await audio_file.read())
            batch_uttid.append(audio_file.filename)
            batch_wav_path.append(temp_path)
        
        # 执行语音识别
        with model_lock:
            results = model.transcribe(
                batch_uttid,
                batch_wav_path,
                {
                    "use_gpu": config.use_gpu,
                    "beam_size": config.beam_size,
                    "nbest": config.nbest,
                    "decode_max_len": config.decode_max_len,
                    "softmax_smoothing": config.softmax_smoothing,
                    "aed_length_penalty": config.aed_length_penalty,
                    "eos_penalty": config.eos_penalty
                }
            )
        
        process_time = time.time() - start_time
        return {
            "total_files": len(audio_files),
            "results": results,
            "status": "success",
            "request_id": current_count,
            "process_time": f"{process_time:.4f}秒"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "request_id": current_count,
            "traceback": traceback.format_exc(),
            "status": "error"
        }, 500
    finally:
        # 清理临时文件和目录
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            for file_path in batch_wav_path:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            os.rmdir(temp_dir)
        # 减少请求计数
        with counter_lock:
            request_counter -= 1

if __name__ == '__main__':
    # 先加载模型
    load_model()
    
    # 启动服务（使用命令行参数中的端口和主机）
    import uvicorn
    print(f"启动API服务，地址: {args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,  # 单进程，通过线程锁保证并发安全
        reload=False,
        timeout_keep_alive=120
    )
