# ===== token_runner_api.py (fixed: start runner inside FastAPI event loop) =====
import os
import argparse
import tempfile
import time
import traceback
from typing import Dict, Any, List, Optional, Tuple

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from fireredasr.models.fireredasr import FireRedAsr
from fireredasr.models.module.varlen_decoder import TransformerDecoder
from fireredasr.models.module.conformer_encoder import ConformerEncoder  # 仅类型提示

from fireredasr.utils.KVCacheManager import KVSpec,KVCacheManager
from runtime.kv_adapter import LayeredKVCacheManager
from runtime.runner_dynamic import TokenRunner
from runtime.step_impl import Stepper, build_enc_state_for_request

# ---------------------------
# CLI
# ---------------------------
parser = argparse.ArgumentParser(description="FireRedASR token-level dynamic batching API")
parser.add_argument("--model-path", "-m", default="/root/autodl-tmp/tenxun/FireRedASR-AED-L", help="模型目录/权重路径")
parser.add_argument("--port", "-p", type=int, default=8025, help="端口，默认8025")
parser.add_argument("--host", "-H", default="0.0.0.0", help="主机地址，默认0.0.0.0")
parser.add_argument("--slots", type=int, default=8, help="KV池最大并发序列数（SLOTS）")
parser.add_argument("--cap", type=int, default=128, help="每序列最大解码步（CAP）")
parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16", help="KV/FA计算精度")
parser.add_argument("--device", default="cuda", help="设备：cuda/cuda:0/cpu（FlashAttention 需要 CUDA）")
args = parser.parse_args()

# ---------------------------
# FastAPI
# ---------------------------
app = FastAPI(
    title="FireRedASR TokenRunner API",
    description="vLLM-style continuous batching (token-level) for FireRedASR",
)

# ---------------------------
# 全局对象
# ---------------------------
asr: Optional[FireRedAsr] = None            # 包装类：含 feat_extractor / tokenizer / model(FireRedAsrAed)
decoder: Optional[TransformerDecoder] = None
runner: Optional[TokenRunner] = None
kv: Optional[KVCacheManager] = None
_device: Optional[torch.device] = None

# ---------------------------
# 配置
# ---------------------------
class ASRRequest(BaseModel):
    max_len: int = 256
    bos_id: Optional[int] = None
    eos_id: Optional[int] = None
    softmax_smoothing: float = 1.0
    length_penalty: float = 0.0
    eos_penalty: float = 1.0

def _dtype_of(s: str) -> torch.dtype:
    return torch.float16 if s == "fp16" else torch.bfloat16

# ---------------------------
# Encoder：走 FireRedAsr 的特征器 + FireRedAsrAed.encoding()
# ---------------------------
def _run_encoder_strict(wrapper: FireRedAsr, wav_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    feats, lengths = wrapper.feat_extractor([wav_path])
    enc_outputs, enc_mask, _ = wrapper.model.encoding(feats, lengths)
    返回:
      enc_outputs: [1, Ti, d_model]
      enc_mask:    [1, Ti]（True/1=有效）
    """
    device = next(wrapper.model.parameters()).device
    feats, lengths, _ = wrapper.feat_extractor([wav_path])    # [1, T', F], [1]
    feats = feats.to(device)
    lengths = lengths.to(device)

    enc_outputs, enc_mask, _ = wrapper.model.encoding(feats, lengths)
    if enc_mask.dim() == 3:
        enc_mask = enc_mask[:, 0, :]
    enc_mask = enc_mask.to(torch.bool)
    return enc_outputs, enc_mask

# ---------------------------
# FastAPI 事件（在事件循环里启动/停止 Runner）
# ---------------------------
@app.on_event("startup")
async def _startup():
    global asr, decoder, kv, runner, _device
    print(f"[Init] 从 {args.model_path} 加载 FireRedAsr …")

    torch.serialization.add_safe_globals([argparse.Namespace])
    _device = torch.device(args.device if args.device != "cuda" else ("cuda" if torch.cuda.is_available() else "cpu"))

    # 1) 加载包装类（内部加载 AED 模型 / tokenizer / 特征器）
    asr = FireRedAsr.from_pretrained("aed", args.model_path, device=_device)

    # 2) 取底层 AED 模型对象
    base_model = asr.model
    if not hasattr(base_model, "decoder") or not hasattr(base_model, "encoding"):
        raise RuntimeError("底层 AED 模型缺少 decoder / encoding() 接口，请检查 FireRedAsrAed 实现。")

    dec = base_model.decoder
    dec.eval()
    print(f"[Init] decoder: n_head={dec.n_head}, d_k={dec.d_k}, d_model={dec.d_model}, n_layers={dec.n_layers}")

    # 3) KV 池
    spec = KVSpec(
        slots=args.slots,
        n_heads=dec.n_head,
        cap=args.cap,
        head_dim=dec.d_k,
        dtype=_dtype_of(args.dtype),
        device=str(_device),
    )
    kv_mod = LayeredKVCacheManager(spec, n_layers=dec.n_layers)
    print(f"[Init] KVCache: SLOTS={spec.slots}, CAP={spec.cap}, H={spec.n_heads}, D={spec.head_dim}, dtype={spec.dtype}, device={spec.device}")

    # 4) Stepper & Runner（！！现在在事件循环里，可以安全 start）
    stepper = Stepper(decoder=dec, kv=kv_mod, device=str(_device))
    run = TokenRunner(kv=kv_mod, step_fn=stepper, bos_id=dec.sos_id, eos_id=dec.eos_id, join_new_on_next_step=True)
    run._fn_append_batch = lambda *a, **k: None   # Stepper 内部已写 KV + bump
    run.start()  # 此处已有 running loop，create_task 不会报错

    decoder = dec
    kv = kv_mod
    runner = run
    print("[Init] Runner ready. Continuous batching enabled.")

@app.on_event("shutdown")
async def _shutdown():
    # 尽量优雅地停止 Runner（兼容不同实现）
    if runner is None:
        return
    try:
        if hasattr(runner, "stop"):
            res = runner.stop()
            # 如果是协程，await 一下
            import inspect, asyncio
            if inspect.iscoroutine(res):
                await res
        elif hasattr(runner, "close"):
            res = runner.close()
            import inspect
            if inspect.iscoroutine(res):
                await res
        elif hasattr(runner, "_task"):
            runner._task.cancel()
    except Exception as e:
        print(f"[Shutdown] runner stop error: {e}")

# ---------------------------
# API 路由
# ---------------------------
@app.get("/health")
async def health():
    ok = (asr is not None) and (decoder is not None) and (runner is not None)
    return {
        "ok": ok,
        "slots": getattr(getattr(kv, "spec", None), "slots", None),
        "cap": getattr(getattr(kv, "spec", None), "cap", None),
        "dtype": str(getattr(getattr(kv, "spec", None), "dtype", None)),
    }

class ASRRequest(BaseModel):
    max_len: int = 256
    bos_id: Optional[int] = None
    eos_id: Optional[int] = None
    softmax_smoothing: float = 1.0
    length_penalty: float = 0.0
    eos_penalty: float = 1.0

@app.post("/asr/single")
async def asr_single(audio_file: UploadFile = File(...), config: Optional[ASRRequest] = None) -> Dict[str, Any]:
    if asr is None or decoder is None or runner is None:
        raise HTTPException(status_code=503, detail="服务未就绪")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename or "")[-1] or ".wav") as tf:
        wav_path = tf.name
        tf.write(await audio_file.read())

    try:
        t0 = time.time()
        enc_out, src_mask = _run_encoder_strict(asr, wav_path)  # [1, Ti, d_model], [1, Ti]
        enc_state = build_enc_state_for_request(decoder, enc_out, src_mask)

        cfg = config or ASRRequest()
        bos = cfg.bos_id if (cfg and cfg.bos_id is not None) else decoder.sos_id
        eos = cfg.eos_id if (cfg and cfg.eos_id is not None) else decoder.eos_id

        fut = await runner.submit(enc=enc_state, max_len=cfg.max_len, bos_id=bos, eos_id=eos, filename=audio_file.filename)
        result = await fut
        ids: List[int] = result["tokens"]

        text = asr.tokenizer.detokenize(ids) if hasattr(asr.tokenizer, "detokenize") else (
            asr.tokenizer.decode(ids) if hasattr(asr.tokenizer, "decode") else " ".join(map(str, ids))
        )

        t1 = time.time()
        return {
            "filename": audio_file.filename,
            "tokens": ids,
            "text": text,
            "elapsed_sec": round(t1 - t0, 4),
            "max_len": cfg.max_len,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"ASR失败: {e}")
    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass

@app.post("/asr/batch")
async def asr_batch(audio_files: List[UploadFile] = File(...), config: Optional[ASRRequest] = None) -> Dict[str, Any]:
    if asr is None or decoder is None or runner is None:
        raise HTTPException(status_code=503, detail="服务未就绪")

    results = []
    cfg = config or ASRRequest()
    bos = cfg.bos_id if (cfg and cfg.bos_id is not None) else decoder.sos_id
    eos = cfg.eos_id if (cfg and cfg.eos_id is not None) else decoder.eos_id

    temps: List[str] = []
    try:
        for uf in audio_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uf.filename or "")[-1] or ".wav") as tf:
                tf.write(await uf.read())
                temps.append(tf.name)

        t0 = time.time()
        futs = []
        for (uf, path) in zip(audio_files, temps):
            enc_out, src_mask = _run_encoder_strict(asr, path)
            enc_state = build_enc_state_for_request(decoder, enc_out, src_mask)
            fut = await runner.submit(enc=enc_state, max_len=cfg.max_len, bos_id=bos, eos_id=eos, filename=uf.filename)
            futs.append((uf.filename, fut))

        out = []
        for fname, fut in futs:
            r = await fut
            ids = r["tokens"]
            text = asr.tokenizer.detokenize(ids) if hasattr(asr.tokenizer, "detokenize") else (
                asr.tokenizer.decode(ids) if hasattr(asr.tokenizer, "decode") else " ".join(map(str, ids))
            )
            out.append({"filename": fname, "tokens": ids, "text": text})

        t1 = time.time()
        return {"num_files": len(audio_files), "results": out, "elapsed_sec": round(t1 - t0, 4)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Batch ASR失败: {e}")
    finally:
        for p in temps:
            try:
                os.unlink(p)
            except Exception:
                pass

# ---------------------------
# 入口
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    print(f"[Start] http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, workers=1, reload=False, timeout_keep_alive=120)

