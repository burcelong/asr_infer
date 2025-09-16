import argparse, asyncio, time, json, statistics, os
from typing import List, Dict, Any
import aiohttp

def fmt(s): return f"{s:.4f}"

# -------------------------
# HTTP helpers
# -------------------------
async def post_file(session, url: str, wav_path: str) -> Dict[str, Any]:
    """
    发送单个文件到 {url}/asr/single
    返回：status/resp/wall/filename/client_start/client_end 等
    """
    filename = os.path.basename(wav_path)
    form = aiohttp.FormData()
    with open(wav_path, "rb") as f:
        form.add_field("audio_file", f, filename=filename, content_type="audio/wav")
        t0 = time.perf_counter()
        async with session.post(f"{url}/asr/single", data=form) as resp:
            client_status = resp.status
            client_end = None
            try:
                js = await resp.json(content_type=None)
            except Exception:
                # 非 JSON 时兜底读取文本
                txt = await resp.text()
                try:
                    js = json.loads(txt)
                except Exception:
                    js = {"error": txt}
            finally:
                client_end = time.perf_counter()
    return {
        "status": client_status,
        "resp": js,
        "wall": client_end - t0,
        "filename": filename,
        "client_start": t0,
        "client_end": client_end
    }

async def run_sequential(url: str, files: List[str]) -> List[Dict[str, Any]]:
    out = []
    async with aiohttp.ClientSession(raise_for_status=False) as session:
        for f in files:
            out.append(await post_file(session, url, f))
    return out

async def run_concurrent(url: str, files: List[str], stagger: float) -> List[Dict[str, Any]]:
    """
    并发+错峰：第 i 个请求在 i*stagger 秒后发出
    """
    results = [None] * len(files)
    async with aiohttp.ClientSession(raise_for_status=False) as session:
        start = time.perf_counter()
        async def one(i, path):
            await asyncio.sleep(i * stagger)
            tstart = time.perf_counter() - start
            r = await post_file(session, url, path)
            r["offset"] = i * stagger
            r["start_since"] = tstart
            results[i] = r
        await asyncio.gather(*[one(i, p) for i, p in enumerate(files)])
    return results

# -------------------------
# Stats helpers
# -------------------------
def percentile(sorted_vals, p: float) -> float:
    """线性插值的分位数，p∈[0,1]"""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)

def summarise(tag: str, arr: List[Dict[str, Any]]):
    # 1) 基础延迟（客户端视角）
    lats = [x["wall"] for x in arr]
    sum_lat = sum(lats)
    client_first_start = min((x["client_start"] for x in arr), default=0.0)
    client_last_end = max((x["client_end"] for x in arr), default=0.0)
    wall_total = max(0.0, client_last_end - client_first_start)

    sorted_lats = sorted(lats)
    p50 = percentile(sorted_lats, 0.50) if sorted_lats else 0.0
    p95 = percentile(sorted_lats, 0.95) if sorted_lats else 0.0

    # 2) 服务端效率（按识别文本长度 & 顶层 process_time）
    total_text_len = 0
    server_time_list = []
    char_per_req = []

    for x in arr:
        js = x["resp"]
        if (isinstance(js, dict)
            and js.get("status") == "success"
            and isinstance(js.get("results"), dict)):
            # 顶层 process_time，如 "0.5928秒"
            process_time_str = js.get("process_time", "0秒")
            try:
                srv_elapsed = float(process_time_str.replace("秒", ""))
            except (ValueError, AttributeError):
                srv_elapsed = 0.0

            asr_text = js["results"].get("text", "")
            text_len = len(asr_text.strip())

            if srv_elapsed > 0:
                total_text_len += text_len
                server_time_list.append(srv_elapsed)
                char_per_req.append(text_len / srv_elapsed)

    # 3) 整体吞吐量（字符/秒）：用总墙钟近似窗口
    server_total_time = wall_total if server_time_list else 0.0

    # 4) 输出指标
    print(f"\n[{tag}] 基础性能：")
    print(f"  任务数={len(arr)}  总墙钟={fmt(wall_total)}s  Σ单次延迟={fmt(sum_lat)}s")
    print(f"  p50延迟={fmt(p50)}s  p95延迟={fmt(p95)}s  有效并行度≈{fmt(sum_lat / wall_total if wall_total>0 else 0.0)}")

    print(f"\n[{tag}] 服务端效率（基于识别文本）：")
    if total_text_len > 0 and server_total_time > 0:
        overall_char_s = total_text_len / server_total_time
        print(f"  整体吞吐量≈{fmt(overall_char_s)} 字符/秒  总识别长度={total_text_len} 字符  服务端总耗时(近似)={fmt(server_total_time)}s")
    else:
        print("  整体吞吐量：无法计算（缺少有效识别文本或服务端耗时）")
    if char_per_req:
        avg_char_s = sum(char_per_req) / len(char_per_req)
        print(f"  单请求平均≈{fmt(avg_char_s)} 字符/秒  有效请求数={len(char_per_req)}")
    else:
        print("  单请求平均：无法计算（缺少有效服务端数据）")

    # 5) 请求明细
    print(f"\n[{tag}] 请求明细：")
    for i, x in enumerate(arr):
        js = x["resp"]
        filename = x.get("filename", "未知文件")
        http_status = x["status"]

        is_success = (
            http_status == 200 and
            isinstance(js, dict) and
            js.get("status") == "success" and
            isinstance(js.get("results"), dict) and
            "text" in js["results"]
        )

        if isinstance(js, dict) and isinstance(js.get("results"), dict):
            process_time   = js.get("process_time", "N/A")       # 顶层
            asr_text       = js["results"].get("text", "无识别结果")
            asr_score      = js["results"].get("score", "N/A")
            audio_duration = js.get("audio_duration", "N/A")     # 顶层
            uttid          = js["results"].get("uttid", "N/A")
            text_len       = len(asr_text.strip())
        else:
            process_time   = "N/A"
            asr_text       = "响应格式异常"
            asr_score      = "N/A"
            audio_duration = "N/A"
            uttid          = "N/A"
            text_len       = 0

        try:
            srv_elapsed = float(process_time.replace("秒", "")) if process_time != "N/A" else 0.0
            char_s = fmt(text_len / srv_elapsed) if srv_elapsed > 0 else "N/A"
        except (ValueError, AttributeError):
            char_s = "N/A"

        print(f"  #{i:02d} {filename}：")
        print(f"    基础状态：HTTP={http_status}  客户端耗时={fmt(x['wall'])}s  整体成功={is_success}")
        print(f"    服务端信息：处理时间={process_time}  音频时长={audio_duration}  识别置信度={asr_score}  uttid={uttid}")
        if len(asr_text) > 50:
            preview = asr_text[:50] + "..."
        else:
            preview = asr_text
        print(f"    识别结果：文本长度={text_len} 字符  字符/秒={char_s}  识别文本：{preview}")
        if "offset" in x and "start_since" in x:
            print(f"    错峰配置：偏移={fmt(x['offset'])}s  启动时刻={fmt(x['start_since'])}s")
        print("  " + "-"*80)

# -------------------------
# Warmup (optional)
# -------------------------
async def warmup(url: str, wav_path: str):
    """一次预热，不计入统计（建立连接/JIT kernel）"""
    async with aiohttp.ClientSession(raise_for_status=False) as session:
        try:
            await post_file(session, url, wav_path)
        except Exception:
            pass

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="ASR服务性能测试工具（适配8024/8025响应）")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="服务根地址（不含路径，如 http://127.0.0.1:8000 或 :8025）")
    ap.add_argument("--files", nargs="+", required=True, help="音频文件列表（WAV）")
    ap.add_argument("--stagger", type=float, default=0.1, help="并发错峰间隔（秒）")
    ap.add_argument("--warmup", action="store_true", help="先做一次预热（建议打开）")
    args = ap.parse_args()

    # 校验文件
    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"错误：音频文件不存在 → {file_path}")
            return
        if not file_path.lower().endswith(".wav"):
            print(f"警告：文件不是 WAV → {file_path}（建议仅用 WAV）")

    # 预热（可选）
    if args.warmup:
        print("="*80)
        print("→ 预热一次（不计入统计）")
        print("="*80)
        asyncio.run(warmup(args.url, args.files[0]))

    # 顺序基线
    print("="*80)
    print("→ 顺序基线测试（逐个发送，无并发）")
    print("="*80)
    try:
        seq_results = asyncio.run(run_sequential(args.url, args.files))
    except Exception as e:
        print(f"顺序测试失败：{str(e)}")
        return
    summarise("sequential", seq_results)

    # 并发+错峰
    print("\n" + "="*80)
    print(f"→ 并发+错峰测试（错峰间隔={args.stagger}s）")
    print("="*80)
    try:
        conc_results = asyncio.run(run_concurrent(args.url, args.files, args.stagger))
    except Exception as e:
        print(f"并发测试失败：{str(e)}")
        return
    summarise("concurrent", conc_results)

    # 对比
    print("\n" + "="*80)
    print("→ 顺序 vs 并发 核心对比")
    print("="*80)
    seq_wall = (max((x["client_end"] for x in seq_results), default=0.0)
                - min((x["client_start"] for x in seq_results), default=0.0))
    seq_total_char = sum(
        len(x["resp"]["results"]["text"].strip())
        for x in seq_results
        if isinstance(x["resp"], dict)
        and x["resp"].get("status") == "success"
        and isinstance(x["resp"].get("results"), dict)
        and "text" in x["resp"]["results"]
    )

    conc_wall = (max((x["client_end"] for x in conc_results), default=0.0)
                 - min((x["client_start"] for x in conc_results), default=0.0))
    conc_total_char = sum(
        len(x["resp"]["results"]["text"].strip())
        for x in conc_results
        if isinstance(x["resp"], dict)
        and x["resp"].get("status") == "success"
        and isinstance(x["resp"].get("results"), dict)
        and "text" in x["resp"]["results"]
    )

    speedup_ratio = (seq_wall / conc_wall) if conc_wall > 0 else 0.0
    seq_char_s = (seq_total_char / seq_wall) if (seq_total_char > 0 and seq_wall > 0) else 0.0
    conc_char_s = (conc_total_char / conc_wall) if (conc_total_char > 0 and conc_wall > 0) else 0.0
    throughput_ratio = (conc_char_s / seq_char_s) if seq_char_s > 0 else 0.0

    print(f"1. 总墙钟时间：顺序 {fmt(seq_wall)}s  |  并发 {fmt(conc_wall)}s  |  加速比 {fmt(speedup_ratio)}x")
    print(f"2. 总识别文本：顺序 {seq_total_char} 字符  |  并发 {conc_total_char} 字符  |  Δ={conc_total_char - seq_total_char}")
    print(f"3. 整体吞吐量（字符/秒）：顺序 {fmt(seq_char_s)}  |  并发 {fmt(conc_char_s)}  |  提升 {fmt(throughput_ratio)}x")

if __name__ == "__main__":
    main()