import argparse, asyncio, time, json, statistics, os
from typing import List, Dict, Any
import aiohttp

def fmt(s): return f"{s:.4f}"

async def get_health(session, base):
    async with session.get(f"{base}/health") as r:
        r.raise_for_status()
        return await r.json()

async def post_file(session, url: str, wav_path: str) -> Dict[str, Any]:
    filename = os.path.basename(wav_path)
    form = aiohttp.FormData()
    # 用with语句显式关闭文件，避免资源泄漏
    with open(wav_path, "rb") as f:
        form.add_field("audio_file", f, filename=filename, content_type="audio/wav")
        t0 = time.perf_counter()
        async with session.post(f"{url}/asr/single", data=form) as resp:
            text = await resp.text()
            t1 = time.perf_counter()
            try:
                js = json.loads(text)
            except Exception:
                js = {"error": text}
            return {
                "status": resp.status,
                "resp": js,
                "wall": t1 - t0,  # 客户端感知总耗时（网络+服务端）
                "filename": filename,
                # 记录客户端请求开始/结束的绝对时间
                "client_start": t0,
                "client_end": t1
            }

async def run_sequential(url: str, files: List[str]) -> List[Dict[str, Any]]:
    out = []
    async with aiohttp.ClientSession(raise_for_status=False) as session:
        for f in files:
            out.append(await post_file(session, url, f))
    return out

async def run_concurrent(url: str, files: List[str], stagger: float) -> List[Dict[str, Any]]:
    results = [None] * len(files)
    async with aiohttp.ClientSession(raise_for_status=False) as session:
        start = time.perf_counter()
        async def one(i, path):
            await asyncio.sleep(i * stagger)
            tstart = time.perf_counter() - start  # 相对开始时刻（错峰用）
            r = await post_file(session, url, path)
            r["offset"] = i * stagger  # 错峰偏移时间
            r["start_since"] = tstart  # 相对于测试开始的请求启动时间
            results[i] = r
        await asyncio.gather(*[one(i, p) for i, p in enumerate(files)])
    return results

def summarise(tag: str, arr: List[Dict[str, Any]]):
    # 1. 基础延迟指标
    lats = [x["wall"] for x in arr]
    sum_lat = sum(lats)
    # 计算总墙钟时间（客户端视角）
    client_first_start = min(x["client_start"] for x in arr)
    client_last_end = max(x["client_end"] for x in arr)
    wall_total = client_last_end - client_first_start
    # 分位数计算
    sorted_lats = sorted(lats)
    p50 = statistics.median(sorted_lats)
    p95 = sorted_lats[int(len(sorted_lats) * 0.95) - 1] if len(sorted_lats) > 1 else sorted_lats[0]

    # 2. 服务端效率指标
    total_tokens = 0  # 所有请求的总token数
    server_time_list = []  # 每个请求的服务端耗时
    toks_per_req = []  # 单请求的tok/s列表

    for x in arr:
        js = x["resp"]
        if isinstance(js, dict) and "tokens" in js and "elapsed_sec" in js:
            tok_len = len(js["tokens"])
            srv_elapsed = float(js["elapsed_sec"])
            total_tokens += tok_len
            server_time_list.append(srv_elapsed)
            if srv_elapsed > 0:
                toks_per_req.append(tok_len / srv_elapsed)

    # 3. 计算服务端整体吞吐量
    server_total_time = 0.0
    if server_time_list and len(arr) > 0:
        # 优先使用服务端返回的start_time（若有）
        if all(isinstance(x["resp"], dict) and "start_time" in x["resp"] for x in arr):
            server_first_start = min(float(x["resp"]["start_time"]) for x in arr)
            server_last_end = max(float(x["resp"]["start_time"]) + float(x["resp"]["elapsed_sec"]) for x in arr)
            server_total_time = server_last_end - server_first_start
        else:
            server_total_time = wall_total  # 用客户端总墙钟近似

    # 4. 输出指标
    print(f"\n[{tag}] 基础性能：")
    print(f"  任务数={len(arr)}  总墙钟={fmt(wall_total)}s  Σ单次延迟={fmt(sum_lat)}s")
    print(f"  p50延迟={fmt(p50)}s  p95延迟={fmt(p95)}s  有效并行度≈{fmt(sum_lat / wall_total if wall_total>0 else 0.0)}")
    '''
    print(f"\n[{tag}] 服务端效率：")
    if total_tokens > 0 and server_total_time > 0:
        overall_tok_s = total_tokens / server_total_time
        print(f"  整体吞吐量≈{fmt(overall_tok_s)} tok/s  总token数={total_tokens}  服务端总耗时={fmt(server_total_time)}s")
    else:
        print(f"  整体吞吐量：无法计算（缺少token数或服务端耗时数据）")
    if toks_per_req:
        avg_tok_s = sum(toks_per_req) / len(toks_per_req)
        print(f"  单请求平均tok/s≈{fmt(avg_tok_s)}  有效请求数={len(toks_per_req)}")
    else:
        print(f"  单请求平均tok/s：无法计算（缺少有效服务端数据）")

    # 5. 请求明细（修复遍历语法错误）
    print(f"\n[{tag}] 请求明细：")
    # 修正：使用enumerate获取索引和元素，而不是直接解包
    for i, x in enumerate(arr):
        js = x["resp"]
        ok = (x["status"] == 200) and isinstance(js, dict) and ("text" in js or "tokens" in js)
        srv_elapsed = js.get("elapsed_sec", None)
        tok_len = len(js.get("tokens", [])) if isinstance(js, dict) else 0
        # 计算单请求tok/s
        req_tok_s = f"{fmt(tok_len / float(srv_elapsed))}" if (srv_elapsed and float(srv_elapsed) > 0 and tok_len > 0) else "N/A"
        print(f"  #{i:02d} {x.get('filename','?')}:")
        print(f"    状态：ok={ok}  HTTP={x['status']}  客户端耗时={fmt(x['wall'])}s")
        print(f"    服务端：耗时={fmt(srv_elapsed)}s  token数={tok_len}  tok/s={req_tok_s}")
        if "offset" in x:
            print(f"    错峰：偏移={fmt(x['offset'])}s  启动时刻={fmt(x['start_since'])}s")
'''
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8025", help="服务根地址（不含路径）")
    ap.add_argument("--files", nargs="+", required=True, help="音频文件列表（需为WAV格式）")
    ap.add_argument("--stagger", type=float, default=0.1, help="并发错峰间隔（秒）")
    args = ap.parse_args()

    # 1. 顺序基线测试
    print("="*60)
    print("→ 开始顺序基线测试（逐个发送请求，无并发）")
    print("="*60)
    seq = asyncio.run(run_sequential(args.url, args.files))
    summarise("sequential", seq)

    # 2. 并发+错峰测试
    print("\n" + "="*60)
    print(f"→ 开始并发+错峰测试（错峰间隔={args.stagger}s）")
    print("="*60)
    conc = asyncio.run(run_concurrent(args.url, args.files, args.stagger))
    summarise("concurrent", conc)

    # 3. 跨模式对比
    print("\n" + "="*60)
    print("→ 顺序 vs 并发 对比总结")
    print("="*60)
    # 顺序模式核心指标
    seq_wall = max(x["client_end"] for x in seq) - min(x["client_start"] for x in seq) if seq else 0.0
    seq_total_tok = sum(len(x["resp"].get("tokens", [])) for x in seq if isinstance(x["resp"], dict))
    # 并发模式核心指标
    conc_wall = max(x["client_end"] for x in conc) - min(x["client_start"] for x in conc) if conc else 0.0
    conc_total_tok = sum(len(x["resp"].get("tokens", [])) for x in conc if isinstance(x["resp"], dict))
    # 输出对比
    print(f"1. 总墙钟时间：顺序={fmt(seq_wall)}s  并发={fmt(conc_wall)}s  加速比≈{fmt(seq_wall/conc_wall if conc_wall>0 else 0.0)}x")
    print(f"2. 总token数：顺序={seq_total_tok}  并发={conc_total_tok}")
    # 对比整体吞吐量
    seq_overall_tok_s = seq_total_tok / seq_wall if (seq_total_tok > 0 and seq_wall > 0) else 0.0
    conc_overall_tok_s = conc_total_tok / conc_wall if (conc_total_tok > 0 and conc_wall > 0) else 0.0
    print(f"3. 整体吞吐量：顺序={fmt(seq_overall_tok_s)} tok/s  并发={fmt(conc_overall_tok_s)} tok/s  提升比≈{fmt(conc_overall_tok_s/seq_overall_tok_s if seq_overall_tok_s>0 else 0.0)}x")

if __name__ == "__main__":
    main()
