# FireRedASR-vLLM-ez
An easy implementation of vLLM based on the FireRedASR project

## 介绍
目前lm支持的ASR模型只有Wisper, 但是Whisper的中文推理效果不是很好，其他优秀的中文开源ASR模型基本都不支持vllm部署。
本项目参考[@GeeeekExplorer]https://github.com/GeeeekExplorer/nano-vllm 大佬的nano-vllm项目，利用flash-attn2简单实现了vllm的kvcache算法。
由于ASR任务通常都是短序列输入，本项目没有实现pageattention, 统一根据解码最大长度预分配缓存空间。用FireRedAsr官方音频测试的推理速度结果如下：
| 测试版本 | 音频时长 | 平均响应时间（秒） | 提升百分比  |
|----------|----------|--------------------|-------------|
| 提速版   | 4s       | 0.1467             | 25.04%      |
| 原版     | 4s       | 0.1957             | -           |
| 提速版   | 12s      | 0.5386             | 30.68%      |
| 原版     | 12s      | 0.7770             | -           |



## 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/burcelong/FireRedASR-vLLM-ez.git
cd FireRedASR-vLLM-ez
```
### 2. 创建并激活虚拟环境
```bash
conda create -n fireredasr-vllm python=3.10 -y
conda activate fireredasr-vllm

# 或使用venv（Python内置）
python -m venv venv

# Linux/Mac激活
source venv/bin/activate
```
### 3.安装依赖
```bash
pip install -r requirements.txt
```
### 4.在线推理api(测试推理速度)
```bash
# 使用默认端口启动
python main.py --model-path /path/to/your/model

# 指定端口启动
python main.py -m /path/to/your/model -p 8080

# 指定主机和端口
python main.py -m /path/to/your/model -H 127.0.0.1 -p 8000

# 请求样例
   curl -X POST "http://localhost:8000/asr/single" \
   -F "audio_file=@/path/to/your/audio.wav"
```
### 5.模仿vllm动态批处理结果
    通过借鉴 VLLM 动态批处理的核心设计思路，我们构建了请求队列缓存机制：将并发抵达的多个语音识别请求暂存于队列中，根据任务积累量与超时策略动态合并为批量推理任务，实现计算资源的高效复用。基于大量真实项目场景下的语音数据开展压力测试，在单张 RTX 4090 显卡硬件环境中，该方案实测语音识别处理吞吐量达 38 请求 / 秒！没经过推理优化的传统api部署压测结果只有19请求/秒！吞吐几乎多了一倍！
