# FireRedASR-vLLM-ez
An easy implementation of vLLM based on the FireRedASR project

#快速开始

#1.克隆仓库
首先，克隆本项目到本地
git clone https://github.com/burcelong/FireRedASR-vLLM-ez.git
cd FireRedASR-vLLM-ez

#2.创建并激活虚拟环境
# 使用conda创建虚拟环境（需提前安装conda）
conda create -n fireredasr-vllm python=3.10 -y
conda activate fireredasr-vllm

# 或使用venv（Python内置）
python -m venv venv
# Windows激活
venv\Scripts\activate
# Linux/Mac激活
source venv/bin/activate

#3. 安装依赖
pip install -r requirements.txt