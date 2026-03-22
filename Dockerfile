# ============================================================
# 边缘端交通语义场景图构建系统 - Docker 镜像
# ============================================================
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL maintainer="traffic-scene-graph"
LABEL description="Edge-side Traffic Semantic Scene Graph Construction"

# 避免交互式安装提示
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 替换为阿里云镜像源，避免部分代理软件（如 Clash TUN 模式）导致的 网络/Fake-IP 路由问题，同时加速国内下载
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list \
    && sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 配置 pip 使用清华镜像源加速
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装系统依赖 (包括 OpenCV 所需的共享库)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb1 \
    libx11-6 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制依赖文件以利用 Docker 缓存
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 安装 PyTorch Geometric 及其依赖
RUN pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
    && pip install --no-cache-dir torch-geometric

# 复制项目源码
COPY . .

# 安装项目本身(可编辑模式)
RUN pip install --no-cache-dir -e .

# 默认执行训练脚本
CMD ["python", "scripts/train.py", "--config", "configs/default.yaml"]
