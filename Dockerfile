# AlphaZero Chess AI Docker 配置
# 基於 NVIDIA CUDA 映像以支援 GPU 加速
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# 設置非互動模式，避免安裝時的提示
ENV DEBIAN_FRONTEND=noninteractive

# 設置工作目錄
WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    graphviz \
    graphviz-dev \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# 建立 Python 3.9 的符號連結
RUN ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.9 /usr/bin/python

# 升級 pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# 複製需求檔案
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip3 install --no-cache-dir -r requirements.txt

# 安裝額外的開發工具
RUN pip3 install --no-cache-dir \
    jupyter \
    jupyterlab \
    tensorboard \
    black \
    flake8 \
    pytest \
    mypy

# 複製專案檔案
COPY . .

# 安裝專案本身
RUN pip3 install -e .

# 建立必要的目錄
RUN mkdir -p /app/checkpoints /app/logs /app/games /app/data

# 設置環境變數
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 建立非 root 使用者
RUN useradd -m -s /bin/bash alphazero && \
    chown -R alphazero:alphazero /app

# 切換到非 root 使用者
USER alphazero

# 暴露端口
EXPOSE 8888 6006

# 設置啟動命令
CMD ["python3", "main.py"]