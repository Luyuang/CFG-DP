FROM continuumio/miniconda3:latest

# 配置 APT 清华源
RUN mkdir -p /etc/apt && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# 配置 Conda 清华源
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2 && \
    conda config --set show_channel_urls yes
ARG ENV_NAME=venv_lerobot
RUN conda create -n $ENV_NAME python=3.10.16 -y
ENV PATH /opt/conda/envs/$ENV_NAME/bin:$PATH
# 安装 FFmpeg (通过 Conda)
RUN conda install -y -c conda-forge ffmpeg
WORKDIR /workspace
COPY . /workspace/kuavo_il/
# 安装 Python 依赖
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --upgrade pip && \
    pip install lerobot bagpy && \  
    cd /workspace/kuavo_il/lerobot && \
    pip install -e .
RUN pip install opencv-python-headless
# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace


    