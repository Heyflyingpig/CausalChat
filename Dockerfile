# CausalChat Docker 镜像构建文件

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    default-libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 先安装基础依赖（很少变化）
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 再安装所有依赖（包括新增的）
COPY requirements.txt .
# 使用官方PyPI源避免哈希验证问题（torch等大包从官方源下载更可靠）
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 5001

CMD ["python", "Causalchat.py"]

