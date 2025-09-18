FROM apache/airflow:2.10.4

USER root
# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    gcc \
    g++ \
    libsndfile1

USER airflow
# 安装 Python 依赖
RUN pip install --no-cache-dir \
    bertopic \
    nltk \
    numpy \
    pandas \
    python-dotenv \
    requests \
    scikit_learn \
    scipy \
    sentence_transformers \
    snowflake_connector_python \
    snowflake_snowpark_python \
    torch \
    tqdm \
    transformers