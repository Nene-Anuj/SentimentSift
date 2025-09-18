from airflow import DAG
from airflow.operators.python import ShortCircuitOperator, PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
from pathlib import Path

import json
import os
import subprocess

from airflow.hooks.base import BaseHook

from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
def sync_to_snowflake(**kwargs):
    """
    使用 SnowflakeHook 连接到 Snowflake 并同步数据
    """
    try:
        # 获取 Snowflake 连接
        hook = SnowflakeHook(snowflake_conn_id="snowflake")
        conn = hook.get_conn()
        
        try:
            # 设置 JSON 文件路径
            json_file_path = f"{BASE_DIR}/data/merge/integrated_cafes.json"
            print(f"使用 JSON 文件: {json_file_path}")
            
            # 导入同步函数
            import sys
            import importlib.util
            
            # 使用绝对路径指定 snowflake_sync.py 文件
            script_path = f"{TASKS_DIR}/snowflake_sync.py"
            
            # 加载脚本作为模块
            spec = importlib.util.spec_from_file_location("snowflake_sync_module", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 从模块中获取函数
            sync_from_json_to_snowflake = module.sync_from_json_to_snowflake
            
            # 执行同步操作
            records_count = sync_from_json_to_snowflake(conn, json_file_path)
            print(f"成功同步 {records_count} 条记录到 Snowflake")
            
            return records_count  # 返回同步的记录数量（一个整数值）
        finally:
            # 确保连接关闭
            conn.close()
            
    except Exception as e:
        print(f"Snowflake 同步失败: {e}")
        import traceback
        traceback.print_exc()
        raise

# --- Constants ---------------------------------------------------------------
BASE_DIR   = "/opt/airflow"                    # container project root
DATA_DIR   = f"{BASE_DIR}/boston_cafes_data"   # mounted raw data
PROCESSED  = f"{BASE_DIR}/data/processed"      # processed output
TASKS_DIR  = f"{BASE_DIR}/tasks"               # all your python scripts

DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

# --- Helper functions --------------------------------------------------------
# need_update 函数（放在 DAG 文件里）
def need_update(**context) -> bool:
    cafes_file   = Path(DATA_DIR) / "boston_cafes.json"
    reviews_file = Path(DATA_DIR) / "all_reviews.json"

    if not cafes_file.exists() or not reviews_file.exists():
        return True

    last_exec_ts = context["ti"].xcom_pull(
        key="last_execution_time",
        task_ids="record_execution_time",     # 上一轮记录的时间戳
    )
    if not last_exec_ts:
        return True   # 第一次运行

    last_mtime = max(cafes_file.stat().st_mtime, reviews_file.stat().st_mtime)

    # 如果文件修改时间小于或等于上次成功处理时间 → 不更新
    return last_mtime > last_exec_ts


def record_execution_time(**context):
    """Cache current timestamp for potential future use."""
    ts = int(datetime.now().timestamp())
    context["ti"].xcom_push(key="last_execution_time", value=ts)

# --- DAG ---------------------------------------------------------------------
with DAG(
    dag_id="boston_cafes_workflow",
    description="Boston cafe ETL + sentiment/theme analysis + Snowflake sync",
    start_date=datetime(2025, 4, 1),
    schedule_interval="@daily",
    catchup=False,
    default_args=DEFAULT_ARGS,
) as dag:

    # 0. decide whether to proceed
    check_need_update = ShortCircuitOperator(
        task_id="check_need_update",
        python_callable=need_update,
    )

    # 1. optional: fetch fresh data (only when check_need_update == True)
    data_fetch = BashOperator(
        task_id="data_fetch",
        bash_command=f"python {TASKS_DIR}/data_fetch.py",
    )

    # 2. record timestamp after fetch
    record_time = PythonOperator(
        task_id="record_execution_time",
        python_callable=record_execution_time,
    )

    # 3. process raw data to structured form
    data_process = BashOperator(
        task_id="data_process",
        bash_command=f"""
python {TASKS_DIR}/data_process.py \
  --business_json {DATA_DIR}/boston_cafes.json \
  --reviews_dir   {DATA_DIR} \
  --output_dir    {PROCESSED}
""",
    )
    nltk_setup = BashOperator(
    task_id="nltk_setup",
    bash_command="""
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
    """,
)

    # 4. sentiment analysis
    sentiment_analysis = BashOperator(
    task_id="sentiment_analysis",
    bash_command=f"cd {BASE_DIR} && python {TASKS_DIR}/sentiment_analysis.py",
)

    # 5. theme/topic analysis
    theme_analysis = BashOperator(
    task_id="theme_analysis",
    bash_command=f"cd {BASE_DIR} && python {TASKS_DIR}/theme_analysis.py --output_dir {BASE_DIR}/data/trend_data",
)

    # 6. merge results
    merge = BashOperator(
    task_id="merge",
    bash_command=f"cd {BASE_DIR} && python {TASKS_DIR}/merge.py",
)

    # 7. sync to Snowflake
    snowflake_sync = PythonOperator(
        task_id="snowflake_sync",
        python_callable=sync_to_snowflake,  # 使用新函数名
)

    # --- Dependencies --------------------------------------------------------
    check_need_update >> data_fetch >> data_process >> nltk_setup >> sentiment_analysis >> theme_analysis >> merge >> snowflake_sync >> record_time