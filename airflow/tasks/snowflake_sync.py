import pandas as pd
import numpy as np
import os
import sys
import json
import snowflake.connector
from typing import Dict, List, Any
from dotenv import load_dotenv
import re

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    从配置文件或环境变量加载Snowflake配置
    
    Args:
        config_path: 配置文件路径（可选）
        
    Returns:
        包含Snowflake配置的字典
    """
    # 优先从环境变量加载
    load_dotenv()
    
    # 检查环境变量是否存在
    if all(key in os.environ for key in [
        'SNOWFLAKE_ACCOUNT', 'SNOWFLAKE_USER', 'SNOWFLAKE_PASSWORD',
        'SNOWFLAKE_WAREHOUSE', 'SNOWFLAKE_DATABASE', 'SNOWFLAKE_SCHEMA'
    ]):
        return {
            'account': os.environ['SNOWFLAKE_ACCOUNT'],
            'user': os.environ['SNOWFLAKE_USER'],
            'password': os.environ['SNOWFLAKE_PASSWORD'],
            'warehouse': os.environ['SNOWFLAKE_WAREHOUSE'],
            'database': os.environ['SNOWFLAKE_DATABASE'],
            'schema': os.environ['SNOWFLAKE_SCHEMA']
        }
    
    # 如果环境变量不完整且提供了配置文件路径，从配置文件加载
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    # 如果环境变量和配置文件都不可用，抛出异常
    raise ValueError("无法加载Snowflake配置。请确保.env文件或配置文件存在并包含所需信息。")

def json_to_dataframe(json_file_path: str) -> pd.DataFrame:
    """
    将JSON文件加载为DataFrame，并按照要求处理数据
    
    Args:
        json_file_path: JSON文件路径
        
    Returns:
        包含处理后数据的DataFrame
    """
    try:
        # 读取JSON文件
        if json_file_path.endswith('.json'):
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        print(f"警告: 文件 {json_file_path} 为空")
                        return pd.DataFrame()
                    
                    # 检查文件格式
                    if content.startswith('[') and content.endswith(']'):
                        # 处理JSON数组
                        try:
                            data = json.loads(content)
                            print(f"成功加载JSON数组，包含 {len(data)} 个对象")
                        except json.JSONDecodeError as e:
                            print(f"解析JSON数组时出错: {str(e)}")
                            # 尝试修复常见JSON问题
                            if content.endswith(',]'):
                                content = content[:-2] + ']'
                                data = json.loads(content)
                                print("修复了末尾逗号后成功加载")
                            else:
                                raise
                    elif content.startswith('{') and content.endswith('}'):
                        # 单个JSON对象
                        data_obj = json.loads(content)
                        if 'business_id' in data_obj:
                            data = [data_obj]
                            print("加载了单个咖啡店对象")
                        else:
                            # 可能是个字典，值是咖啡店对象
                            data = []
                            for key, value in data_obj.items():
                                if isinstance(value, dict) and 'business_id' in value:
                                    data.append(value)
                            print(f"从字典中提取了 {len(data)} 个咖啡店对象")
                    else:
                        # 尝试加载JSONL
                        data = []
                        for line in content.split('\n'):
                            if line.strip():
                                try:
                                    obj = json.loads(line)
                                    if isinstance(obj, dict) and 'business_id' in obj:
                                        data.append(obj)
                                except json.JSONDecodeError:
                                    print(f"警告: 跳过无效的JSON行: {line[:50]}...")
                        print(f"从JSONL格式中加载了 {len(data)} 个对象")
            except UnicodeDecodeError:
                # 尝试其他编码
                print("UTF-8编码失败，尝试latin1编码")
                with open(json_file_path, 'r', encoding='latin1') as f:
                    content = f.read().strip()
                    # 类似的处理逻辑...
                    if content.startswith('[') and content.endswith(']'):
                        data = json.loads(content)
                    else:
                        # 其他格式处理...
                        data = []
        else:
            raise ValueError(f"不支持的文件格式: {json_file_path}")
            
        # 打印数据样本以进行验证
        if data and len(data) > 0:
            print(f"数据样本 (第一个对象的键): {list(data[0].keys())[:5]}...")
            print(f"总共发现 {len(data)} 个咖啡店对象")
        else:
            print("警告: 未能从文件中提取有效数据")
            return pd.DataFrame()
        
        print(f"成功从{json_file_path}加载了 {len(data)} 条记录")
        
        # 统一数据结构，展平嵌套字典
        flat_data = []
        
        # 收集所有可能的字段
        all_about_fields = {}
        all_working_hours = set()
        
        # 第一次遍历，收集所有可能的字段
        for cafe in data:
            # 收集工作时间字段
            if 'working_hours' in cafe and cafe['working_hours'] is not None:
                for day in cafe['working_hours'].keys():
                    all_working_hours.add(day)
            
            # 收集about字段
            if 'about' in cafe and cafe['about'] is not None and 'details' in cafe['about'] and cafe['about']['details'] is not None:
                for category, fields in cafe['about']['details'].items():
                    if category not in all_about_fields:
                        all_about_fields[category] = set()
                    if fields is not None:  # 确保fields不是None
                        for field in fields.keys():
                            all_about_fields[category].add(field)
        
        # 排序天数以确保一周的顺序
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        all_working_hours = sorted(all_working_hours, key=lambda x: day_order.index(x) if x in day_order else 999)
        
        # 第二次遍历，展平数据
        for cafe in data:
            flat_cafe = {}
            
            # 复制基本字段（排除不需要的字段）
            excluded_fields = ['opening_status', 'photos_sample', 'state', 'country', 'city', 'street_address', 'address', 'about', 'working_hours', 'topics', 'sentiment_scores', 'sentiment_percentages']
            for key, value in cafe.items():
                if key not in excluded_fields:
                    # 处理字符串中的引号，以防止snowflake导入问题
                    if isinstance(value, str):
                        value = value.replace('"', '')
                    flat_cafe[key] = value
            
            # 处理工作时间
            if 'working_hours' in cafe and cafe['working_hours'] is not None:
                for day in all_working_hours:
                    column_name = f"working_hours_{day}"
                    if day in cafe['working_hours'] and cafe['working_hours'][day]:
                        # 多个时间段合并为一个字符串
                        flat_cafe[column_name] = '; '.join(cafe['working_hours'][day]).replace('"', '')
                    else:
                        flat_cafe[column_name] = None
            else:
                for day in all_working_hours:
                    flat_cafe[f"working_hours_{day}"] = None
            
            # 检查topics结构并展平
            if 'topics' in cafe and cafe['topics'] is not None:
                if isinstance(cafe['topics'], list):
                    for i, topic in enumerate(cafe['topics']):
                        if i >= 5:  # 限制最多处理5个主题
                            break
                        topic_idx = i + 1
                        if isinstance(topic, dict):
                            for key, value in topic.items():
                                col_name = f"topics_{topic_idx}_{key}".replace(' ', '_')
                                if isinstance(value, str):
                                    flat_cafe[col_name] = value.replace('"', '')
                                else:
                                    flat_cafe[col_name] = value
                else:
                    # 如果topics不是列表，设置空值
                    print(f"警告: cafe_id为{cafe.get('business_id')}的topics不是列表")
                    
            # 处理sentiment_scores和sentiment_percentages
            for sentiment_field in ['sentiment_scores', 'sentiment_percentages']:
                if sentiment_field in cafe and cafe[sentiment_field] is not None:
                    for key, value in cafe[sentiment_field].items():
                        col_name = f"{sentiment_field}_{key}".replace(' ', '_')
                        flat_cafe[col_name] = value
            
            flat_data.append(flat_cafe)
        
        # 创建DataFrame
        df = pd.DataFrame(flat_data)
        
        # 打印列和数据类型以便调试
        print(f"总共有 {len(df.columns)} 列:")
        for col in df.columns:
            print(f"{col}: {df[col].dtype}")
            
        # 检查是否有NaN值并正确处理
        for col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                print(f"列 '{col}' 包含 {null_count} 个空值")
                
        return df
    except Exception as e:
        print(f"读取JSON文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def create_snowflake_table(conn, table_name: str, columns: Dict[str, str]):
    """
    在Snowflake中创建表
    
    Args:
        conn: Snowflake连接对象
        table_name: 表名
        columns: 列名到数据类型的映射
    """
    cursor = conn.cursor()
    try:
        # 首先检查表是否存在
        cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            print(f"表 {table_name} 已存在，将被删除并重新创建")
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        column_defs = ", ".join([f'"{name}" {dtype}' for name, dtype in columns.items()])
        query = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({column_defs})'
        
        print(f"创建表 {table_name}")
        print(f"SQL: {query}")
        cursor.execute(query)
    except Exception as e:
        print(f"创建表时出错: {str(e)}")
        print(f"查询: {query}")
        raise
    finally:
        cursor.close()

def is_nan_value(val):
    """
    安全地检查值是否为NaN，适用于各种数据类型
    """
    if val is None:
        return True
    if isinstance(val, float) and pd.isna(val):
        return True
    if isinstance(val, str) and (val.upper() == "NAN" or val.lower() == "nan"):
        return True
    return False

def convert_complex_type(val):
    """
    将复杂数据类型转换为JSON字符串
    """
    if val is None:
        return None
    if isinstance(val, (list, dict)):
        return json.dumps(val)
    if isinstance(val, np.ndarray):
        return json.dumps(val.tolist())
    return val

def insert_dataframe_to_snowflake(conn, table_name: str, df: pd.DataFrame):
    """
    将DataFrame插入到Snowflake表中
    
    Args:
        conn: Snowflake连接对象
        table_name: 表名
        df: 要插入的DataFrame
    """
    # 如果DataFrame为空，直接返回
    if df.empty:
        print(f"提供的DataFrame为空，跳过插入表 {table_name}")
        return
    
    # 处理复杂数据类型和NaN值
    print("处理DataFrame以准备插入Snowflake")
    df_clean = df.copy()
    
    # 将所有列转换为适合Snowflake的格式
    for col in df_clean.columns:
        print(f"处理列: {col}")
        # 创建新的值列表
        new_values = []
        for i in range(len(df_clean)):
            val = df_clean.loc[i, col]
            
            # 处理NaN值
            if isinstance(val, (float, int)) and pd.isna(val):
                new_values.append(None)
                continue
                
            # 处理字符串的"NAN"或"nan"
            if isinstance(val, str) and (val.upper() == "NAN" or val.lower() == "nan"):
                new_values.append(None)
                continue
                
            # 处理复杂数据类型
            if isinstance(val, (dict, list, np.ndarray)):
                try:
                    new_values.append(json.dumps(val))
                except Exception as e:
                    print(f"警告: 无法序列化列 '{col}' 中的值，错误: {str(e)}")
                    new_values.append(None)
                continue
                
            # 其他值保持不变
            new_values.append(val)
            
        # 更新列值
        df_clean[col] = new_values
    
    # 再次检查是否有NaN值
    for col in df_clean.columns:
        nan_count = df_clean[col].isna().sum()
        if nan_count > 0:
            print(f"列 '{col}' 中有 {nan_count} 个NaN值，将转换为NULL")
            df_clean.loc[df_clean[col].isna(), col] = None
    
    # 手动创建元组列表，确保所有值都是Snowflake可接受的类型
    values = []
    for i in range(len(df_clean)):
        row_values = []
        for col in df_clean.columns:
            val = df_clean.loc[i, col]
            # 最后检查一次是否为NaN
            if pd.isna(val):
                row_values.append(None)
            else:
                row_values.append(val)
        values.append(tuple(row_values))
    
    # 生成列名，带引号以确保Snowflake能识别
    column_names = ", ".join([f'"{col}"' for col in df_clean.columns])
    
    # 生成占位符，和列数一致
    placeholders = ", ".join(["(%s)" % ", ".join(["%s"] * len(df_clean.columns))])
    
    # 生成INSERT语句
    query = f'INSERT INTO "{table_name}" ({column_names}) VALUES {placeholders}'
    
    print(f"向表 {table_name} 插入 {len(df_clean)} 条记录")
    cursor = conn.cursor()
    try:
        cursor.executemany(query, values)
    except Exception as e:
        print(f"插入数据时出错: {str(e)}")
        print(f"查询: {query[0:500]}...")  # 只打印前500个字符以避免过长
        print(f"数据框列: {df_clean.columns.tolist()}")
        
        # 检查所有值类型
        print(f"检查第一行数据，确保没有NaN或其他问题值:")
        
        if values and len(values) > 0:
            for i, val in enumerate(values[0]):
                col_name = df_clean.columns[i] if i < len(df_clean.columns) else f"Column_{i}"
                print(f"  {col_name}: {type(val).__name__ if val is not None else 'NoneType'}, 值: {val}")
        raise
    finally:
        cursor.close()

def clear_table(conn, table_name: str):
    """
    清空表中的所有数据
    
    Args:
        conn: Snowflake连接对象
        table_name: 表名
    """
    cursor = conn.cursor()
    try:
        query = f'DELETE FROM "{table_name}"'
        print(f"清空表 {table_name}")
        cursor.execute(query)
    except Exception as e:
        print(f"清空表时出错: {str(e)}")
        raise
    finally:
        cursor.close()

def sanitize_column_name(name):
    """
    清理列名，使其符合Snowflake要求
    
    Args:
        name: 原始列名
        
    Returns:
        清理后的列名
    """
    # 替换不合法字符
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    
    # 确保不以数字开头
    if name[0].isdigit():
        name = 'C_' + name
        
    return name

def sync_from_json_to_snowflake(conn, json_file_path: str, table_name: str = "COFFEE_SHOPS"):
    """
    将JSON数据同步到Snowflake表
    
    Args:
        conn: Snowflake连接对象
        json_file_path: JSON文件的路径
        table_name: 目标表名
    """
    # 从JSON加载DataFrame
    df = json_to_dataframe(json_file_path)
    
    # 确保列名符合Snowflake要求
    df.columns = [sanitize_column_name(col) for col in df.columns]
    
    # 列名转换为大写
    df.columns = [col.upper() for col in df.columns]
    
    # 生成schema字典
    columns = {}
    for col in df.columns:
        # 基本列处理
        if col == 'BUSINESS_ID':
            columns[col] = 'VARCHAR(255) PRIMARY KEY'
        elif 'ID' in col:
            columns[col] = 'VARCHAR(255)'
        elif col.startswith('ABOUT_'):
            if df[col].dtype == bool or df[col].dtype == np.bool_:
                columns[col] = 'BOOLEAN'
            else:
                columns[col] = 'VARCHAR(1000)'
        elif col.startswith('WORKING_HOURS_'):
            columns[col] = 'VARCHAR(255)'
        else:
            # 根据数据类型动态分配数据类型
            dtype = df[col].dtype
            
            if pd.api.types.is_object_dtype(dtype):
                # 检查是否所有值都是布尔型
                if all(isinstance(x, bool) or x is None for x in df[col].dropna()):
                    columns[col] = 'BOOLEAN'
                else:
                    # 对于字符串列，检查最大长度并分配适当空间
                    max_len = df[col].astype(str).str.len().max()
                    if max_len <= 255:
                        columns[col] = 'VARCHAR(255)'
                    elif max_len <= 1000:
                        columns[col] = 'VARCHAR(1000)'
                    else:
                        columns[col] = 'TEXT'
            elif pd.api.types.is_bool_dtype(dtype):
                columns[col] = 'BOOLEAN'
            elif pd.api.types.is_float_dtype(dtype):
                columns[col] = 'FLOAT'
            elif pd.api.types.is_integer_dtype(dtype):
                columns[col] = 'INTEGER'
            else:
                # 默认为VARCHAR
                columns[col] = 'VARCHAR(255)'
    
    # 打印列映射以便调试
    print("列映射到Snowflake数据类型:")
    for col, dtype in columns.items():
        print(f"  {col}: {dtype}")
    
    # 创建表
    create_snowflake_table(conn, table_name, columns)
    
    # 清空表，然后插入所有记录
    clear_table(conn, table_name)
    insert_dataframe_to_snowflake(conn, table_name, df)
    
    print(f"成功同步 {len(df)} 条记录到表 {table_name}")
    return len(df)

def run_snowflake_sync_from_json(config_path: str = None, json_file_path: str = None):
    """
    运行从JSON到Snowflake的同步过程
    
    Args:
        config_path: Snowflake配置文件路径（如果使用环境变量则可选）
        json_file_path: JSON文件的路径
    """
    # 设置默认路径
    if json_file_path is None:
        json_file_path = "data/merge/integrated_cafes.json"
    
    # 加载Snowflake配置
    config = load_config(config_path)
    
    # 连接到Snowflake
    conn = None
    try:
        print(f"连接到Snowflake...")
        conn = snowflake.connector.connect(
            account=config['account'],
            user=config['user'],
            password=config['password'],
            warehouse=config['warehouse'],
            database=config['database'],
            schema=config['schema']
        )
        
        # 从JSON同步到Snowflake表
        records_count = sync_from_json_to_snowflake(conn, json_file_path)
        
        print(f"Snowflake 同步完成，成功同步 {records_count} 条记录")
        return True
        
    except Exception as e:
        print(f"同步过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 断开Snowflake连接
        if conn:
            conn.close()
            print("Snowflake连接已关闭")

if __name__ == "__main__":
    # 默认情况下从环境变量加载配置
    # 可以通过参数指定配置文件
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # 默认JSON文件路径
    json_file_path = "data/merge/integrated_cafes.json"
    if len(sys.argv) > 2:
        json_file_path = sys.argv[2]
    
    success = run_snowflake_sync_from_json(config_path, json_file_path)
    sys.exit(0 if success else 1)