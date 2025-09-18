from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
from sqlalchemy import text
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# 導入已有的 SQL 和 ChatBot 代理
from sql_agent import SQLAgent
from chatbot_agent import ChatbotAgent

app = FastAPI(title="SentimentSift Restaurant Analysis API")

# 初始化 SQL 代理
sql_agent = SQLAgent("snowflake_config.json")

# 初始化聊天機器人代理
chatbot_agent = ChatbotAgent()

# 定義請求模型
class QueryRequest(BaseModel):
    question: str

class ChartDataRequest(BaseModel):
    chart_type: str
    city: Optional[str] = None
    limit: Optional[int] = 50
    min_score: Optional[float] = None
    category: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Welcome to the SentimentSift API. Visit /docs for API documentation."}

@app.post("/query")
async def natural_language_query(request: QueryRequest):
    """
    將自然語言問題轉換為 SQL 查詢並返回結果
    """
    try:
        # 使用 SQL 代理執行自然語言查詢
        query_result = sql_agent.natural_language_query(request.question)
        
        # 確保結果包含經緯度資訊用於地圖
        if query_result["results"] and len(query_result["results"]) > 0:
            for restaurant in query_result["results"]:
                # 如果資料庫中沒有經緯度，可以使用一個模擬的方法生成
                if "LATITUDE" not in restaurant or "LONGITUDE" not in restaurant:
                    # 這裡只是演示，實際應用中應該從資料庫獲取真實資料
                    if "FULL_ADDRESS" in restaurant and "Boston" in restaurant["FULL_ADDRESS"]:
                        base_lat, base_lng = 42.3601, -71.0589
                    elif "FULL_ADDRESS" in restaurant and "New York" in restaurant["FULL_ADDRESS"]:
                        base_lat, base_lng = 40.7128, -74.0060
                    else:
                        base_lat, base_lng = 37.7749, -122.4194
                    
                    # 添加一些小的隨機偏移以區分餐廳位置
                    restaurant["LATITUDE"] = base_lat + (np.random.random() - 0.5) * 0.05
                    restaurant["LONGITUDE"] = base_lng + (np.random.random() - 0.5) * 0.05
        
        return query_result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查詢處理錯誤: {str(e)}")

@app.get("/search")
async def search_restaurants(
    q: str,
    city: Optional[str] = None,
    min_score: Optional[float] = None,
    limit: int = 10,
    include_reviews: bool = False
):
    """
    基於參數搜索餐廳
    """
    try:
        # 構建 SQL 查詢
        query = """
        SELECT 
            c.BUSINESS_ID, 
            c.NAME, 
            c.FULL_ADDRESS as address, 
            c.LATITUDE,
            c.LONGITUDE,
            c.RATING as stars, 
            c.SENTIMENT_FOOD as food_score, 
            c.SENTIMENT_SERVICE as service_score, 
            c.SENTIMENT_AMBIANCE as ambiance_score,
            (c.SENTIMENT_FOOD + c.SENTIMENT_SERVICE + c.SENTIMENT_AMBIANCE)/3 as overall_score,
            c.PRICE_LEVEL as price_range,
            c.TYPE as categories
        FROM 
            COFFEE_SHOPS c
        WHERE 
            1=1
        """
        
        # 添加過濾條件
        params = {}
        if city:
            query += " AND c.FULL_ADDRESS LIKE :city"
            params["city"] = f"%{city}%"
        
        if min_score:
            query += " AND c.RATING >= :min_score"
            params["min_score"] = min_score
        
        # 添加關鍵字搜索
        if q and q.strip():
            query += """ 
            AND (
                c.NAME ILIKE :search_term 
                OR c.TYPE ILIKE :search_term
                OR c.TOPIC_KEYWORDS ILIKE :search_term
            )
            """
            params["search_term"] = f"%{q}%"
        
        # 添加排序和限制
        query += " ORDER BY (c.SENTIMENT_FOOD + c.SENTIMENT_SERVICE + c.SENTIMENT_AMBIANCE)/3 DESC LIMIT :limit"
        params["limit"] = limit
        
        # 執行查詢
        results, _ = sql_agent.execute_query(query, params)
        
        # 提取城市信息從地址
        for restaurant in results:
            if "address" in restaurant:
                # 嘗試從地址中提取城市
                address_parts = restaurant["address"].split(',')
                if len(address_parts) > 1:
                    restaurant["city"] = address_parts[-2].strip()
                else:
                    restaurant["city"] = "Unknown"
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索錯誤: {str(e)}")

@app.get("/restaurants")
async def get_top_restaurants(
    limit: int = 10,
    city: Optional[str] = None,
    min_score: Optional[float] = None,
    category: Optional[str] = None
):
    """
    獲取頂級餐廳列表
    """
    try:
        # 構建 SQL 查詢
        query = """
        SELECT 
            c.BUSINESS_ID, 
            c.NAME, 
            c.FULL_ADDRESS as address, 
            c.LATITUDE,
            c.LONGITUDE,
            c.RATING as stars, 
            c.SENTIMENT_FOOD as food_score, 
            c.SENTIMENT_SERVICE as service_score, 
            c.SENTIMENT_AMBIANCE as ambiance_score,
            (c.SENTIMENT_FOOD + c.SENTIMENT_SERVICE + c.SENTIMENT_AMBIANCE)/3 as overall_score,
            c.PRICE_LEVEL as price_range,
            c.TYPE as categories
        FROM 
            COFFEE_SHOPS c
        WHERE 
            1=1
        """
        
        # 添加過濾條件
        params = {}
        if city:
            query += " AND c.FULL_ADDRESS LIKE :city"
            params["city"] = f"%{city}%"
        
        if min_score:
            query += " AND c.RATING >= :min_score"
            params["min_score"] = min_score
        
        if category:
            query += " AND c.TYPE ILIKE :category"
            params["category"] = f"%{category}%"
        
        # 添加排序和限制
        query += " ORDER BY (c.SENTIMENT_FOOD + c.SENTIMENT_SERVICE + c.SENTIMENT_AMBIANCE)/3 DESC LIMIT :limit"
        params["limit"] = limit
        
        # 執行查詢
        results, _ = sql_agent.execute_query(query, params)
        
        # 提取城市信息從地址
        for restaurant in results:
            if "address" in restaurant:
                # 嘗試從地址中提取城市
                address_parts = restaurant["address"].split(',')
                if len(address_parts) > 1:
                    restaurant["city"] = address_parts[-2].strip()
                else:
                    restaurant["city"] = "Unknown"
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取餐廳錯誤: {str(e)}")

@app.get("/stats/overall")
async def get_overall_stats(city: Optional[str] = None):
    """
    獲取整體統計資料，用於儀表板
    """
    try:
        # 構建 SQL 查詢
        query = """
        SELECT 
            COUNT(*) as total_restaurants,
            AVG(RATING) as avg_rating,
            AVG(SENTIMENT_FOOD) as avg_food_score,
            AVG(SENTIMENT_SERVICE) as avg_service_score,
            AVG(SENTIMENT_AMBIANCE) as avg_ambiance_score
        FROM 
            COFFEE_SHOPS
        """
        
        params = {}
        if city:
            query += " WHERE FULL_ADDRESS LIKE :city"
            params["city"] = f"%{city}%"
        
        # 執行查詢
        results, _ = sql_agent.execute_query(query, params)
        
        if not results or len(results) == 0:
            return {
                "total_restaurants": 0,
                "avg_rating": 0,
                "avg_food_score": 0,
                "avg_service_score": 0,
                "avg_ambiance_score": 0
            }
        
        # 城市分布（從地址中提取）
        city_query = """
        SELECT 
            REGEXP_SUBSTR(FULL_ADDRESS, '[^,]+', 1, 2) as city, 
            COUNT(*) as count
        FROM 
            COFFEE_SHOPS
        GROUP BY 
            city
        ORDER BY 
            count DESC
        LIMIT 10
        """
        
        city_results, _ = sql_agent.execute_query(city_query, {})
        
        # 分類分布
        category_query = """
        SELECT 
            TYPE as category, 
            COUNT(*) as count
        FROM 
            COFFEE_SHOPS
        GROUP BY 
            TYPE
        ORDER BY 
            count DESC
        LIMIT 10
        """
        
        category_results, _ = sql_agent.execute_query(category_query, {})
        
        # 組合結果
        return {
            "summary": results[0],
            "cities": city_results,
            "categories": category_results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取統計錯誤: {str(e)}")

@app.get("/stats/ratings")
async def get_rating_distribution(city: Optional[str] = None):
    """
    獲取評分分布，用於直方圖
    """
    try:
        # 構建 SQL 查詢
        query = """
        SELECT 
            FLOOR(RATING * 2) / 2 as rating_bin,
            COUNT(*) as count
        FROM 
            COFFEE_SHOPS
        """
        
        params = {}
        if city:
            query += " WHERE FULL_ADDRESS LIKE :city"
            params["city"] = f"%{city}%"
        
        query += """
        GROUP BY 
            rating_bin
        ORDER BY 
            rating_bin
        """
        
        # 執行查詢
        results, _ = sql_agent.execute_query(query, params)
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取評分分布錯誤: {str(e)}")

@app.get("/stats/scores")
async def get_score_comparison(city: Optional[str] = None):
    """
    獲取不同評分類別的比較資料，用於比較圖表
    """
    try:
        # 構建 SQL 查詢
        query = """
        SELECT 
            AVG(SENTIMENT_FOOD) as avg_food_score,
            AVG(SENTIMENT_SERVICE) as avg_service_score,
            AVG(SENTIMENT_AMBIANCE) as avg_ambiance_score,
            STDDEV(SENTIMENT_FOOD) as std_food_score,
            STDDEV(SENTIMENT_SERVICE) as std_service_score,
            STDDEV(SENTIMENT_AMBIANCE) as std_ambiance_score,
            MIN(SENTIMENT_FOOD) as min_food_score,
            MIN(SENTIMENT_SERVICE) as min_service_score,
            MIN(SENTIMENT_AMBIANCE) as min_ambiance_score,
            MAX(SENTIMENT_FOOD) as max_food_score,
            MAX(SENTIMENT_SERVICE) as max_service_score,
            MAX(SENTIMENT_AMBIANCE) as max_ambiance_score
        FROM 
            COFFEE_SHOPS
        """
        
        params = {}
        if city:
            query += " WHERE FULL_ADDRESS LIKE :city"
            params["city"] = f"%{city}%"
        
        # 執行查詢
        results, _ = sql_agent.execute_query(query, params)
        
        if not results or len(results) == 0:
            return {
                "avg_scores": {
                    "Food": 0,
                    "Service": 0,
                    "Ambiance": 0
                },
                "std_scores": {
                    "Food": 0,
                    "Service": 0,
                    "Ambiance": 0
                },
                "min_scores": {
                    "Food": 0,
                    "Service": 0,
                    "Ambiance": 0
                },
                "max_scores": {
                    "Food": 0,
                    "Service": 0,
                    "Ambiance": 0
                }
            }
        
        # 格式化結果
        result = results[0]
        return {
            "avg_scores": {
                "Food": result["avg_food_score"],
                "Service": result["avg_service_score"],
                "Ambiance": result["avg_ambiance_score"]
            },
            "std_scores": {
                "Food": result["std_food_score"],
                "Service": result["std_service_score"],
                "Ambiance": result["std_ambiance_score"]
            },
            "min_scores": {
                "Food": result["min_food_score"],
                "Service": result["min_service_score"],
                "Ambiance": result["min_ambiance_score"]
            },
            "max_scores": {
                "Food": result["max_food_score"],
                "Service": result["max_service_score"],
                "Ambiance": result["max_ambiance_score"]
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取評分比較錯誤: {str(e)}")

@app.get("/stats/top_by_category")
async def get_top_by_category(
    category: str,
    city: Optional[str] = None,
    limit: int = 10
):
    """
    獲取特定類別的頂級餐廳
    """
    try:
        # 構建 SQL 查詢
        query = """
        SELECT 
            c.BUSINESS_ID, 
            c.NAME, 
            c.FULL_ADDRESS as address, 
            c.RATING as stars,
            c.SENTIMENT_FOOD as food_score, 
            c.SENTIMENT_SERVICE as service_score, 
            c.SENTIMENT_AMBIANCE as ambiance_score,
            (c.SENTIMENT_FOOD + c.SENTIMENT_SERVICE + c.SENTIMENT_AMBIANCE)/3 as overall_score
        FROM 
            COFFEE_SHOPS c
        WHERE 
            c.TYPE ILIKE :category
        """
        
        params = {"category": f"%{category}%"}
        
        if city:
            query += " AND c.FULL_ADDRESS LIKE :city"
            params["city"] = f"%{city}%"
        
        # 依評分排序
        if category.lower() in ["food", "restaurant", "cafe", "bakery"]:
            query += " ORDER BY c.SENTIMENT_FOOD DESC"
        elif category.lower() in ["service", "hospitality"]:
            query += " ORDER BY c.SENTIMENT_SERVICE DESC"
        elif category.lower() in ["ambiance", "atmosphere", "environment"]:
            query += " ORDER BY c.SENTIMENT_AMBIANCE DESC"
        else:
            query += " ORDER BY (c.SENTIMENT_FOOD + c.SENTIMENT_SERVICE + c.SENTIMENT_AMBIANCE)/3 DESC"
        
        query += " LIMIT :limit"
        params["limit"] = limit
        
        # 執行查詢
        results, _ = sql_agent.execute_query(query, params)
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取頂級類別錯誤: {str(e)}")
    
@app.post("/chat")
async def chat_with_bot(request: QueryRequest):
    """
    與聊天機器人進行對話
    """
    try:
        # 使用聊天機器人代理處理問題
        response = await chatbot_agent.process_restaurant_query(request.question)
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理聊天請求時出錯: {str(e)}")
    
@app.get("/stats/city_comparison")
async def get_city_comparison():
    """
    獲取不同城市的餐廳評分比較
    """
    try:
        # 構建 SQL 查詢
        query = """
        SELECT 
            REGEXP_SUBSTR(FULL_ADDRESS, '[^,]+', 1, 2) as city,
            COUNT(*) as restaurant_count,
            AVG(RATING) as avg_rating,
            AVG(SENTIMENT_FOOD) as avg_food_score,
            AVG(SENTIMENT_SERVICE) as avg_service_score,
            AVG(SENTIMENT_AMBIANCE) as avg_ambiance_score
        FROM 
            COFFEE_SHOPS
        GROUP BY 
            city
        ORDER BY 
            restaurant_count DESC
        LIMIT 10
        """
        
        # 執行查詢
        results, _ = sql_agent.execute_query(query, {})
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取城市比較錯誤: {str(e)}")

# 運行服務器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)