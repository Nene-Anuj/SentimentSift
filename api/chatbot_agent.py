import httpx
import json
from typing import List, Dict, Any, Optional
import re

class ChatbotAgent:
    """
    Agent responsible for handling restaurant-related queries through the chatbot interface.
    Integrates with the SQL API to fetch structured data.
    """
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        """
        Initialize the chatbot agent with API connection details.
        
        Args:
            api_base_url: Base URL for the restaurant API endpoints
        """
        self.api_base_url = api_base_url
        
    async def find_restaurants_by_location_and_rating(self, location: str, min_rating: float = 4.0) -> List[Dict[str, Any]]:
        """
        Query restaurants by location and minimum rating.
        
        Args:
            location: City name (e.g., "New York")
            min_rating: Minimum rating threshold (default: 4.0)
            
        Returns:
            List of restaurant data matching the criteria
        """
        params = {
            "q": "",
            "city": location,
            "min_score": min_rating,
            "limit": 10
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.api_base_url}/search", params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                # Handle error cases
                return []
    
    def parse_restaurant_query(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query to extract location and rating parameters.
        
        Args:
            query: Natural language query string
            
        Returns:
            Dictionary with extracted parameters
        """
        query = query.lower()
        params = {}
        
        # Extract location (city)
        location_patterns = [
            r"in\s+([a-zA-Z\s]+)",
            r"at\s+([a-zA-Z\s]+)",
            r"(?:restaurants|places)(?:\s+in|\s+at)\s+([a-zA-Z\s]+)"
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, query)
            if match:
                # Clean up extracted city name
                city = match.group(1).strip()
                # Remove trailing words that might not be part of the city
                city = re.sub(r'\s+with.*$|\s+that.*$|\s+and.*$', '', city)
                params["location"] = city
                break
        
        # Extract minimum rating
        rating_patterns = [
            r"(\d+\.?\d*)\s+stars?",
            r"rating\s+(?:of|above|over)\s+(\d+\.?\d*)",
            r"rated\s+(?:above|over)\s+(\d+\.?\d*)"
        ]
        
        for pattern in rating_patterns:
            match = re.search(pattern, query)
            if match:
                params["min_rating"] = float(match.group(1))
                break
        
        return params
    
    async def process_restaurant_query(self, query: str) -> str:
        """
        處理自然語言餐廳查詢並返回格式化結果
        """
        query_lower = query.lower()
        
        # 確定查詢類型
        # 1. 檢查是否是食物評分相關查詢
        is_food_query = any(keyword in query_lower for keyword in ["food", "delicious", "taste", "flavor", "cuisine", "dish", "meal"])
        # 2. 檢查是否是服務評分相關查詢
        is_service_query = any(keyword in query_lower for keyword in ["service", "staff", "waiter", "waitress", "hospitality"])
        # 3. 檢查是否是氛圍評分相關查詢
        is_ambiance_query = any(keyword in query_lower for keyword in ["ambiance", "atmosphere", "environment", "decor", "romantic", "cozy"])
        # 4. 檢查是否是最高或最低評分查詢
        is_highest_query = any(keyword in query_lower for keyword in ["best", "highest", "top", "great"])
        is_lowest_query = any(keyword in query_lower for keyword in ["worst", "lowest", "bad", "poor"])
        
        # 構建適當的API請求
        params = {}
        
        # 嘗試提取位置信息
        location_match = re.search(r"in\s+([a-zA-Z\s]+)(?:\s+with|\s+that|\s*\?|$)", query_lower)
        location = None
        if location_match:
            location = location_match.group(1).strip()
            params["city"] = location
        
        # 設置默認評分
        params["min_score"] = 4.0
        
        # 根據查詢類型設置排序字段
        if is_food_query:
            if is_lowest_query:
                endpoint = "/stats/top_by_category"
                params["category"] = "food"
                params["limit"] = 10
                sort_field = "food_score"
                ascending = True
            else:
                endpoint = "/stats/top_by_category"
                params["category"] = "food"
                params["limit"] = 10
                sort_field = "food_score"
                ascending = False
        elif is_service_query:
            if is_lowest_query:
                endpoint = "/stats/top_by_category"
                params["category"] = "service"
                params["limit"] = 10
                sort_field = "service_score"
                ascending = True
            else:
                endpoint = "/stats/top_by_category"
                params["category"] = "service"
                params["limit"] = 10
                sort_field = "service_score"
                ascending = False
        elif is_ambiance_query:
            if is_lowest_query:
                endpoint = "/stats/top_by_category"
                params["category"] = "ambiance"
                params["limit"] = 10
                sort_field = "ambiance_score"
                ascending = True
            else:
                endpoint = "/stats/top_by_category"
                params["category"] = "ambiance"
                params["limit"] = 10
                sort_field = "ambiance_score"
                ascending = False
        else:
            # 一般餐廳查詢
            endpoint = "/restaurants"
            if is_lowest_query:
                params["limit"] = 10
                sort_field = "stars"
                ascending = True
            else:
                params["limit"] = 10
                sort_field = "stars"
                ascending = False
        
        # 執行API請求
        async with httpx.AsyncClient() as client:
            try:
                if endpoint == "/stats/top_by_category":
                    # 特殊處理，因為top_by_category需要category參數在路徑中
                    category = params.pop("category", "restaurant")
                    response = await client.get(
                        f"{self.api_base_url}{endpoint}?category={category}",
                        params=params
                    )
                else:
                    response = await client.get(
                        f"{self.api_base_url}{endpoint}",
                        params=params
                    )
                
                if response.status_code == 200:
                    restaurants = response.json()
                    
                    # 根據排序字段和順序排序結果
                    if isinstance(restaurants, list) and restaurants:
                        if sort_field in restaurants[0]:
                            if ascending:
                                restaurants.sort(key=lambda x: x.get(sort_field, 0))
                            else:
                                restaurants.sort(key=lambda x: x.get(sort_field, 0), reverse=True)
                    
                    # 生成智能回應
                    if not restaurants or len(restaurants) == 0:
                        return f"對不起，我找不到符合您條件的餐廳。請嘗試不同的條件或城市。"
                    
                    # 根據查詢類型生成回應
                    response_text = self._generate_restaurant_response(
                        restaurants, 
                        is_food_query, 
                        is_service_query, 
                        is_ambiance_query,
                        is_highest_query,
                        is_lowest_query,
                        location
                    )
                    return response_text
                else:
                    return f"抱歉，在查詢餐廳時發生了錯誤。請再試一次或者換一個問法。"
            
            except Exception as e:
                return f"處理您的請求時發生錯誤: {str(e)}。請再試一次。"
        
    def _generate_restaurant_response(self, restaurants, is_food_query, is_service_query, 
                                    is_ambiance_query, is_highest_query, is_lowest_query, location):
        """生成針對不同查詢類型的智能回應"""
        
        # 確定回應的主題
        if is_food_query:
            aspect = "食物"
            score_field = "food_score"
        elif is_service_query:
            aspect = "服務"
            score_field = "service_score"
        elif is_ambiance_query:
            aspect = "氛圍"
            score_field = "ambiance_score"
        else:
            aspect = "整體"
            score_field = "stars"
        
        # 確定排名描述
        if is_lowest_query:
            ranking = "最低"
        else:
            ranking = "最高"
        
        # 地點描述
        location_text = f"在{location}" if location else ""
        
        # 生成回應標題
        title = f"以下是{location_text}{aspect}評分{ranking}的餐廳："
        
        # 生成餐廳列表
        restaurant_details = []
        for i, restaurant in enumerate(restaurants[:5], 1):  # 只顯示前5間
            name = restaurant.get("NAME", restaurant.get("name", "未知餐廳"))
            
            # 獲取評分，首先檢查新的字段名，然後檢查舊的字段名
            if score_field == "food_score":
                score = restaurant.get("food_score", restaurant.get("SENTIMENT_FOOD", "N/A"))
            elif score_field == "service_score":
                score = restaurant.get("service_score", restaurant.get("SENTIMENT_SERVICE", "N/A"))
            elif score_field == "ambiance_score":
                score = restaurant.get("ambiance_score", restaurant.get("SENTIMENT_AMBIANCE", "N/A"))
            else:
                score = restaurant.get("stars", restaurant.get("RATING", "N/A"))
            
            # 獲取地址
            address = restaurant.get("address", restaurant.get("FULL_ADDRESS", "地址未知"))
            
            # 獲取類型
            category = restaurant.get("categories", restaurant.get("TYPE", "類型未知"))
            
            # 獲取其他評分
            food_score = restaurant.get("food_score", restaurant.get("SENTIMENT_FOOD", "N/A"))
            service_score = restaurant.get("service_score", restaurant.get("SENTIMENT_SERVICE", "N/A"))
            ambiance_score = restaurant.get("ambiance_score", restaurant.get("SENTIMENT_AMBIANCE", "N/A"))
            
            # 生成描述文字
            description = f"{i}. {name} - {aspect}評分: {score}\n"
            description += f"   地址: {address}\n"
            description += f"   類型: {category}\n"
            description += f"   評分詳情: 食物({food_score}) | 服務({service_score}) | 氛圍({ambiance_score})\n"
            
            # 添加個性化推薦理由
            if is_food_query and is_highest_query:
                description += f"   推薦理由: 這家餐廳提供了令人驚嘆的美食體驗，食物質量非常出色。\n"
            elif is_food_query and is_lowest_query:
                description += f"   參考說明: 這家餐廳的食物評分較低，可能需要改進。\n"
            elif is_service_query and is_highest_query:
                description += f"   推薦理由: 這家餐廳的服務人員熱情且專業，提供卓越的顧客體驗。\n"
            elif is_ambiance_query and is_highest_query:
                description += f"   推薦理由: 這家餐廳擁有令人愉悅的環境和氛圍，非常適合享受用餐體驗。\n"
            
            restaurant_details.append(description)
        
        # 組合完整回應
        full_response = title + "\n\n" + "\n".join(restaurant_details)
        
        # 添加總結建議
        if is_highest_query:
            if is_food_query:
                full_response += "\n\n如果您特別在意食物品質，我強烈推薦第一個選項。它的食物評分最高，肯定能滿足您的味蕾。"
            elif is_service_query:
                full_response += "\n\n如果優質服務對您很重要，請考慮前兩個選項。它們的服務評分特別高，確保您有愉快的用餐體驗。"
            elif is_ambiance_query:
                full_response += "\n\n想要特別的氛圍體驗，請選擇第一個選項。它的環境評分最高，適合特殊場合。"
            else:
                full_response += "\n\n綜合各方面因素，我推薦您嘗試列表中的前三個選項，它們都能提供出色的整體體驗。"
        
        return full_response

def create_restaurant_tools(chatbot_agent):
    """
    Create tools for restaurant queries.
    
    Args:
        chatbot_agent: Instance of ChatbotAgent
        
    Returns:
        List of tool objects
    """
    return [{
        "name": "RestaurantFinder",
        "description": "Find restaurants by location and rating. Input should be a question about restaurants in a specific location."
    }]