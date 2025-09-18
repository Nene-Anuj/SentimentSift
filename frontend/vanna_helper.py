from vanna.chromadb import ChromaDB_VectorStore
from vanna.google import GoogleGeminiChat
from snowflake.snowpark import Session
import os
from dotenv import load_dotenv
import pandas as pd

# 加載環境變量
load_dotenv()

class CoffeeShopVanna(ChromaDB_VectorStore, GoogleGeminiChat):
    def __init__(self):
        # 初始化基類
        ChromaDB_VectorStore.__init__(self)
        GoogleGeminiChat.__init__(self, config={
            'api_key': os.getenv("GEMINI_API_KEY"),
            'model': os.getenv("GEMINI_MODEL", "gemini-pro")
        })
        
        # 連接Snowflake
        self.connect_to_snowflake()
        
        # 初始化訓練數據
        self.initialize_training()
    
    def connect_to_snowflake(self):
        """連接到Snowflake數據庫"""
        try:
            # 方法1: 使用Vanna提供的方法
            self.connect_to_snowflake(
                account=os.getenv("SNOWFLAKE_ACCOUNT"),
                username=os.getenv("SNOWFLAKE_USER"),
                password=os.getenv("SNOWFLAKE_PASSWORD"),
                database=os.getenv("SNOWFLAKE_DATABASE"),
                schema=os.getenv("SNOWFLAKE_SCHEMA"),
                warehouse=os.getenv("SNOWFLAKE_WAREHOUSE")
            )
            
            # 方法2: 使用Snowflake Snowpark
            # connection_parameters = {
            #     "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            #     "user": os.getenv("SNOWFLAKE_USER"),
            #     "password": os.getenv("SNOWFLAKE_PASSWORD"),
            #     "database": os.getenv("SNOWFLAKE_DATABASE"),
            #     "schema": os.getenv("SNOWFLAKE_SCHEMA"),
            #     "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE")
            # }
            # self.snowflake_session = Session.builder.configs(connection_parameters).create()
            
            print("成功連接到Snowflake")
            return True
        except Exception as e:
            print(f"連接Snowflake時出錯: {e}")
            return False
    
    def initialize_training(self):
        """初始化訓練數據"""
        try:
            # 1. 添加數據庫表結構信息
            self.add_documentation('''
            The FINAL_DB.FINAL.COFFEE_SHOPS table contains data about coffee shops with the following columns:
            - BUSINESS_ID: Unique identifier for each coffee shop
            - NAME: Name of the coffee shop
            - FULL_ADDRESS: Complete address of the coffee shop
            - LATITUDE: Latitude coordinate of the coffee shop
            - LONGITUDE: Longitude coordinate of the coffee shop
            - RATING: Overall rating of the coffee shop (1-5 scale)
            - REVIEW_COUNT: Number of reviews received
            - PHONE_NUMBER: Contact phone number
            - PRICE_LEVEL: Price category ($, $$, $$$, or $$$$)
            - TYPE: Type of establishment (e.g., 'Coffee Shop', 'Cafe', etc.)
            - SENTIMENT_SERVICE: Score for service quality (0-1 scale)
            - SENTIMENT_SERVICE_TIER: Categorization of service quality (Poor, Average, Good, Excellent)
            - SENTIMENT_FOOD: Score for food quality (0-1 scale)
            - SENTIMENT_FOOD_TIER: Categorization of food quality (Poor, Average, Good, Excellent)
            - SENTIMENT_AMBIANCE: Score for ambiance quality (0-1 scale)
            - SENTIMENT_AMBIANCE_TIER: Categorization of ambiance quality (Poor, Average, Good, Excellent)
            - TOPIC_ID: Identifier for topic analysis
            - TOPIC_COUNT: Number of mentions for this topic
            - TOPIC_KEYWORDS: Keywords associated with this topic
            - TOPIC_NAME: Name of the topic identified in reviews
            - TOPIC_RANK: Ranking of the topic importance
            - WEBSITE: Website URL of the coffee shop
            ''')
            
            # 2. 自動從數據庫獲取表結構信息
            # df_information_schema = self.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'COFFEE_SHOPS'")
            # plan = self.get_training_plan_generic(df_information_schema)
            # self.train(plan=plan)
            
            # 3. 添加示例問題和對應的SQL查詢
            self.add_question_sql(
                question="What are the coffee shops with the highest food ratings?",
                sql="""
                SELECT NAME, FULL_ADDRESS, RATING, SENTIMENT_FOOD, SENTIMENT_FOOD_TIER, LATITUDE, LONGITUDE, TOPIC_KEYWORDS, TOPIC_NAME, TOPIC_COUNT
                FROM FINAL_DB.FINAL.COFFEE_SHOPS
                ORDER BY SENTIMENT_FOOD DESC
                LIMIT 10;
                """
            )
            
            self.add_question_sql(
                question="Which coffee shops have the worst service?",
                sql="""
                SELECT NAME, FULL_ADDRESS, RATING, SENTIMENT_SERVICE, SENTIMENT_SERVICE_TIER, LATITUDE, LONGITUDE, TOPIC_KEYWORDS, TOPIC_NAME, TOPIC_COUNT
                FROM FINAL_DB.FINAL.COFFEE_SHOPS
                ORDER BY SENTIMENT_SERVICE ASC
                LIMIT 10;
                """
            )
            
            self.add_question_sql(
                question="Show me coffee shops with the best ambiance",
                sql="""
                SELECT NAME, FULL_ADDRESS, RATING, SENTIMENT_AMBIANCE, SENTIMENT_AMBIANCE_TIER, LATITUDE, LONGITUDE, TOPIC_KEYWORDS, TOPIC_NAME, TOPIC_COUNT
                FROM FINAL_DB.FINAL.COFFEE_SHOPS
                ORDER BY SENTIMENT_AMBIANCE DESC
                LIMIT 10;
                """
            )
            
            self.add_question_sql(
                question="What are the most popular coffee shops based on review count?",
                sql="""
                SELECT NAME, FULL_ADDRESS, RATING, REVIEW_COUNT, PRICE_LEVEL, LATITUDE, LONGITUDE, TOPIC_KEYWORDS, TOPIC_NAME, TOPIC_COUNT
                FROM FINAL_DB.FINAL.COFFEE_SHOPS
                ORDER BY REVIEW_COUNT DESC
                LIMIT 15;
                """
            )
            
            # 4. 添加中文示例問題
            self.add_question_sql(
                question="哪些咖啡店的食物評分最高？",
                sql="""
                SELECT NAME, FULL_ADDRESS, RATING, SENTIMENT_FOOD, SENTIMENT_FOOD_TIER, LATITUDE, LONGITUDE, TOPIC_KEYWORDS, TOPIC_NAME, TOPIC_COUNT
                FROM FINAL_DB.FINAL.COFFEE_SHOPS
                ORDER BY SENTIMENT_FOOD DESC
                LIMIT 10;
                """
            )
            
            self.add_question_sql(
                question="波士頓哪些咖啡店服務最差？",
                sql="""
                SELECT NAME, FULL_ADDRESS, RATING, SENTIMENT_SERVICE, SENTIMENT_SERVICE_TIER, LATITUDE, LONGITUDE, TOPIC_KEYWORDS, TOPIC_NAME, TOPIC_COUNT
                FROM FINAL_DB.FINAL.COFFEE_SHOPS
                WHERE FULL_ADDRESS LIKE '%Boston%'
                ORDER BY SENTIMENT_SERVICE ASC
                LIMIT 10;
                """
            )
            
            print("訓練數據初始化完成")
        except Exception as e:
            print(f"初始化訓練數據時出錯: {e}")