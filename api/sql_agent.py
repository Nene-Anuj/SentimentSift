import snowflake.connector
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import json
import re
import os
from dotenv import load_dotenv

class SQLAgent:
    def __init__(self, snowflake_config_path: str):
        """
        Initialize SQL Agent with Snowflake configuration
        
        Args:
            snowflake_config_path: Path to Snowflake config file
        """
        # Load Snowflake configuration
        load_dotenv()
        
        # 嘗試從環境變量獲取配置
        self.account = os.getenv("SNOWFLAKE_ACCOUNT")
        self.user = os.getenv("SNOWFLAKE_USER")
        self.password = os.getenv("SNOWFLAKE_PASSWORD")
        self.warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
        self.database = os.getenv("SNOWFLAKE_DATABASE")
        self.schema = os.getenv("SNOWFLAKE_SCHEMA")
        
        self.conn = None
    
    def connect(self):
        """
        Connect to Snowflake
        """
        try:
            self.conn = snowflake.connector.connect(
                account=self.account,
                user=self.user,
                password=self.password,
                warehouse=self.warehouse,
                database=self.database,
                schema=self.schema
            )
            return True
        except Exception as e:
            print(f"Snowflake connection error: {str(e)}")
            return False
    
    def disconnect(self):
        """
        Disconnect from Snowflake
        """
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Execute a SQL query with parameters
        
        Args:
            query: SQL query to execute
            params: Dictionary of parameters for the query
            
        Returns:
            Tuple of (results as list of dictionaries, column names)
        """
        if not self.conn:
            self.connect()
        
        cursor = self.conn.cursor()
        
        try:
            # Execute query with parameters if provided
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Get column names
            column_names = [desc[0] for desc in cursor.description] if cursor.description else []
            
            # Fetch results
            results = cursor.fetchall()
            
            # Convert to list of dictionaries
            rows = []
            for row in results:
                row_dict = {}
                for i, col in enumerate(column_names):
                    row_dict[col] = row[i]
                rows.append(row_dict)
            
            return rows, column_names
        
        except Exception as e:
            print(f"Query execution error: {str(e)}")
            raise e
        
        finally:
            cursor.close()
    
    def natural_language_query(self, question: str) -> Dict[str, Any]:
        """
        Process a natural language query and convert to SQL
        
        Args:
            question: Natural language question
            
        Returns:
            Dictionary with query results and metadata
        """
        # Convert natural language to SQL
        sql_query = self.nl_to_sql(question)
        
        # Execute SQL query
        results, columns = self.execute_query(sql_query)
        
        return {
            'question': question,
            'sql_query': sql_query,
            'results': results,
            'columns': columns
        }
    
    def nl_to_sql(self, question: str) -> str:
        """
        Enhanced version: Convert natural language question to SQL query with improved detection
        
        Args:
            question: Natural language question from user
        
        Returns:
            SQL query corresponding to the natural language question
        """
        question = question.lower()
        
        # Extract city information using regex
        city_pattern = r"in\s+([a-zA-Z\s]+)(?:\s+with|\s+and|\s+that|\s*$|\s*\?)"
        city_match = re.search(city_pattern, question)
        city = None
        if city_match:
            # Extract city name and clean it
            city = city_match.group(1).strip()
            # Remove trailing words that might not be part of the city
            city = re.sub(r'\s+with.*$|\s+that.*$|\s+and.*$|\s+having.*$', '', city)
        
        # Extract minimum rating information using regex
        rating_pattern = r"(\d+(?:\.\d+)?)\s*(?:star|stars|rating|or above|or higher|\+)"
        rating_match = re.search(rating_pattern, question)
        min_rating = None
        if rating_match:
            # Convert matched rating to float
            min_rating = float(rating_match.group(1))
        
        # Check for specific cities - extend this list as needed
        cities = ["boston", "new york", "chicago", "los angeles", "san francisco"]
        detected_city = city
        if not detected_city:
            # If no city was extracted with regex, check for city names directly
            for c in cities:
                if c in question:
                    detected_city = c
                    break
        
        # Determine if we're asking for a count or aggregation
        is_count_query = any(word in question for word in ["how many", "count", "number of"])
        is_avg_query = any(word in question for word in ["average", "avg", "mean"])
        is_top_query = any(word in question for word in ["top", "best", "highest"])
        
        # Build the base query
        if is_count_query:
            query = "SELECT COUNT(*) as restaurant_count FROM COFFEE_SHOPS c WHERE 1=1"
        elif is_avg_query:
            query = """
            SELECT 
                AVG(c.RATING) as avg_rating, 
                AVG(c.SENTIMENT_FOOD) as avg_food_score, 
                AVG(c.SENTIMENT_SERVICE) as avg_service_score, 
                AVG(c.SENTIMENT_AMBIANCE) as avg_ambiance_score
            FROM COFFEE_SHOPS c
            WHERE 1=1
            """
        else:
            query = """
            SELECT 
                c.NAME, c.FULL_ADDRESS as address, c.RATING as stars, 
                c.SENTIMENT_FOOD as food_score, c.SENTIMENT_SERVICE as service_score, c.SENTIMENT_AMBIANCE as ambiance_score,
                (c.SENTIMENT_FOOD + c.SENTIMENT_SERVICE + c.SENTIMENT_AMBIANCE)/3 as overall_score,
                c.LATITUDE, c.LONGITUDE, c.PRICE_LEVEL as price_range, c.TYPE as categories
            FROM COFFEE_SHOPS c
            WHERE 1=1
            """
        
        # Add city filter if detected
        if detected_city:
            # Capitalize city name correctly (e.g., "New York" instead of "new york")
            formatted_city = ' '.join(word.capitalize() for word in detected_city.split())
            query += f" AND c.FULL_ADDRESS LIKE '%{formatted_city}%'"
        
        # Add rating filter if detected
        if min_rating:
            query += f" AND c.RATING >= {min_rating}"
        
        # Check for restaurant type in the question
        if "restaurant" in question or "restaurants" in question or "places to eat" in question:
            query += " AND c.TYPE LIKE '%Restaurant%'"
        
        # Check for cafe/coffee mentions
        if "cafe" in question or "cafes" in question or "coffee" in question:
            query += " AND c.TYPE LIKE '%Cafe%'"
        
        # Check for specific cuisine keywords
        cuisines = ["italian", "chinese", "mexican", "japanese", "thai", "indian", "french"]
        for cuisine in cuisines:
            if cuisine in question:
                query += f" AND c.TYPE LIKE '%{cuisine.capitalize()}%'"
        
        # Check for food mentions
        if "food" in question or "delicious" in question:
            if is_top_query:
                query += " AND c.SENTIMENT_FOOD >= 4.0"
            query += " ORDER BY c.SENTIMENT_FOOD DESC"
        # Check for service mentions
        elif "service" in question or "staff" in question or "waiter" in question:
            if is_top_query:
                query += " AND c.SENTIMENT_SERVICE >= 4.0"
            query += " ORDER BY c.SENTIMENT_SERVICE DESC"
        # Check for ambiance mentions
        elif "ambiance" in question or "atmosphere" in question or "romantic" in question:
            if is_top_query:
                query += " AND c.SENTIMENT_AMBIANCE >= 4.0"
            query += " ORDER BY c.SENTIMENT_AMBIANCE DESC"
        # Default ordering
        else:
            if is_top_query:
                query += " AND (c.SENTIMENT_FOOD + c.SENTIMENT_SERVICE + c.SENTIMENT_AMBIANCE)/3 >= 4.0"
            query += " ORDER BY (c.SENTIMENT_FOOD + c.SENTIMENT_SERVICE + c.SENTIMENT_AMBIANCE)/3 DESC"
        
        # Check for specific ambiance/environment preferences
        if "romantic" in question:
            query += " AND c.SENTIMENT_AMBIANCE >= 4.0"
        
        # Limit the number of results
        limit = 10  # Default limit
        limit_pattern = r"(?:show|get|find|list)\s+(?:top\s+)?(\d+)"
        limit_match = re.search(limit_pattern, question)
        if limit_match:
            limit = int(limit_match.group(1))
        
        # If it's not an aggregation query, add limit
        if not is_avg_query:
            query += f" LIMIT {limit}"
        
        return query
    
    def get_restaurant_stats(self, city: Optional[str] = None) -> Dict[str, Any]:
        """
        Get restaurant statistics for dashboard
        
        Args:
            city: Optional city to filter results
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        try:
            # Build the base query
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
            
            # Add city filter if specified
            if city:
                query += f" WHERE FULL_ADDRESS LIKE '%{city}%'"
            
            # Execute query
            results, _ = self.execute_query(query)
            
            if results and len(results) > 0:
                stats["summary"] = results[0]
            
            # Get rating distribution
            rating_query = """
            SELECT 
                FLOOR(RATING * 2) / 2 as rating_bin,
                COUNT(*) as count
            FROM 
                COFFEE_SHOPS
            """
            
            if city:
                rating_query += f" WHERE FULL_ADDRESS LIKE '%{city}%'"
            
            rating_query += """
            GROUP BY 
                rating_bin
            ORDER BY 
                rating_bin
            """
            
            rating_results, _ = self.execute_query(rating_query)
            stats["rating_distribution"] = rating_results
            
            # Get top restaurants
            top_query = """
            SELECT 
                NAME, 
                RATING as stars, 
                SENTIMENT_FOOD as food_score, 
                SENTIMENT_SERVICE as service_score, 
                SENTIMENT_AMBIANCE as ambiance_score
            FROM 
                COFFEE_SHOPS
            """
            
            if city:
                top_query += f" WHERE FULL_ADDRESS LIKE '%{city}%'"
            
            top_query += """
            ORDER BY 
                (SENTIMENT_FOOD + SENTIMENT_SERVICE + SENTIMENT_AMBIANCE)/3 DESC
            LIMIT 10
            """
            
            top_results, _ = self.execute_query(top_query)
            stats["top_restaurants"] = top_results
            
            return stats
            
        except Exception as e:
            print(f"Error getting stats: {str(e)}")
            return {"error": str(e)}
    
    def get_category_distribution(self, city: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get distribution of restaurant categories
        
        Args:
            city: Optional city to filter results
            
        Returns:
            List of category counts
        """
        try:
            # 獲取類型分布
            query = """
            SELECT 
                TYPE as category, 
                COUNT(*) as count
            FROM 
                COFFEE_SHOPS
            """
            
            if city:
                query += f" WHERE FULL_ADDRESS LIKE '%{city}%'"
            
            query += """
            GROUP BY 
                TYPE
            ORDER BY 
                count DESC
            LIMIT 15
            """
            
            results, _ = self.execute_query(query)
            return results
            
        except Exception as e:
            print(f"Error getting category distribution: {str(e)}")
            return []