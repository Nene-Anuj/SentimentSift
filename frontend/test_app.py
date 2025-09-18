# test_app.py - Integrated version, combining original functionality and new interface design
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from api.rag_agent import ReviewSummarizationAgent

import streamlit as st
import pandas as pd
import plotly.express as px
import time
import random
from dotenv import load_dotenv
from vanna.chromadb import ChromaDB_VectorStore
from vanna.google import GoogleGeminiChat


@st.cache_resource
def get_rag_agent():
    try:
        agent = ReviewSummarizationAgent()
        return agent
    except Exception as e:
        st.error(f"Error initializing RAG agent: {str(e)}")
        return None
    
# Load environment variables
load_dotenv()

# Get configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_DATABASE = "FINAL_DB"
SNOWFLAKE_SCHEMA = "FINAL"
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")  # Using default value

# Set page configuration
st.set_page_config(
    page_title="Boston Coffee Shop Explorer",
    page_icon="☕",
    layout="wide"
)

# Initialize session state
if 'selected_shop' not in st.session_state:
    st.session_state.selected_shop = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "owner_chat" not in st.session_state:
    st.session_state.owner_chat = []

# Custom Vanna class
class BostonCoffeeVanna(ChromaDB_VectorStore, GoogleGeminiChat):
    def __init__(self):
        # Initialize base classes
        ChromaDB_VectorStore.__init__(self)
        GoogleGeminiChat.__init__(self, config={
            'api_key': GEMINI_API_KEY,
            'model': GEMINI_MODEL
        })
        
        # Connect to Snowflake - this is a key step
        self.connect_to_snowflake(
            account=SNOWFLAKE_ACCOUNT,
            username=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA,
            warehouse=SNOWFLAKE_WAREHOUSE
        )
        
        # Add documentation
        self._add_documentation()
        
        # Add example questions
        self._add_example_questions()


        sample_sql_query = """
        SELECT 
            NAME, 
            FULL_ADDRESS, 
            RATING, 
            SENTIMENT_SCORES_SERVICE, 
            SENTIMENT_SCORES_SERVICE_TIER,
            SENTIMENT_SCORES_FOOD,
            SENTIMENT_SCORES_FOOD_TIER,
            SENTIMENT_SCORES_AMBIANCE,
            SENTIMENT_SCORES_AMBIANCE_TIER,
            LATITUDE,
            LONGITUDE,
            TOPICS_1_KEYWORDS,
            TOPICS_1_NAME,
            TOPICS_1_COUNT,
            PRICE_LEVEL,
            REVIEW_COUNT,
            TYPE
        FROM FINAL_DB.FINAL.COFFEE_SHOPS
        WHERE FULL_ADDRESS LIKE '%Boston%'
        ORDER BY RATING DESC
        LIMIT 200;
        """

    def _add_documentation(self):
        """Add database documentation with updated column names"""
        self.add_documentation("""
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
        
        # Sentiment scores (updated column names)
        - SENTIMENT_SCORES_SERVICE: Score for service quality (0-1 scale)
        - SENTIMENT_SCORES_SERVICE_TIER: Categorization of service quality (Tier-1, Tier-2, etc.)
        - SENTIMENT_SCORES_FOOD: Score for food quality (0-1 scale)
        - SENTIMENT_SCORES_FOOD_TIER: Categorization of food quality (Tier-1, Tier-2, etc.)
        - SENTIMENT_SCORES_AMBIANCE: Score for ambiance quality (0-1 scale)
        - SENTIMENT_SCORES_AMBIANCE_TIER: Categorization of ambiance quality (Tier-1, Tier-2, etc.)
        
        # Sentiment percentages
        - SENTIMENT_PERCENTAGES_SERVICE_GOOD_PCT: Percentage of good service mentions
        - SENTIMENT_PERCENTAGES_SERVICE_NEUTRAL_PCT: Percentage of neutral service mentions
        - SENTIMENT_PERCENTAGES_SERVICE_BAD_PCT: Percentage of bad service mentions
        - SENTIMENT_PERCENTAGES_FOOD_GOOD_PCT: Percentage of good food mentions
        - SENTIMENT_PERCENTAGES_FOOD_NEUTRAL_PCT: Percentage of neutral food mentions
        - SENTIMENT_PERCENTAGES_FOOD_BAD_PCT: Percentage of bad food mentions
        - SENTIMENT_PERCENTAGES_AMBIANCE_NEUTRAL_PCT: Percentage of neutral ambiance mentions
        
        # Topics information
        - TOPICS_1_TOPIC_ID: Identifier for topic 1
        - TOPICS_1_COUNT: Number of mentions for topic 1
        - TOPICS_1_KEYWORDS: Keywords associated with topic 1
        - TOPICS_1_NAME: Name of topic 1
        
        - TOPICS_2_TOPIC_ID: Identifier for topic 2
        - TOPICS_2_COUNT: Number of mentions for topic 2
        - TOPICS_2_KEYWORDS: Keywords associated with topic 2
        - TOPICS_2_NAME: Name of topic 2
        
        - TOPICS_3_TOPIC_ID: Identifier for topic 3
        - TOPICS_3_COUNT: Number of mentions for topic 3
        - TOPICS_3_KEYWORDS: Keywords associated with topic 3
        - TOPICS_3_NAME: Name of topic 3
        
        # Additional information
        - WEBSITE: Website URL of the coffee shop
        - TLD: Top Level Domain of the website
        - TIMEZONE: Timezone of the coffee shop location
        - VERIFIED: Whether the business is verified
        - SUBTYPES: Additional subtypes of the business
        - SUBTYPE_GCIDS: Google Category IDs for subtypes
        
        # Working hours
        - WORKING_HOURS_MONDAY: Monday business hours
        - WORKING_HOURS_TUESDAY: Tuesday business hours
        - WORKING_HOURS_WEDNESDAY: Wednesday business hours
        - WORKING_HOURS_THURSDAY: Thursday business hours
        - WORKING_HOURS_FRIDAY: Friday business hours
        - WORKING_HOURS_SATURDAY: Saturday business hours
        - WORKING_HOURS_SUNDAY: Sunday business hours
        
        - ZIPCODE: Postal code of the coffee shop location
        """)
        
    def _add_example_questions(self):
        """Add example questions to train Vanna"""
        self.add_question_sql(
            question="What are the coffee shops with the highest food ratings?",
            sql="""
            SELECT 
                NAME, 
                FULL_ADDRESS, 
                RATING, 
                SENTIMENT_SCORES_FOOD, 
                SENTIMENT_SCORES_FOOD_TIER,
                SENTIMENT_SCORES_SERVICE,
                SENTIMENT_SCORES_SERVICE_TIER,
                SENTIMENT_SCORES_AMBIANCE,
                SENTIMENT_SCORES_AMBIANCE_TIER,
                LATITUDE,
                LONGITUDE,
                TOPICS_1_KEYWORDS,
                TOPICS_1_NAME,
                TOPICS_1_COUNT,
                PRICE_LEVEL,
                TYPE
            FROM FINAL_DB.FINAL.COFFEE_SHOPS
            ORDER BY SENTIMENT_SCORES_FOOD DESC
            LIMIT 10;
            """
        )
        
        self.add_question_sql(
            question="Which coffee shops have the worst service?",
            sql="""
            SELECT 
                NAME, 
                FULL_ADDRESS, 
                RATING, 
                SENTIMENT_SCORES_SERVICE, 
                SENTIMENT_SCORES_SERVICE_TIER,
                SENTIMENT_SCORES_FOOD,
                SENTIMENT_SCORES_FOOD_TIER,
                SENTIMENT_SCORES_AMBIANCE,
                SENTIMENT_SCORES_AMBIANCE_TIER,
                LATITUDE,
                LONGITUDE,
                TOPICS_1_KEYWORDS,
                TOPICS_1_NAME,
                TOPICS_1_COUNT,
                PRICE_LEVEL,
                TYPE
            FROM FINAL_DB.FINAL.COFFEE_SHOPS
            ORDER BY SENTIMENT_SCORES_SERVICE ASC
            LIMIT 10;
            """
        )
        
        self.add_question_sql(
            question="Show me Boston coffee shops with the best ambiance",
            sql="""
            SELECT 
                NAME, 
                FULL_ADDRESS, 
                RATING, 
                SENTIMENT_SCORES_AMBIANCE, 
                SENTIMENT_SCORES_AMBIANCE_TIER,
                SENTIMENT_SCORES_FOOD,
                SENTIMENT_SCORES_FOOD_TIER,
                SENTIMENT_SCORES_SERVICE,
                SENTIMENT_SCORES_SERVICE_TIER,
                LATITUDE,
                LONGITUDE,
                TOPICS_1_KEYWORDS,
                TOPICS_1_NAME,
                TOPICS_1_COUNT,
                PRICE_LEVEL,
                TYPE
            FROM FINAL_DB.FINAL.COFFEE_SHOPS
            WHERE FULL_ADDRESS LIKE '%Boston%'
            ORDER BY SENTIMENT_SCORES_AMBIANCE DESC
            LIMIT 10;
            """
        )
        
        self.add_question_sql(
            question="Which coffee shops have the highest food ratings?",
            sql="""
            SELECT 
                NAME, 
                FULL_ADDRESS, 
                RATING, 
                SENTIMENT_SCORES_FOOD, 
                SENTIMENT_SCORES_FOOD_TIER,
                SENTIMENT_SCORES_SERVICE,
                SENTIMENT_SCORES_SERVICE_TIER,
                SENTIMENT_SCORES_AMBIANCE,
                SENTIMENT_SCORES_AMBIANCE_TIER,
                LATITUDE,
                LONGITUDE,
                TOPICS_1_KEYWORDS,
                TOPICS_1_NAME,
                TOPICS_1_COUNT,
                PRICE_LEVEL,
                TYPE
            FROM FINAL_DB.FINAL.COFFEE_SHOPS
            ORDER BY SENTIMENT_SCORES_FOOD DESC
            LIMIT 10;
            """
        )

# Modified display_restaurant_list_safe function to adjust display format

def display_restaurant_list_safe(df):
    """Safely display coffee shop list, handling various possible error conditions, and displaying price and rating tiers"""
    # Final safety check
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.info("No coffee shops found matching the criteria.")
        return
    
    # Display each coffee shop
    for i, row in df.iterrows():
        try:
            shop_name = row.get('NAME', 'Unknown Shop') if isinstance(row, pd.Series) else 'Unknown Shop'
            shop_type = row.get('TYPE', 'Coffee Shop') if isinstance(row, pd.Series) else 'Coffee Shop'
            
            # Price conversion logic
            original_price = row.get('PRICE_LEVEL', '$$') if isinstance(row, pd.Series) else '$$'
            if original_price == '$':
                display_price = '$1-10'
            elif original_price == '$$':
                display_price = '$10-20'
            else:
                display_price = original_price
            
            with st.expander(f"{shop_name} - {shop_type} ({display_price})"):
                # Create three columns to display rating tiers - only show Tier, not Score
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if isinstance(row, pd.Series) and 'SENTIMENT_SCORES_FOOD_TIER' in row and row['SENTIMENT_SCORES_FOOD_TIER'] is not None:
                        food_tier = row['SENTIMENT_SCORES_FOOD_TIER']
                        st.markdown(f"**Food**  \n{food_tier}")
                
                with col2:
                    if isinstance(row, pd.Series) and 'SENTIMENT_SCORES_AMBIANCE_TIER' in row and row['SENTIMENT_SCORES_AMBIANCE_TIER'] is not None:
                        ambiance_tier = row['SENTIMENT_SCORES_AMBIANCE_TIER']
                        st.markdown(f"**Ambiance**  \n{ambiance_tier}")
                
                with col3:
                    if isinstance(row, pd.Series) and 'SENTIMENT_SCORES_SERVICE_TIER' in row and row['SENTIMENT_SCORES_SERVICE_TIER'] is not None:
                        service_tier = row['SENTIMENT_SCORES_SERVICE_TIER']
                        st.markdown(f"**Service**  \n{service_tier}")
                
                # Display address
                if isinstance(row, pd.Series) and 'FULL_ADDRESS' in row and row['FULL_ADDRESS'] is not None:
                    st.write(f"**Address:** {row['FULL_ADDRESS']}")
                
                # Add a view details button
                if st.button(f"View Details", key=f"view_{i}"):
                    # Ensure conversion to dictionary doesn't error
                    if isinstance(row, pd.Series):
                        st.session_state.selected_shop = row.to_dict()
                    else:
                        st.warning("Unable to get shop details.")
        except Exception as e:
            st.error(f"Error displaying shop {i}: {str(e)}")

def display_shop_details(shop_data):
    """Display detailed information for the selected coffee shop including map and review summary"""
    st.subheader(f"☕ {shop_data['NAME']}")
    
    # Basic information
    st.markdown(f"**Address:** {shop_data['FULL_ADDRESS']}")
    st.markdown(f"**Rating:** {shop_data['RATING']}⭐")
    
    # Price conversion logic
    original_price = shop_data.get('PRICE_LEVEL', '$$') 
    if original_price == '$':
        display_price = '$1-10'
    elif original_price == '$$':
        display_price = '$10-20'
    else:
        display_price = original_price
    
    st.markdown(f"**Price Range:** {display_price}")
    
    # Rating metrics
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        if 'SENTIMENT_SCORES_SERVICE' in shop_data and shop_data['SENTIMENT_SCORES_SERVICE'] is not None:
            service_score = shop_data['SENTIMENT_SCORES_SERVICE']
            if isinstance(service_score, (int, float)):
                st.metric("Service Rating", f"{service_score:.2f}", 
                         delta="Higher" if service_score > 0.5 else "Lower")
        
        if 'SENTIMENT_SCORES_SERVICE_TIER' in shop_data and shop_data['SENTIMENT_SCORES_SERVICE_TIER'] is not None:
            st.write(f"**Tier:** {shop_data['SENTIMENT_SCORES_SERVICE_TIER']}")
    
    with metrics_col2:
        if 'SENTIMENT_SCORES_FOOD' in shop_data and shop_data['SENTIMENT_SCORES_FOOD'] is not None:
            food_score = shop_data['SENTIMENT_SCORES_FOOD']
            if isinstance(food_score, (int, float)):
                st.metric("Food Rating", f"{food_score:.2f}", 
                         delta="Higher" if food_score > 0.5 else "Lower")
        
        if 'SENTIMENT_SCORES_FOOD_TIER' in shop_data and shop_data['SENTIMENT_SCORES_FOOD_TIER'] is not None:
            st.write(f"**Tier:** {shop_data['SENTIMENT_SCORES_FOOD_TIER']}")
    
    with metrics_col3:
        if 'SENTIMENT_SCORES_AMBIANCE' in shop_data and shop_data['SENTIMENT_SCORES_AMBIANCE'] is not None:
            ambiance_score = shop_data['SENTIMENT_SCORES_AMBIANCE']
            if isinstance(ambiance_score, (int, float)):
                st.metric("Ambiance Rating", f"{ambiance_score:.2f}", 
                         delta="Higher" if ambiance_score > 0.5 else "Lower")
        
        if 'SENTIMENT_SCORES_AMBIANCE_TIER' in shop_data and shop_data['SENTIMENT_SCORES_AMBIANCE_TIER'] is not None:
            st.write(f"**Tier:** {shop_data['SENTIMENT_SCORES_AMBIANCE_TIER']}")
    
    # Map view
    st.subheader("📍 Location")
    if 'LATITUDE' in shop_data and 'LONGITUDE' in shop_data:
        lat_val = shop_data["LATITUDE"]
        lon_val = shop_data["LONGITUDE"]
        
        # Ensure coordinates are valid numbers
        if isinstance(lat_val, (int, float)) and isinstance(lon_val, (int, float)) and lat_val != 0 and lon_val != 0:
            map_df = pd.DataFrame({
                "lat": [lat_val],
                "lon": [lon_val],
                "name": [shop_data["NAME"]]
            })
            
            fig = px.scatter_mapbox(
                map_df, 
                lat="lat", 
                lon="lon", 
                hover_name="name",
                zoom=15, 
                height=300,
                mapbox_style="carto-positron"
            )
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("This coffee shop has invalid or incomplete location information.")
    else:
        st.info("This coffee shop has no location information.")
    
    # Review summary section
    st.subheader("💬 Review Summary")
    
    # Get business ID from BUSINESS_ID field
    business_id = shop_data.get('BUSINESS_ID')
    
    # Debug mode to help with troubleshooting
    debug_mode = st.checkbox("Show Debug Info", value=False)
    if debug_mode:
        st.write("Business ID:", business_id)
        
        # Provide sample IDs from Pinecone
        sample_ids = [
            "0x89e37089316eb5663:0x3d5608ce38f956f0",
            "0x89e37bee9336338:0xa58da4d1718a865",
            "0x89e37082d2b824eb:0xc9712079ef572421", 
            "0x89e37103a0008ecb:0x107d299fa6cf29c",
            "0x89e3716acd458b69:0xceb25fc9b660c4d5"
        ]
        
        selected_id = st.selectbox("Try with a sample business ID:", [""] + sample_ids)
        if selected_id:
            business_id = selected_id
            st.write(f"Using sample ID: {business_id}")
    
    # Initialize review summary session state
    if 'review_summary' not in st.session_state:
        st.session_state.review_summary = {}
    
    # Display generate summary button if we have a business ID
    if business_id:
        # If we don't already have a summary for this business
        if business_id not in st.session_state.review_summary:
            if st.button("Generate Review Summary"):
                with st.spinner("Analyzing reviews and generating summary..."):
                    try:
                        # Get RAG agent
                        rag_agent = get_rag_agent()
                        
                        # Generate summary
                        summary = rag_agent.summarize_business_reviews(business_id)
                        
                        # Store summary
                        st.session_state.review_summary[business_id] = summary
                        
                        # Refresh display
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating review summary: {str(e)}")
        
        # Display summary if available
        if business_id in st.session_state.review_summary:
            st.markdown(st.session_state.review_summary[business_id])
        else:
            # Show topic information as a preview
            has_topic_info = False
            
            # Try to display Topic 1 information
            if ('TOPICS_1_KEYWORDS' in shop_data and shop_data['TOPICS_1_KEYWORDS'] is not None):
                st.markdown("### Preview of Topics")
                st.markdown(f"**Topic 1:** {shop_data['TOPICS_1_KEYWORDS']}")
                has_topic_info = True
            
                # Try to display Topic 2 information
                if ('TOPICS_2_KEYWORDS' in shop_data and shop_data['TOPICS_2_KEYWORDS'] is not None):
                    st.markdown(f"**Topic 2:** {shop_data['TOPICS_2_KEYWORDS']}")
                
                # Try to display Topic 3 information
                if ('TOPICS_3_KEYWORDS' in shop_data and shop_data['TOPICS_3_KEYWORDS'] is not None):
                    st.markdown(f"**Topic 3:** {shop_data['TOPICS_3_KEYWORDS']}")
            
            if not has_topic_info:
                st.info("Click 'Generate Review Summary' to analyze reviews and get detailed insights.")
    else:
        # If business ID is missing, show warning and option to enter manually
        st.warning("Business ID not available for this coffee shop.")
        
        manual_id = st.text_input("Enter business ID manually:")
        if manual_id and st.button("Generate with manual ID"):
            with st.spinner("Analyzing reviews..."):
                try:
                    rag_agent = get_rag_agent()
                    summary = rag_agent.summarize_business_reviews(manual_id)
                    st.session_state.review_summary[manual_id] = summary
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Update the main function to initialize the RAG agent
def main():
    # Existing initializations...
    
    # Initialize Vanna and connect to Snowflake
    @st.cache_resource
    def get_vanna():
        try:
            vn = BostonCoffeeVanna()
            return vn
        except Exception as e:
            st.error(f"Error initializing Vanna: {str(e)}")
            return None
    
    # Also initialize the RAG agent for review summaries
    @st.cache_resource
    def get_rag_agent():
        try:
            return ReviewSummarizationAgent()
        except Exception as e:
            st.error(f"Error initializing RAG agent: {str(e)}")
            return None

def customer_explorer_page(vn):
    """Customer browsing page, displaying coffee shop list and details"""
    st.header("Boston Coffee Shop Directory")
    
    # Initialize filtered_shops if it doesn't exist yet
    if "filtered_shops" not in st.session_state:
        st.session_state.filtered_shops = pd.DataFrame(columns=["NAME", "FULL_ADDRESS", "RATING"])
    
    # Get all coffee shop data - increased to 200 shops
    if "all_coffee_shops" not in st.session_state:
        with st.spinner("Loading data..."):
            try:
                query = """
                SELECT 
                    NAME, 
                    FULL_ADDRESS, 
                    RATING, 
                    SENTIMENT_SCORES_SERVICE, 
                    SENTIMENT_SCORES_SERVICE_TIER,
                    SENTIMENT_SCORES_FOOD,
                    SENTIMENT_SCORES_FOOD_TIER,
                    SENTIMENT_SCORES_AMBIANCE,
                    SENTIMENT_SCORES_AMBIANCE_TIER,
                    LATITUDE,
                    LONGITUDE,
                    TOPICS_1_KEYWORDS,
                    TOPICS_1_NAME,
                    TOPICS_1_COUNT,
                    TOPICS_2_KEYWORDS,
                    TOPICS_2_NAME,
                    TOPICS_2_COUNT,
                    TOPICS_3_KEYWORDS,
                    TOPICS_3_NAME,
                    TOPICS_3_COUNT,
                    PRICE_LEVEL,
                    TYPE
                FROM FINAL_DB.FINAL.COFFEE_SHOPS
                WHERE FULL_ADDRESS LIKE '%Boston%'
                ORDER BY RATING DESC
                LIMIT 200;
                """
                
                results = vn.run_sql(query)
                if results is not None and len(results) > 0:
                    st.session_state.all_coffee_shops = pd.DataFrame(results)
                    st.session_state.filtered_shops = st.session_state.all_coffee_shops.copy()
                else:
                    st.error("Unable to fetch coffee shop data.")
                    st.write("SQL Query:", query)
                    return
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                st.write("SQL Query:", query)
                return
    
    # Set up filters
    st.subheader("Filter Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_rating = st.slider("Minimum Rating", 1.0, 5.0, 1.0, 0.5)
    
    with col2:
        if 'all_coffee_shops' in st.session_state and 'TYPE' in st.session_state.all_coffee_shops.columns:
            # Ensure no None or NaN values in type column
            valid_types = st.session_state.all_coffee_shops['TYPE'].dropna().unique().tolist()
            shop_types = ["All"] + sorted(valid_types)
            selected_type = st.selectbox("Shop Type", shop_types)
        else:
            selected_type = "All"
    
    with col3:
        # Modified price options
        price_levels = ["All", "$1-10", "$10-20"]
        selected_price = st.selectbox("Price Range", price_levels)
    
    # Ensure we have data before filtering
    if "all_coffee_shops" in st.session_state and not st.session_state.all_coffee_shops.empty:
        # Apply filters
        filtered_df = st.session_state.all_coffee_shops.copy()
        
        # Filter by rating
        if 'RATING' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['RATING'] >= min_rating]
        
        # Filter by type
        if selected_type != "All" and 'TYPE' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['TYPE'] == selected_type]
        
        # Filter by price - modified price filtering logic
        if selected_price != "All" and 'PRICE_LEVEL' in filtered_df.columns:
            if selected_price == "$1-10":
                filtered_df = filtered_df[filtered_df['PRICE_LEVEL'] == '$']
            elif selected_price == "$10-20":
                filtered_df = filtered_df[filtered_df['PRICE_LEVEL'] == '$$']
        
        st.session_state.filtered_shops = filtered_df
    else:
        st.error("Unable to load or filter coffee shop data")
        return
    
    # Split the page into left and right sections
    col_list, col_detail = st.columns([3, 3])
    
    with col_list:
        # Ensure filtered_df exists
        if 'filtered_shops' in st.session_state and not st.session_state.filtered_shops.empty:
            st.subheader(f"Coffee Shop List ({len(st.session_state.filtered_shops)} shops)")
            # Display coffee shop list
            display_restaurant_list_safe(st.session_state.filtered_shops)
        else:
            st.info("No coffee shops found matching the criteria")
    
    with col_detail:
        if st.session_state.selected_shop is not None:
            # Display details of the selected coffee shop
            display_shop_details(st.session_state.selected_shop)
        else:
            st.info("👈 Please select a coffee shop from the left to view details")

def home_page():
    """Home page displays welcome information and basic introduction"""
    st.title("☕ Boston Coffee Shop Explorer")
    st.write(
        "Welcome! Use the sidebar to navigate between **Coffee Shop Explorer**, **AI Query Assistant**, and **Data Analysis**."
    )
    
    st.markdown("""
    ### Features:
    - **Coffee Shop Explorer**: Browse coffee shops in the Boston area, view ratings and reviews
    - **AI Query Assistant**: Ask questions about coffee shops using natural language
    - **Data Analysis**: Get statistical analysis of coffee shop ratings and reviews
    
    This application uses natural language processing to analyze coffee shop reviews and provide insights based on sentiment analysis.
    """)
    
    # Display some example questions
    st.subheader("Try these questions:")
    examples = [
        "What are the best coffee shops in Boston?",
        "Which coffee shops have the worst service?",
        "Which shops offer Italian-style coffee?",
        "Where are the coffee shops with the highest ambiance ratings?"
    ]
    
    for ex in examples:
        st.markdown(f"- *{ex}*")

# Modified AI query assistant part, fixing column name issues

def ai_query_assistant_page(vn):
    """AI Query Assistant page, allows users to query using natural language"""
    st.header("AI Query Assistant")
    st.write("Ask questions about Boston coffee shops using natural language.")
    
    # Try to fix generated SQL, replacing old column names with new ones
    def fix_generated_sql(sql_query):
        """Fix column name issues in generated SQL"""
        if sql_query is None:
            return None
            
        # Replace all possible old column names
        replacements = {
            'SENTIMENT_FOOD': 'SENTIMENT_SCORES_FOOD',
            'SENTIMENT_SERVICE': 'SENTIMENT_SCORES_SERVICE',
            'SENTIMENT_AMBIANCE': 'SENTIMENT_SCORES_AMBIANCE',
            'SENTIMENT_FOOD_TIER': 'SENTIMENT_SCORES_FOOD_TIER',
            'SENTIMENT_SERVICE_TIER': 'SENTIMENT_SCORES_SERVICE_TIER',
            'SENTIMENT_AMBIANCE_TIER': 'SENTIMENT_SCORES_AMBIANCE_TIER',
            'TOPIC_KEYWORDS': 'TOPICS_1_KEYWORDS',
            'TOPIC_NAME': 'TOPICS_1_NAME',
            'TOPIC_COUNT': 'TOPICS_1_COUNT'
        }
        
        fixed_sql = sql_query
        for old, new in replacements.items():
            fixed_sql = fixed_sql.replace(old, new)
        
        return fixed_sql
    
    # Split the page into left and right sections
    chat_col, results_col = st.columns([2, 4])
    
    with chat_col:
        # Chat interface
        st.subheader("💬 Chat")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        user_question = st.chat_input("Ask a question about coffee shops...", key="chat_input")
        
        if user_question:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # Process user query
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                
                with st.spinner("Thinking..."):
                    try:
                        # Generate SQL
                        generated_sql = vn.generate_sql(user_question)
                        
                        if generated_sql is not None and isinstance(generated_sql, str) and generated_sql.strip():
                            # Fix generated SQL, replacing old column names with new ones
                            fixed_sql = fix_generated_sql(generated_sql)
                            
                            # Display fixed SQL (for debugging)
                            with st.expander("Generated SQL (Debug)"):
                                st.code(fixed_sql, language="sql")
                            
                            # Execute the fixed query
                            results = vn.run_sql(fixed_sql)
                            
                            # Correctly check results
                            if results is not None and len(results) > 0:
                                # Update filtered coffee shop list
                                df = pd.DataFrame(results)
                                st.session_state.filtered_shops = df
                                
                                # Generate response
                                response = f"I found {len(results)} coffee shops that match your criteria. You can view the results on the right."
                            else:
                                # Ensure we don't try to create an empty DataFrame when no results are found
                                if "filtered_shops" in st.session_state:
                                    # Create an empty DataFrame with the correct structure
                                    st.session_state.filtered_shops = pd.DataFrame(columns=["NAME", "FULL_ADDRESS", "RATING"])
                                response = "Sorry, I couldn't find any coffee shops that match your criteria. Please try a different question."
                        else:
                            response = "Sorry, I couldn't understand your question. Please try phrasing it differently."
                        
                        # Display response
                        response_placeholder.markdown(response)
                        
                        # Add response to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = str(e)
                        response = f"Error processing your question: {error_msg}"
                        
                        # If error is related to column names, provide more specific guidance
                        if "invalid identifier" in error_msg:
                            column_name = error_msg.split("invalid identifier '")[1].split("'")[0]
                            response += f"\n\nThe error appears to be related to the column name '{column_name}'. This may be due to changes in the database schema. Please try a different question."
                        
                        response_placeholder.markdown(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    with results_col:
        # Results display
        st.subheader("Query Results")
        
        # Modified condition check to ensure DataFrame exists and is not empty
        if ("filtered_shops" in st.session_state and 
            st.session_state.filtered_shops is not None and 
            isinstance(st.session_state.filtered_shops, pd.DataFrame) and 
            not st.session_state.filtered_shops.empty):
            
            # Display map
            if 'LATITUDE' in st.session_state.filtered_shops.columns and 'LONGITUDE' in st.session_state.filtered_shops.columns:
                # Create map
                st.subheader("📍 Location Distribution")
                map_df = st.session_state.filtered_shops.copy()
                
                # Remove any rows with NaN latitude/longitude
                map_df = map_df.dropna(subset=['LATITUDE', 'LONGITUDE'])
                
                # Create map only if there's data after dropping NaNs
                if not map_df.empty:
                    fig = px.scatter_mapbox(
                        map_df, 
                        lat="LATITUDE", 
                        lon="LONGITUDE", 
                        hover_name="NAME",
                        color="RATING",
                        color_continuous_scale=px.colors.sequential.Viridis,
                        zoom=12, 
                        height=400,
                        mapbox_style="carto-positron"
                    )
                    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough location data to display a map.")
            
            # Display coffee shop list - another safety check
            if st.session_state.filtered_shops is not None and not st.session_state.filtered_shops.empty:
                display_restaurant_list_safe(st.session_state.filtered_shops)
            else:
                st.info("No coffee shops match your criteria.")
        else:
            st.info("Enter a question on the left to start querying.")

def owner_analysis_page(vn):
    """Owner analysis page, providing statistics and insights"""
    st.header("Coffee Shop Data Analysis")
    
    # Get data
    if "all_coffee_shops" not in st.session_state:
        with st.spinner("Loading data..."):
            try:
                query = """
                SELECT 
                    NAME, 
                    FULL_ADDRESS, 
                    RATING, 
                    SENTIMENT_SCORES_SERVICE, 
                    SENTIMENT_SCORES_FOOD,
                    SENTIMENT_SCORES_AMBIANCE,
                    PRICE_LEVEL,
                    TYPE,
                    REVIEW_COUNT,  -- Ensure REVIEW_COUNT field is included
                    SENTIMENT_PERCENTAGES_SERVICE_GOOD_PCT,
                    SENTIMENT_PERCENTAGES_SERVICE_NEUTRAL_PCT,
                    SENTIMENT_PERCENTAGES_SERVICE_BAD_PCT,
                    SENTIMENT_PERCENTAGES_FOOD_GOOD_PCT,
                    SENTIMENT_PERCENTAGES_FOOD_NEUTRAL_PCT,
                    SENTIMENT_PERCENTAGES_FOOD_BAD_PCT,
                    SENTIMENT_PERCENTAGES_AMBIANCE_GOOD_PCT,
                    SENTIMENT_PERCENTAGES_AMBIANCE_NEUTRAL_PCT,
                    SENTIMENT_PERCENTAGES_AMBIANCE_BAD_PCT
                                FROM FINAL_DB.FINAL.COFFEE_SHOPS
                WHERE FULL_ADDRESS LIKE '%Boston%'
                """
                
                results = vn.run_sql(query)
                if results is not None and len(results) > 0:
                    st.session_state.all_coffee_shops = pd.DataFrame(results)
                else:
                    st.error("Unable to fetch coffee shop data.")
                    return
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                return
    
    # Ensure DataFrame is correctly created
    if "all_coffee_shops" not in st.session_state or st.session_state.all_coffee_shops is None or st.session_state.all_coffee_shops.empty:
        st.error("Unable to load coffee shop data.")
        return
        
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Rating Overview", "Competitive Analysis", "Trend Analysis"])
    
    with tab1:
        st.subheader("Rating Overview")
        
        # Calculate average ratings
        df = st.session_state.all_coffee_shops
        
        # Create rating data
        sentiment_data = {
        'Category': ['Service', 'Service', 'Service', 'Food', 'Food', 'Food', 'Ambiance', 'Ambiance', 'Ambiance'],
        'Sentiment': ['Good', 'Neutral', 'Bad'] * 3,
        'Percentage': [
            df["SENTIMENT_PERCENTAGES_SERVICE_GOOD_PCT"].mean(),
            df["SENTIMENT_PERCENTAGES_SERVICE_NEUTRAL_PCT"].mean(),
            df["SENTIMENT_PERCENTAGES_SERVICE_BAD_PCT"].mean(),
            df["SENTIMENT_PERCENTAGES_FOOD_GOOD_PCT"].mean(),
            df["SENTIMENT_PERCENTAGES_FOOD_NEUTRAL_PCT"].mean(),
            df["SENTIMENT_PERCENTAGES_FOOD_BAD_PCT"].mean(),
            df["SENTIMENT_PERCENTAGES_AMBIANCE_GOOD_PCT"].mean(),
            df["SENTIMENT_PERCENTAGES_AMBIANCE_NEUTRAL_PCT"].mean(),
            df["SENTIMENT_PERCENTAGES_AMBIANCE_BAD_PCT"].mean()
        ]
    }
        sentiment_df = pd.DataFrame(sentiment_data)

        pivot_df = sentiment_df.pivot(index='Category', columns='Sentiment', values='Percentage').fillna(0)

        sentiment_colors = {
            'Good': '#6ab04c',
            'Neutral': '#f9ca24',
            'Bad': '#eb4d4b'
        }

        import plotly.graph_objects as go
        fig = go.Figure()
        for sentiment in ['Good', 'Neutral', 'Bad']:
            fig.add_trace(go.Bar(
                name=sentiment,
                x=pivot_df.index,
                y=pivot_df[sentiment],
                marker_color=sentiment_colors[sentiment]
            ))

        fig.update_layout(
            barmode='stack',
            title='Sentiment Breakdown by Category',
            xaxis_title='Category',
            yaxis_title='Percentage',
            legend_title='Sentiment',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)
            
        # Display rating chart
        #st.bar_chart(sentiment_df.set_index("Category"))
        
        # Display highest-rated coffee shops
        st.subheader("Highest Rated Coffee Shops")
        top_shops = df.sort_values("RATING", ascending=False).head(5)
        st.dataframe(top_shops[["NAME", "RATING", "SENTIMENT_SCORES_FOOD", "SENTIMENT_SCORES_SERVICE", "SENTIMENT_SCORES_AMBIANCE"]])
        
        # Display most popular coffee shops - now using REVIEW_COUNT
        st.subheader("Most Popular Coffee Shops (Review Count)")
        popular_shops = df.sort_values("REVIEW_COUNT", ascending=False).head(5)
        st.dataframe(popular_shops[["NAME", "REVIEW_COUNT", "RATING"]])
    
    
    with tab2:
        st.subheader("Competitive Analysis")
        
        # Group analysis by type
        if 'TYPE' in df.columns:
            st.write("Average Ratings by Coffee Shop Type")
            
            # Group by type
            type_stats = df.groupby('TYPE').agg({
                'RATING': 'mean',
                'SENTIMENT_SCORES_FOOD': 'mean',  # Updated column name
                'SENTIMENT_SCORES_SERVICE': 'mean',  # Updated column name
                'SENTIMENT_SCORES_AMBIANCE': 'mean',  # Updated column name
                'NAME': 'count'
            }).reset_index()
            
            type_stats = type_stats.rename(columns={'NAME': 'Shop Count'})
            
            # Display type statistics
            st.dataframe(type_stats)
            
            # Create visualization
            fig = px.bar(
                type_stats, 
                x='TYPE', 
                y=['RATING', 'SENTIMENT_SCORES_FOOD', 'SENTIMENT_SCORES_SERVICE', 'SENTIMENT_SCORES_AMBIANCE'],  # Updated column names
                title="Rating Comparison by Coffee Shop Type",
                labels={'TYPE': 'Type', 'value': 'Rating', 'variable': 'Rating Category'},
                barmode='group'
            )
            st.plotly_chart(fig)
        
        # Analysis by price range
        if 'PRICE_LEVEL' in df.columns:
            st.write("Rating Comparison by Price Range")
            
            # Group by price
            price_stats = df.groupby('PRICE_LEVEL').agg({
                'RATING': 'mean',
                'SENTIMENT_SCORES_FOOD': 'mean',  # Updated column name
                'SENTIMENT_SCORES_SERVICE': 'mean',  # Updated column name
                'SENTIMENT_SCORES_AMBIANCE': 'mean',  # Updated column name
                'NAME': 'count'
            }).reset_index()
            
            price_stats = price_stats.rename(columns={'NAME': 'Shop Count'})
            
            # Reorder price levels
            price_order = ['$', '$$', '$$$', '$$$$']
            price_stats['PRICE_LEVEL'] = pd.Categorical(
                price_stats['PRICE_LEVEL'], 
                categories=price_order,
                ordered=True
            )
            price_stats = price_stats.sort_values('PRICE_LEVEL')
            
            # Display price statistics
            st.dataframe(price_stats)
            
            # Create visualization
            fig = px.line(
                price_stats, 
                x='PRICE_LEVEL', 
                y=['RATING', 'SENTIMENT_SCORES_FOOD', 'SENTIMENT_SCORES_SERVICE', 'SENTIMENT_SCORES_AMBIANCE'],  # Updated column names
                title="Rating Comparison by Price Range",
                labels={'PRICE_LEVEL': 'Price Range', 'value': 'Rating', 'variable': 'Rating Category'},
                markers=True
            )
            st.plotly_chart(fig)
    
    with tab3:
        st.subheader("Trend Analysis")
        
        # Since there's no time data, create simulated data
        st.info("Note: The following is based on randomly generated simulated trend data for demonstration purposes only.")
        
        # Generate simulated trend data
        months = ["January", "February", "March", "April", "May", "June"]
        trend_data = {
            "Month": months,
            "Overall Rating": [random.uniform(4.0, 4.8) for _ in range(6)],
            "Food Rating": [random.uniform(4.2, 4.9) for _ in range(6)],
            "Service Rating": [random.uniform(3.8, 4.7) for _ in range(6)]
        }
        
        trend_df = pd.DataFrame(trend_data)
        
        # Display trend chart
        st.line_chart(trend_df.set_index("Month")[["Overall Rating", "Food Rating", "Service Rating"]])
        
        # Display some trend insights
        st.subheader("Trend Insights")
        st.success("Service rating has improved by 12% over the past 3 months")
        st.error("Food rating has slightly decreased last month (-3%)")
        st.info("Overall rating remains stable with minimal fluctuation")
    
    # Strategic insights chat
    st.markdown("---")
    st.subheader("💬 Strategic Insights Chat")
    st.markdown(
        """
        Use this chat interface to ask about trends, rating analysis, or get strategic recommendations.
        """
    )
    
    # Ensure owner_chat exists in session_state
    if "owner_chat" not in st.session_state:
        st.session_state.owner_chat = []
    
    # Display chat history
    for msg in st.session_state.owner_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    owner_input = st.chat_input("Ask about insights, trends, or shop recommendations...", key="owner_chat_input")
    
    if owner_input:
        # Add user message to chat history
        st.session_state.owner_chat.append({"role": "user", "content": owner_input})
        
        with st.chat_message("user"):
            st.markdown(owner_input)
        
        # Generate simulated response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                time.sleep(1)  # Simulate processing time
                
                # Simple keyword matching
                if "improve" in owner_input or "enhancement" in owner_input:
                    response = """**Improvement Opportunities:**
                    
1. **Service Speed**: Reviews show that 15% of customers mention wait time as an issue. Consider optimizing coffee preparation process or adding staff during peak hours.

2. **Food Variety**: There's an increasing customer demand for more plant-based food options. Adding 3-4 vegan options could attract a new customer segment.

3. **Online Presence**: Your social media engagement is 35% below industry average. Implementing a regular schedule of high-quality coffee photos can increase visibility.

Would you like more specific recommendations for any of these areas?"""
                
                elif "competition" in owner_input:
                    response = """**Competitive Analysis:**
                    
Your coffee shop outperforms competitors in food quality (+8%) and ambiance (+12%), but slightly underperforms in value for money (-5%).

**Key Differentiators of Top Competitors:**
- Competitor B offers a membership program with 22% customer participation
- Competitor A's happy hour specials drive 35% of weekday business
- Competitor C's outdoor seating area generates 40% additional revenue in summer

Consider implementing a membership program or expanding outdoor seating to compete more effectively."""
                
                elif "trend" in owner_input:
                    response = """**Current Market Trends:**
                    
1. **Rising Trend**: Farm-to-table concept showing 28% increase in customer interest
2. **Declining Trend**: Formal dining experience (-12% year-over-year)
3. **Emerging Opportunity**: Sustainable packaging positively mentioned in 34% of reviews

Given your existing supplier relationships, your coffee shop is well-positioned to capitalize on the farm-to-table trend."""
                
                else:
                    response = """**Strategic Insight:**
                    
Based on your coffee shop data, I recommend focusing on your signature drinks that receive the highest ratings (Espresso: 4.8/5, Pour-over: 4.9/5).

Customer feedback shows these products are driving repeat visits, with 45% of returning customers specifically mentioning them.

Consider creating a "Manager's Picks" section on your menu to highlight these products, and potentially increase their price by 8-10% without affecting demand."""
                
                st.markdown(response)
                
                # Add response to chat history
                st.session_state.owner_chat.append({"role": "assistant", "content": response})
# Main function
def main():
    # Initialize Vanna and connect to Snowflake
    @st.cache_resource
    def get_vanna():
        try:
            vn = BostonCoffeeVanna()
            return vn
        except Exception as e:
            st.error(f"Error initializing Vanna: {str(e)}")
            return None
    
    with st.spinner("Connecting to database..."):
        vn = get_vanna()
        
    if vn is None:
        st.error("Unable to connect to Snowflake. Please check your credentials.")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    pages = {
        "Home": home_page,
        "Coffee Shop Explorer": lambda: customer_explorer_page(vn),
        "AI Query Assistant": lambda: ai_query_assistant_page(vn),
        "Data Analysis": lambda: owner_analysis_page(vn)
    }
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    # Display the selected page
    pages[selection]()

if __name__ == "__main__":
    main()