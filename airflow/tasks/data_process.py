import json
import pandas as pd
import os
from typing import List, Dict, Any
from pathlib import Path

# project root, e.g. /opt/airflow
BASE_DIR = Path(__file__).resolve().parents[1]

BUSINESS_PATH = BASE_DIR / "boston_cafes_data" / "boston_cafes.json"
REVIEWS_DIR   = BASE_DIR / "boston_cafes_data"
OUTPUT_DIR    = BASE_DIR / "data" / "processed"

def load_json_file(file_path: str) -> Any:
    """
    Load JSON file into a Python object (dict or list)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_string(text):
    """
    Clean string values by replacing escape characters and Unicode characters
    
    Args:
        text: String to clean
        
    Returns:
        Cleaned string
    """
    if not isinstance(text, str):
        return text
    
    # Replace common escape sequences and Unicode characters
    replacements = {
        '\\/': '/',
        '\\u2013': '-',  # Em dash to regular hyphen
        '\\u2014': '-',  # Em dash to regular hyphen
        '\\u2018': "'",  # Left single quotation mark
        '\\u2019': "'",  # Right single quotation mark
        '\\u201C': '"',  # Left double quotation mark
        '\\u201D': '"',  # Right double quotation mark
        '\\u00A0': ' ',  # Non-breaking space
        '\\u00A9': '(c)',  # Copyright symbol
        '\\u00AE': '(R)',  # Registered trademark
        '\\u2026': '...',  # Ellipsis
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Fix price range formats
    if '–' in text and ('$1' in text or '$10' in text or '$20' in text or '$30' in text):
        text = text.replace('–', '-')
    
    return text

def process_business_data(data: Dict[Any, Any]) -> pd.DataFrame:
    """
    Process business data from Google Maps API response
    
    Args:
        data: Dictionary containing Google Maps API response
        
    Returns:
        DataFrame containing processed business data
    """
    # Extract businesses from the 'data' list
    businesses = data.get('data', [])
    
    # Create DataFrame
    business_df = pd.DataFrame(businesses)
    
    # Select relevant columns if available
    if not business_df.empty:
        relevant_columns = [
            'business_id', 'name', 'full_address',
            'review_count', 'rating', 'phone_number', 'website', 'type',
            'subtypes', 'price_level', 'latitude', 'longitude'  # 添加了latitude和longitude字段
        ]
        
        # Only include columns that exist
        existing_columns = [col for col in relevant_columns if col in business_df.columns]
        business_df = business_df[existing_columns]
    
    return business_df

def process_review_data(data: Any) -> Dict[str, pd.DataFrame]:
    """
    Process review data from the reviews JSON file
    
    Args:
        data: JSON data containing reviews (can be list of business reviews)
        
    Returns:
        Dictionary of DataFrames containing review data for each business
    """
    # Create a dictionary to store reviews for each business
    business_reviews = {}
    
    # Handle the structure as seen in all_reviews.json
    if isinstance(data, list):
        # This is an array of business review objects
        for business_review in data:
            if isinstance(business_review, dict):
                # Extract business_id from parameters
                business_id = business_review.get('parameters', {}).get('business_id')
                
                if business_id and 'data' in business_review:
                    # Extract reviews for this business
                    reviews = business_review.get('data', [])
                    
                    # Filter reviews that have text (not null)
                    text_reviews = [review for review in reviews if review.get('review_text') is not None]
                    
                    if text_reviews:
                        review_df = pd.DataFrame(text_reviews)
                        
                        # Select relevant columns if available
                        relevant_columns = [
                            'review_id', 'review_text', 'rating', 'review_datetime_utc',
                            'review_timestamp', 'like_count'
                        ]
                        
                        # Only include columns that exist
                        existing_columns = [col for col in relevant_columns if col in review_df.columns]
                        if existing_columns:
                            review_df = review_df[existing_columns]
                            business_reviews[business_id] = review_df
                            print(f"Processed {len(text_reviews)} text reviews for business {business_id}")
    
    return business_reviews

def combine_all_reviews(review_files_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Process all reviews from all_reviews.json file
    
    Args:
        review_files_dir: Directory containing all_reviews.json file
        
    Returns:
        Dictionary of DataFrames containing review data for each business
    """
    all_reviews = {}
    
    # Path to all_reviews.json which contains all reviews for all cafes
    all_reviews_path = os.path.join(review_files_dir, 'all_reviews.json')
    
    if os.path.exists(all_reviews_path):
        print(f"Found all_reviews.json at {all_reviews_path}")
        try:
            data = load_json_file(all_reviews_path)
            
            # Process reviews for businesses
            business_reviews = process_review_data(data)
            
            # Add to all_reviews dictionary
            all_reviews.update(business_reviews)
            
            print(f"Extracted reviews for {len(all_reviews)} businesses from all_reviews.json")
        except Exception as e:
            print(f"Error processing all_reviews.json: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Warning: all_reviews.json not found in {review_files_dir}")
    
    return all_reviews

def run_data_processing(business_path: str, reviews_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    Run the data processing pipeline
    
    Args:
        business_path: Path to business JSON file (boston_cafes.json)
        reviews_dir: Directory containing all_reviews.json
        output_dir: Directory to save output files
        
    Returns:
        Dictionary containing metadata and paths to processed data
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process business data
    print(f"Loading business data from {business_path}")
    business_data = load_json_file(business_path)
    business_df = process_business_data(business_data)
    print(f"Processed {len(business_df)} businesses")
    
    # Save business dataframe as JSON directly without pandas
    business_output_path = os.path.join(output_dir, 'business.json')
    
    # Convert DataFrame to list of dictionaries and fix escape characters
    business_list = []
    for _, row in business_df.iterrows():
        business_dict = row.to_dict()
        # Clean string values
        for key, value in business_dict.items():
            if isinstance(value, str):
                business_dict[key] = clean_string(value)
        business_list.append(business_dict)
    
    # Write directly to JSON file
    with open(business_output_path, 'w', encoding='utf-8') as f:
        json.dump(business_list, f, indent=2, ensure_ascii=False)
    
    print(f"Saved business data to {business_output_path}")
    
    # Process all review files
    print(f"Processing reviews from {reviews_dir}")
    review_dfs = combine_all_reviews(reviews_dir)
    print(f"Found reviews for {len(review_dfs)} businesses")
    
    # Save review dataframes
    all_reviews = []
    for business_id, review_df in review_dfs.items():
        print(f"Processing {len(review_df)} reviews for business {business_id}")
        
        # Convert the DataFrame to a list of dictionaries
        reviews_list = review_df.to_dict(orient='records')
        
        # Clean string values and add business_id
        for review in reviews_list:
            # Add business_id to each review if not already present
            if 'business_id' not in review:
                review['business_id'] = business_id
                
            # Clean string values
            for key, value in review.items():
                if isinstance(value, str):
                    review[key] = clean_string(value)
        
        all_reviews.extend(reviews_list)
    
    # Save all reviews to a single file
    reviews_output_path = os.path.join(output_dir, 'reviews.json')
    with open(reviews_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_reviews, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(all_reviews)} reviews to {reviews_output_path}")
    
    # Return metadata
    return {
        'business_path': business_output_path,
        'reviews_path': reviews_output_path,
        'business_count': len(business_df),
        'review_count': len(all_reviews)
    }

if __name__ == "__main__":
    metadata = run_data_processing(
        str(BUSINESS_PATH),
        str(REVIEWS_DIR),
        str(OUTPUT_DIR)
    )
    print(f"Processed data saved to {OUTPUT_DIR}")