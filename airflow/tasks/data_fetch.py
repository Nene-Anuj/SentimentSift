#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
import json
import time
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# API configuration
API_HOST = "local-business-data.p.rapidapi.com"
API_KEY = os.getenv("API_KEY")
HEADERS = {
    "x-rapidapi-host": API_HOST,
    "x-rapidapi-key": API_KEY
}

# Create output directory
OUTPUT_DIR = "boston_cafes_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create a progress tracking file
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.json")

def load_progress():
    """
    Load the progress data from file if it exists
    """
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Progress file exists but is invalid. Creating new progress tracking.")
    
    # Initialize empty progress data
    return {
        "cafes_collected": False,
        "reviews_collected": [],
        "last_updated": datetime.now().isoformat()
    }

def save_progress(progress):
    """
    Save the progress data to file
    """
    progress["last_updated"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def get_boston_cafes(limit=200):
    """
    Get cafe data for Boston
    """
    print(f"Retrieving data for {limit} cafes in Boston...")
    
    # Check if the data already exists
    filename = os.path.join(OUTPUT_DIR, "boston_cafes.json")
    if os.path.exists(filename):
        print(f"Cafe data already exists at {filename}, loading from file...")
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Existing file is invalid. Will fetch new data.")
    
    url = "https://local-business-data.p.rapidapi.com/search"
    params = {
        "query": "Cafe in Boston,MA",
        "limit": limit,
        "lat": 42.3601,  # Corrected latitude for Boston
        "lng": -71.0589, # Corrected longitude for Boston
        "zoom": 13,
        "language": "en",
        "region": "us",
        "extract_emails_and_contacts": "false"
    }
    
    try:
        print("Sending request to get cafe data...")
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        
        # Add delay after API request
        print("Waiting for 2 seconds after API request...")
        time.sleep(2)
        
        # Save the raw JSON response
        cafes_data = response.json()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(cafes_data, f, ensure_ascii=False, indent=2)
            
        print(f"Cafe data saved to {filename}")
        return cafes_data
    
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving cafe data: {e}")
        return None

def get_cafe_reviews(business_id, limit=100):
    """
    Get reviews for a specific cafe
    """
    print(f"Retrieving reviews for cafe {business_id}...")
    
    # Check if the reviews already exist
    filename = os.path.join(OUTPUT_DIR, f"reviews_{business_id.replace(':', '_')}.json")
    if os.path.exists(filename):
        print(f"Review data already exists at {filename}, loading from file...")
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Existing review file is invalid. Will fetch new data.")
    
    url = "https://local-business-data.p.rapidapi.com/business-reviews"
    params = {
        "business_id": business_id,
        "limit": limit,
        "sort_by": "newest",
        "region": "us",
        "language": "en"
    }
    
    try:
        print("Sending request to get review data...")
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        
        # Add delay after API request
        print("Waiting for 2 seconds after API request...")
        time.sleep(2)
        
        # Save the raw JSON response
        reviews_data = response.json()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(reviews_data, f, ensure_ascii=False, indent=2)
            
        print(f"Review data saved to {filename}")
        return reviews_data
        
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving reviews: {e}")
        return None

def main():
    """
    Main function to orchestrate the data collection process
    """
    # Load progress data
    progress = load_progress()
    print(f"Loaded progress data: {progress}")
    
    # Step 1: Get cafes in Boston if not already collected
    cafes_data = None
    if not progress["cafes_collected"]:
        cafes_data = get_boston_cafes(limit=200)
        if cafes_data and "data" in cafes_data:
            progress["cafes_collected"] = True
            save_progress(progress)
    else:
        # Load existing cafe data
        filename = os.path.join(OUTPUT_DIR, "boston_cafes.json")
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                cafes_data = json.load(f)
            print("Loaded existing cafe data")
    
    if not cafes_data or "data" not in cafes_data:
        print("Failed to get cafe data or invalid response format")
        return
    
    # Step 2: Get reviews for each cafe
    cafes = cafes_data.get("data", [])
    print(f"Found {len(cafes)} cafes. Starting to collect reviews...")
    
    # Prepare all reviews data
    all_reviews = []
    
    for i, cafe in enumerate(cafes):
        business_id = cafe.get("business_id")
        name = cafe.get("name", "Unknown")
        
        if not business_id:
            print(f"Skipping cafe {name} - missing business_id")
            continue
        
        # Check if we already collected reviews for this cafe
        if business_id in progress["reviews_collected"]:
            print(f"\n[{i+1}/{len(cafes)}] Already processed: {name}")
            
            # Load the existing reviews
            filename = os.path.join(OUTPUT_DIR, f"reviews_{business_id.replace(':', '_')}.json")
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    reviews_data = json.load(f)
                    all_reviews.append(reviews_data)
            continue
        
        print(f"\n[{i+1}/{len(cafes)}] Processing: {name}")
        
        # Get review data
        reviews_data = get_cafe_reviews(business_id, limit=100)
        
        if reviews_data:
            # Add to the collection of all reviews
            all_reviews.append(reviews_data)
            
            # Update progress
            progress["reviews_collected"].append(business_id)
            save_progress(progress)
        
        # Add delay to avoid exceeding API rate limits
        print("Waiting for 2 seconds before the next request...")
        time.sleep(2)
    
    # Save the merged file of all reviews
    with open(os.path.join(OUTPUT_DIR, "all_reviews.json"), 'w', encoding='utf-8') as f:
        json.dump(all_reviews, f, ensure_ascii=False, indent=2)
    
    print("\nData collection complete!")
    print(f"Collected data for {len(cafes)} cafes")
    print(f"Processed reviews for {len(progress['reviews_collected'])} cafes")
    print(f"All data saved to {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()