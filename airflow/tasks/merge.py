import json
import csv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_directory_exists(file_path):
    """Ensure the directory exists; create it if it doesn't"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        logger.info(f"Creating directory: {directory}")
        os.makedirs(directory)

def integrate_data(cafes_file, sentiment_file, topic_file, output_json):
    """
    Integrate cafe information, sentiment analysis, and topic modeling data, then export as JSON

    Args:
        cafes_file (str): Path to the JSON file with basic cafe info (boston_cafes.json)
        sentiment_file (str): Path to the JSON file with sentiment analysis results
        topic_file (str): Path to the CSV file with topic modeling results
        output_json (str): Path to save the integrated JSON output
    """
    try:
        # Ensure the output directory exists
        ensure_directory_exists(output_json)
        
        # Load cafe data
        logger.info(f"Loading cafe data from: {cafes_file}")
        with open(cafes_file, 'r', encoding='utf-8') as f:
            cafes_json = json.load(f)
            cafes_data = cafes_json.get('data', cafes_json)

        # Load sentiment data
        logger.info(f"Loading sentiment data from: {sentiment_file}")
        with open(sentiment_file, 'r', encoding='utf-8') as f:
            sentiment_data = json.load(f)

        # Load topic data
        logger.info(f"Loading topic data from: {topic_file}")
        topic_data = []
        with open(topic_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                topic_data.append(row)

        logger.info(f"Loaded {len(cafes_data)} cafes, {len(sentiment_data)} sentiment records, {len(topic_data)} topic records")

        # Map sentiment data (business_id -> data)
        sentiment_mapping = {}
        for record in sentiment_data:
            business_id = record['business_id']
            sentiment_mapping[business_id] = {
                'scores': record['scores'],
                'sentiment_percentages': record.get('sentiment_percentages', {})
            }

        # Map topic data (business_id -> [topics])
        topic_mapping = {}
        for record in topic_data:
            business_id = record['business_id']
            if business_id not in topic_mapping:
                topic_mapping[business_id] = []
            try:
                count = int(record['count'])
            except ValueError:
                count = 0

            topic_mapping[business_id].append({
                'topic_id': record.get('topic_id', ''),
                'count': count,
                'keywords': record.get('keywords', ''),
                'name': record.get('name', '')
            })

        # Integrate all data
        integrated_cafes = []
        for cafe in cafes_data:
            business_id = cafe['business_id']

            # Add sentiment scores
            if business_id in sentiment_mapping:
                cafe['sentiment_scores'] = sentiment_mapping[business_id]['scores']
                cafe['sentiment_percentages'] = sentiment_mapping[business_id]['sentiment_percentages']

            # Add topic data
            if business_id in topic_mapping:
                cafe['topics'] = topic_mapping[business_id]

            integrated_cafes.append(cafe)

        # Save the integrated data to JSON
        logger.info(f"Saving integrated data to: {output_json}")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(integrated_cafes, f, ensure_ascii=False, indent=2)

        return integrated_cafes

    except Exception as e:
        logger.error(f"Error occurred during integration: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    cafes_file = "boston_cafes_data/boston_cafes.json"
    sentiment_file = "data/sentiment/cafe_sentiment_with_tiers.json"
    topic_file = "data/trend_data/trend_summary.csv"
    output_json = "data/merge/integrated_cafes.json"

    try:
        integrated_data = integrate_data(
            cafes_file,
            sentiment_file,
            topic_file,
            output_json
        )

        print(f"Integrated data for {len(integrated_data)} cafes.")

        # Count cafes with sentiment and topic data
        with_sentiment = sum(1 for c in integrated_data if 'sentiment_scores' in c)
        with_topics = sum(1 for c in integrated_data if 'topics' in c)

        print(f"Cafes with sentiment scores: {with_sentiment}")
        print(f"Cafes with topic data: {with_topics}")

        # Print example of the first cafe
        if integrated_data:
            print("\nExample: First cafe")
            first_cafe = integrated_data[0]
            print(f"Name: {first_cafe['name']}")
            print(f"Address: {first_cafe['full_address']}")

            if 'sentiment_scores' in first_cafe:
                scores = first_cafe['sentiment_scores']
                print(f"Service: {scores['service']} ({scores['service_tier']})")
                print(f"Food: {scores['food']} ({scores['food_tier']})")
                print(f"Ambiance: {scores['ambiance']} ({scores['ambiance_tier']})")

                if 'sentiment_percentages' in first_cafe:
                    pct = first_cafe['sentiment_percentages']
                    print("\nSentiment percentages:")
                    for dim in ['service', 'food', 'ambiance']:
                        print(f"  {dim}: Positive {pct.get(f'{dim}_good_pct', 0)*100:.1f}%, " +
                              f"Neutral {pct.get(f'{dim}_neutral_pct', 0)*100:.1f}%, " +
                              f"Negative {pct.get(f'{dim}_bad_pct', 0)*100:.1f}%")

            if 'topics' in first_cafe:
                print("\nMain topics:")
                for topic in first_cafe['topics']:
                    print(f"  - {topic['name']} (mentions: {topic['count']})")

        print(f"\nIntegration complete! Output saved to: {output_json}")

    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
