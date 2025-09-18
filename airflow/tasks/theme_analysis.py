# airflow/tasks/theme_analysis.py
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import Dict, List, Any
import os
import datetime
import pickle
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Define SimplifiedTopicModel class outside the function so it can be pickled
class SimplifiedTopicModel:
    def __init__(self, vectorizer, kmeans, docs, topics):
        self.vectorizer = vectorizer
        self.kmeans = kmeans
        self.docs = docs
        self.topics = topics
        self.feature_names = vectorizer.get_feature_names_out() if hasattr(vectorizer, 'get_feature_names_out') else []
    
    def get_topic_info(self):
        counts = np.bincount(self.topics)
        data = []
        for i in range(len(counts)):
            data.append({
                'Topic': i,
                'Count': int(counts[i]),
                'Name': f"Topic {i}",
                'Representation': self._get_topic_words(i)
            })
        return pd.DataFrame(data)
    
    def get_topic(self, topic_id):
        words = self._get_top_words(topic_id)
        return [(word, 0.5) for word in words]  # Dummy weights
    
    def _get_top_words(self, topic_id):
        if len(self.docs) < 3:  # Very small set
            all_words = []
            for doc in self.docs:
                all_words.extend([w for w in doc.lower().split() if len(w) > 3])
            from collections import Counter
            return [word for word, _ in Counter(all_words).most_common(10)]
        else:
            mask = np.array(self.topics) == topic_id
            if not any(mask):
                return ["no", "specific", "words", "found"]
            
            docs_in_topic = [self.docs[i] for i, m in enumerate(mask) if m]
            try:
                tfidf = TfidfVectorizer(max_features=20).fit_transform(docs_in_topic)
                importance = np.asarray(tfidf.sum(axis=0)).flatten()
                indices = importance.argsort()[-10:][::-1]
                return [self.vectorizer.get_feature_names_out()[i] for i in indices if i < len(self.feature_names)]
            except:
                # Fallback for very small document sets
                all_words = []
                for doc in docs_in_topic:
                    all_words.extend([w for w in doc.lower().split() if len(w) > 3])
                from collections import Counter
                return [word for word, _ in Counter(all_words).most_common(10)]
    
    def _get_topic_words(self, topic_id):
        words = self._get_top_words(topic_id)
        return ", ".join(words[:5]) if words else "No specific keywords"

def run_topic_modeling(review_file: str, output_dir: str, min_reviews: int = 10) -> Dict[str, Any]:
    """
    Run topic modeling on reviews for businesses

    Args:
        review_file: Path to the JSON file containing reviews
        output_dir: Directory to save output files
        min_reviews: Minimum number of reviews required for full BERTopic (default: 10)

    Returns:
        Dictionary containing topic modeling results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set the environment variable for tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load reviews
    print(f"Loading reviews from {review_file}...")
    with open(review_file, 'r', encoding='utf-8') as f:
        reviews = json.load(f)
    print(f"Loaded {len(reviews)} reviews")

    # Check review structure
    if reviews and len(reviews) > 0:
        first_review = reviews[0]
        print(f"Review fields: {list(first_review.keys())}")
    
    # Group reviews by business_id
    business_reviews = {}
    for review in reviews:
        bid = review.get('business_id')
        if bid:
            if bid not in business_reviews:
                business_reviews[bid] = []
            business_reviews[bid].append(review)
    
    print(f"Found reviews for {len(business_reviews)} businesses")
    
    # Initialize embedding model for all businesses
    print("Initializing sentence transformer model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    business_topics = {}

    for business_id, biz_reviews in business_reviews.items():
        print(f"\nProcessing business: {business_id}")
        print(f"This business has {len(biz_reviews)} reviews")
        
        # Skip businesses with too few reviews
        if len(biz_reviews) < min_reviews:
            print(f"Warning: Business has fewer than {min_reviews} reviews. Using simplified topic modeling.")
        
        # Convert to DataFrame
        review_df = pd.DataFrame(biz_reviews)
        
        # Add datetime column for time-based analysis
        try:
            review_df['datetime'] = pd.to_datetime(review_df['review_datetime_utc'])
            print("Successfully converted date to datetime format")
        except Exception as e:
            print(f"Date conversion error: {str(e)}")
            # Try using timestamp
            try:
                review_df['datetime'] = pd.to_datetime(review_df['review_timestamp'], unit='s')
                print("Successfully converted timestamp to datetime format")
            except Exception as e:
                print(f"Error converting timestamp: {str(e)}")
                # Use placeholder dates
                review_df['datetime'] = pd.date_range(start='2023-01-01', periods=len(review_df))
                print("Using placeholder dates")
            
        # Sort by date
        review_df = review_df.sort_values('datetime')
        
        # Get review texts
        docs = review_df['review_text'].tolist()
        
        # Convert datetime objects to string format
        timestamps = review_df['datetime'].dt.strftime('%Y-%m-%d').tolist()
        print(f"Timestamp examples: {timestamps[:3] if len(timestamps) >= 3 else timestamps}")
        
        # Initialize model and run topic modeling
        if len(docs) < min_reviews:
            # For small document collections, use simplified approach
            print("Using simplified topic modeling for small document collection...")
            # Extract TF-IDF features
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf = vectorizer.fit_transform(docs)
            
            # Determine number of clusters (min 1, max 3 for small sets)
            n_clusters = min(max(1, len(docs) // 2), 3)
            
            # Apply KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(tfidf)
            topics = kmeans.labels_
            
            # Create simplified topic model
            topic_model = SimplifiedTopicModel(vectorizer, kmeans, docs, topics)
        else:
            # For larger document collections, use standard BERTopic
            print("Running standard BERTopic for larger document collection...")
            try:
                vectorizer = CountVectorizer(stop_words="english")
                topic_model = BERTopic(
                    embedding_model=embedding_model,
                    vectorizer_model=vectorizer,
                    min_topic_size=2,  # Allow smaller topics
                    verbose=True
                )
                topics, _ = topic_model.fit_transform(docs)
                print(f"Found {len(set(topics))} unique topics")
            except Exception as e:
                print(f"Error in BERTopic: {str(e)}")
                print("Falling back to simplified approach...")
                
                # Fallback to simplified approach
                vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                tfidf = vectorizer.fit_transform(docs)
                n_clusters = min(max(1, len(docs) // 3), 5)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans.fit(tfidf)
                topics = kmeans.labels_
                topic_model = SimplifiedTopicModel(vectorizer, kmeans, docs, topics)
        
        # Get topic info regardless of model type
        topic_info = topic_model.get_topic_info()
        
        # Get topic keywords
        topic_keywords = {}
        for topic_id in topic_info['Topic'].tolist():
            if topic_id == -1:  # Skip outlier topic
                continue
            words = [word for word, _ in topic_model.get_topic(topic_id)]
            topic_keywords[topic_id] = words
        
        # Analyze topic evolution over time
        print("Analyzing topic trends over time...")
        try:
            # Check if this is a BERTopic model with topics_over_time method
            if hasattr(topic_model, 'topics_over_time'):
                str_timestamps = [str(ts) for ts in timestamps]
                topics_over_time = topic_model.topics_over_time(docs, topics, str_timestamps)
                print("Topic trend analysis completed successfully")
            else:
                raise AttributeError("Model doesn't have topics_over_time method")
        except Exception as e:
            print(f"Error during topic trend analysis: {str(e)}")
            print("Using fallback method for time analysis...")
            
            # Fallback: Create a simple time-series DataFrame
            time_data = []
            
            # Group by month to reduce data points
            date_counts = {}
            
            # Ensure topics is defined
            if 'topics' not in locals():
                if hasattr(topic_model, 'topics'):
                    topics = topic_model.topics
                else:
                    # Get topics from the model
                    topics = [0] * len(docs)  # Default to single topic if nothing else works
            
            for topic, timestamp in zip(topics, timestamps):
                month_key = timestamp[:7]  # Get YYYY-MM
                key = (int(topic), month_key)
                date_counts[key] = date_counts.get(key, 0) + 1
                
            for (topic, month), count in date_counts.items():
                time_data.append({
                    'Topic': topic,
                    'Timestamp': month,
                    'Count': count
                })
                
            topics_over_time = pd.DataFrame(time_data)
            print(f"Fallback topic time data created with {len(topics_over_time)} rows")
        
        # Store results
        business_topics[business_id] = {
            'topic_model': topic_model,
            'topics': topics if 'topics' in locals() else topic_model.topics,
            'topic_info': topic_info,
            'topic_keywords': topic_keywords,
            'topics_over_time': topics_over_time
        }
        
        # Save topic info
        topic_info_path = os.path.join(output_dir, f'topic_info_{business_id}.csv')
        topic_info.to_csv(topic_info_path, index=False)
        print(f"Topic info saved to {topic_info_path}")
        
        # Save topic trends over time
        topics_over_time_path = os.path.join(output_dir, f'topics_over_time_{business_id}.csv')
        topics_over_time.to_csv(topics_over_time_path, index=False)
        print(f"Topic trend over time saved to {topics_over_time_path}")
        
        # Save the model
        model_path = os.path.join(output_dir, f'topic_model_{business_id}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(topic_model, f)
        print(f"Topic model saved to {model_path}")
    
    # Create summary of trends
    print("\nCreating trend summary...")
    trend_summary = []
    for business_id, results in business_topics.items():
        topic_info = results['topic_info']
        topic_keywords = results['topic_keywords']
        
        # Get top topics (up to 3)
        topic_info_filtered = topic_info[topic_info['Topic'] != -1] if -1 in topic_info['Topic'].values else topic_info
        top_topics = topic_info_filtered.sort_values('Count', ascending=False)
        top_topics = top_topics.head(min(3, len(top_topics)))
        
        for _, row in top_topics.iterrows():
            topic_id = row['Topic']
            count = row['Count']
            keywords = topic_keywords.get(topic_id, [])
            
            trend_summary.append({
                'business_id': business_id,
                'topic_id': topic_id,
                'count': count,
                'keywords': ', '.join(keywords[:5] if keywords else ["No specific keywords"]),
                'name': f"Topic {topic_id}: {', '.join(keywords[:3] if keywords else ['No specific keywords'])}"
            })
    
    # Create DataFrame with trend summary
    if trend_summary:
        trend_summary_df = pd.DataFrame(trend_summary)
        
        # Save trend summary
        trend_summary_path = os.path.join(output_dir, 'trend_summary.csv')
        trend_summary_df.to_csv(trend_summary_path, index=False)
        print(f"Trend summary saved to {trend_summary_path}")
    else:
        print("No trend summary created - no topics found")
        trend_summary_path = None
    
    return {
        'trend_summary_path': trend_summary_path,
        'business_topics': business_topics
    }


if __name__ == "__main__":
    # Automatically find files
    import os
    from pathlib import Path
    
    # Get project root directory
    current_dir = Path(__file__).parent
    project_dir = current_dir.parent
    
    # Define data directories
    processed_dir = project_dir / "data" / "processed"
    output_dir = project_dir / "data" / "trend_data"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Look for reviews file first
    review_file = None
    for candidate in [
        processed_dir / "reviews.json",
        project_dir / "data" / "reviews.json",
        *list(processed_dir.glob("*review*.json")),
        *list(project_dir.glob("**/reviews.json"))
    ]:
        if candidate.exists():
            review_file = candidate
            break
    
    if not review_file:
        print("Could not find reviews file automatically.")
        review_file_input = input("Please enter the path to your reviews JSON file: ")
        review_file = Path(review_file_input)
        
        if not review_file.exists():
            print(f"Error: File {review_file} not found")
            exit(1)
    
    print(f"Using reviews file: {review_file}")
    
    # Run topic modeling with more flexible settings for small document collections
    results = run_topic_modeling(str(review_file), str(output_dir), min_reviews=3)
    if results.get('trend_summary_path'):
        print(f"Topic modeling results saved to {results['trend_summary_path']}")
    else:
        print("Topic modeling completed but no trend summary was generated")