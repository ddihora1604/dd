from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import json
import pandas as pd
from datetime import datetime
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os
import numpy as np
import requests
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import umap.umap_ as umap
import traceback
import math
import glob
import re
import google.generativeai as genai
import logging
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Global variables for model caching
model_cache = {}
MAX_CACHE_SIZE = 2  # Maximum number of models to keep in cache

def load_model_with_cache(model_name, model_loader):
    """Load model with caching and memory management"""
    if model_name in model_cache:
        return model_cache[model_name]
    
    # Check memory before loading new model
    check_memory_threshold()
    
    # Load the model
    model = model_loader()
    
    # Cache management
    if len(model_cache) >= MAX_CACHE_SIZE:
        # Remove oldest model from cache
        oldest_key = next(iter(model_cache))
        del model_cache[oldest_key]
        clear_memory()
    
    model_cache[model_name] = model
    return model

# Initialize models
try:
    # Initialize sentence transformer for embeddings
    semantic_model = None  # Will be loaded on demand to save memory
    
    # Check for Groq API key
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    has_groq = GROQ_API_KEY is not None and GROQ_API_KEY != ""
    if has_groq:
        print("Groq API key found. Enhanced LLM insights will be available.")
    else:
        print("No Groq API key found. Set GROQ_API_KEY in environment or .env file for enhanced insights.")
    
    # Check for Google Gemini API key
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    has_gemini = GEMINI_API_KEY is not None and GEMINI_API_KEY != ""
    if has_gemini:
        try:
            print("Gemini API key found. Enhanced chatbot functionality will be available.")
            # Configure Gemini API
            genai.configure(api_key=GEMINI_API_KEY)
        except Exception as e:
            print(f"Error configuring Gemini API: {e}")
            has_gemini = False
    else:
        print("No Gemini API key found. Set GEMINI_API_KEY in environment or .env file for enhanced chatbot functionality.")
except Exception as e:
    print(f"Error initializing models: {e}")
    semantic_model = None
    has_groq = False
    has_gemini = False

class SocialMediaConnector:
    """Base class for social media platform data connectors."""
    
    def __init__(self, platform_name):
        self.platform_name = platform_name
        self.data = None
    
    def load_data(self, source_path):
        """Load data from the source path."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def normalize_data(self):
        """Normalize data to a standard format."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_data(self):
        """Return the normalized data."""
        return self.data

class RedditConnector(SocialMediaConnector):
    """Connector for Reddit data in JSONL format."""
    
    def __init__(self):
        super().__init__("reddit")
    
    def load_data(self, source_path):
        """Load Reddit data from JSONL file."""
        try:
            if os.path.exists(source_path):
                # Read JSONL file
                self.data = pd.read_json(source_path, lines=True)
                print(f"Loaded {len(self.data)} rows from Reddit data source: {source_path}")
                return True
            else:
                print(f"Reddit data file not found at {source_path}")
                return False
        except Exception as e:
            print(f"Error loading Reddit data: {str(e)}")
            return False
    
    def normalize_data(self):
        """Normalize Reddit data to standard format."""
        if self.data is None:
            return False
        
        try:
            # Normalize the nested JSON structure
            self.data = pd.json_normalize(self.data['data'])
            
            # Convert created_utc to datetime
            self.data['created_utc'] = pd.to_datetime(self.data['created_utc'], unit='s')
            
            # Add platform identifier
            self.data['platform'] = 'reddit'
            
            # Ensure standard column names
            # Map: content_text (post content), title, author, created_at, platform, engagement_count, etc.
            self.data['content_text'] = self.data['selftext'].fillna('')
            self.data['engagement_count'] = self.data['num_comments']
            self.data['created_at'] = self.data['created_utc']
            self.data['post_id'] = self.data['id']
            self.data['community'] = self.data['subreddit']
            
            print(f"Normalized Reddit data: {len(self.data)} rows")
            return True
        except Exception as e:
            print(f"Error normalizing Reddit data: {str(e)}")
            return False

class TwitterConnector(SocialMediaConnector):
    """Connector for Twitter/X data in JSONL format."""
    
    def __init__(self):
        super().__init__("twitter")
    
    def load_data(self, source_path):
        """Load Twitter data from JSONL file."""
        try:
            if os.path.exists(source_path):
                # Read JSONL file
                self.data = pd.read_json(source_path, lines=True)
                print(f"Loaded {len(self.data)} rows from Twitter data source: {source_path}")
                return True
            else:
                print(f"Twitter data file not found at {source_path}")
                return False
        except Exception as e:
            print(f"Error loading Twitter data: {str(e)}")
            return False
    
    def normalize_data(self):
        """Normalize Twitter data to standard format."""
        if self.data is None:
            return False
        
        try:
            # Add platform identifier
            self.data['platform'] = 'twitter'
            
            # Standardize column names - assuming Twitter data structure
            if 'full_text' in self.data.columns:
                self.data['content_text'] = self.data['full_text'].fillna('')
            else:
                self.data['content_text'] = self.data['text'].fillna('')
            
            if 'created_at' in self.data.columns:
                if isinstance(self.data['created_at'].iloc[0], str):
                    self.data['created_at'] = pd.to_datetime(self.data['created_at'])
            
            # Map standard fields
            self.data['title'] = ''  # Twitter doesn't have titles
            self.data['engagement_count'] = self.data.get('retweet_count', 0) + self.data.get('favorite_count', 0)
            self.data['post_id'] = self.data['id_str'] if 'id_str' in self.data.columns else self.data['id']
            self.data['community'] = ''  # Twitter doesn't have direct community equivalent
            
            print(f"Normalized Twitter data: {len(self.data)} rows")
            return True
        except Exception as e:
            print(f"Error normalizing Twitter data: {str(e)}")
            return False

class PlatformDataManager:
    """Manages data from multiple social media platforms."""
    
    def __init__(self):
        self.connectors = {}
        self.integrated_data = None
        self.platform_data = {}
    
    def add_connector(self, connector):
        """Add a platform connector."""
        self.connectors[connector.platform_name] = connector
    
    def load_platform_data(self, platform, source_path):
        """Load and normalize data for a specific platform."""
        if platform not in self.connectors:
            print(f"No connector available for platform: {platform}")
            return False
        
        connector = self.connectors[platform]
        if connector.load_data(source_path):
            if connector.normalize_data():
                self.platform_data[platform] = connector.get_data()
                return True
        return False
    
    def integrate_data(self):
        """Combine data from all platforms into a unified dataset."""
        if not self.platform_data:
            print("No platform data loaded")
            return False
        
        try:
            # Combine all dataframes
            dataframes = []
            for platform, df in self.platform_data.items():
                if not df.empty:
                    dataframes.append(df)
            
            if dataframes:
                self.integrated_data = pd.concat(dataframes, ignore_index=True)
                print(f"Integrated data created with {len(self.integrated_data)} total rows")
                return True
            else:
                print("No valid dataframes to integrate")
                return False
        except Exception as e:
            print(f"Error integrating data: {str(e)}")
            return False
    
    def get_platform_data(self, platform=None):
        """Get data for a specific platform or all integrated data."""
        if platform:
            return self.platform_data.get(platform)
        return self.integrated_data
    
    def get_available_platforms(self):
        """Get list of platforms with loaded data."""
        return list(self.platform_data.keys())
    
    def filter_data(self, query, platform=None):
        """Filter data based on query and optional platform."""
        if platform:
            if platform not in self.platform_data:
                return pd.DataFrame()
            df = self.platform_data[platform]
            return df[
                df['content_text'].str.contains(query, case=False, na=False) | 
                df['title'].str.contains(query, case=False, na=False)
            ]
        else:
            if self.integrated_data is None:
                return pd.DataFrame()
            return self.integrated_data[
                self.integrated_data['content_text'].str.contains(query, case=False, na=False) | 
                self.integrated_data['title'].str.contains(query, case=False, na=False)
            ]
    
    def discover_data_files(self):
        """Automatically discover data files for different platforms."""
        discovered_files = {}
        data_dir = "./data"
        
        # Look for Reddit data files
        reddit_files = glob.glob(f"{data_dir}/*reddit*.jsonl")
        if reddit_files:
            discovered_files["reddit"] = reddit_files[0]
        
        # Look for Twitter data files
        twitter_files = glob.glob(f"{data_dir}/*twitter*.jsonl") + glob.glob(f"{data_dir}/*tweet*.jsonl")
        if twitter_files:
            discovered_files["twitter"] = twitter_files[0]
        
        # Default to data.jsonl as Reddit if no specific files found
        if "reddit" not in discovered_files and os.path.exists(f"{data_dir}/data.jsonl"):
            discovered_files["reddit"] = f"{data_dir}/data.jsonl"
        
        return discovered_files

# Load environment variables from .env file if it exists
load_dotenv()

app = Flask(__name__)
CORS(app)

# Global variable to store the loaded data
data = None
# Path to the dataset file
DATASET_PATH = "./data/data.jsonl"

# Create the platform data manager
platform_manager = PlatformDataManager()

# Global variables for model caching
model_cache = {}
MAX_CACHE_SIZE = 2  # Maximum number of models to keep in cache

def load_model_with_cache(model_name, model_loader):
    """Load model with caching and memory management"""
    if model_name in model_cache:
        return model_cache[model_name]
    
    # Check memory before loading new model
    check_memory_threshold()
    
    # Load the model
    model = model_loader()
    
    # Cache management
    if len(model_cache) >= MAX_CACHE_SIZE:
        # Remove oldest model from cache
        oldest_key = next(iter(model_cache))
        del model_cache[oldest_key]
        clear_memory()
    
    model_cache[model_name] = model
    return model

# Load dataset on startup
def load_dataset():

    global data
    try:
        if os.path.exists(DATASET_PATH):
            # Read JSONL file
            data = pd.read_json(DATASET_PATH, lines=True)
            # Normalize the nested JSON structure
            data = pd.json_normalize(data['data'])
            # Convert created_utc to datetime
            data['created_utc'] = pd.to_datetime(data['created_utc'], unit='s')
            print(f"Dataset loaded successfully: {len(data)} rows")
            return True
        else:
            print(f"Dataset file not found at {DATASET_PATH}")
            return False
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return False

@app.route('/')
def index():

    return render_template('index.html')

@app.route('/api/timeseries', methods=['GET'])
def get_timeseries():
    """
    Generates time series data for posts matching a query within a date range.
    
    This endpoint:
    1. Filters data based on the search query and optional date range
    2. Groups posts by date
    3. Counts posts per date to create the time series
    
    Query Parameters:
        query (str): Search term to filter posts
        start_date (str, optional): Start date for filtering (YYYY-MM-DD)
        end_date (str, optional): End date for filtering (YYYY-MM-DD)
    
    Returns:
        JSON: Array of objects containing date and post count
    """
    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    query = request.args.get('query', '')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Filter data based on query and date range
    filtered_data = data[data['selftext'].str.contains(query, case=False, na=False) | 
                          data['title'].str.contains(query, case=False, na=False)]
    if start_date and end_date:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        filtered_data = filtered_data[
            (filtered_data['created_utc'] >= start_date) & 
            (filtered_data['created_utc'] <= end_date)
        ]
    
    # Group by date and count posts
    timeseries = filtered_data.groupby(filtered_data['created_utc'].dt.date).size().reset_index()
    timeseries.columns = ['date', 'count']
    
    return jsonify(timeseries.to_dict('records'))

@app.route('/api/top_contributors', methods=['GET'])
def get_top_contributors(query=None, limit=10):
 
    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    # If called as an API endpoint, get parameters from request
    if query is None:
        query = request.args.get('query', '')
        limit = int(request.args.get('limit', 10))
    
    filtered_data = data[data['selftext'].str.contains(query, case=False, na=False) | 
                          data['title'].str.contains(query, case=False, na=False)]
    top_users = filtered_data['author'].value_counts().head(limit).reset_index()
    top_users.columns = ['author', 'count']
    
    # Return JSON if called as API endpoint, otherwise return the data
    if request.path == '/api/top_contributors':
        return jsonify(top_users.to_dict('records'))
    return top_users.to_dict('records')

@app.route('/api/network', methods=['GET'])
def get_network():

    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    query = request.args.get('query', '')
    network_type = request.args.get('network_type', 'interaction')
    content_type = request.args.get('content_type', 'all')
    min_similarity = float(request.args.get('min_similarity', 0.2))
    
    # Filter data
    filtered_data = data[
        data['selftext'].str.contains(query, case=False, na=False) |
        data['title'].str.contains(query, case=False, na=False)
    ]
    
    if len(filtered_data) == 0:
        return jsonify({'nodes': [], 'links': []})
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes for all authors
    author_counts = filtered_data['author'].value_counts()
    for author, count in author_counts.items():
        G.add_node(author, size=min(count*3, 30), posts=int(count))
    
    if network_type == 'interaction':
        # Traditional interaction network (unchanged)
        # Add edges based on interactions (comments)
        comment_edges = []
        for idx, row in filtered_data.iterrows():
            if pd.notna(row.get('parent_id')) and row.get('parent_id', '').startswith('t3_'):
                parent_post = filtered_data[filtered_data['id'] == row.get('parent_id')[3:]]
                if not parent_post.empty:
                    parent_author = parent_post.iloc[0]['author']
                    if parent_author != row['author']:  # Don't count self-interactions
                        comment_edges.append((row['author'], parent_author))
        
        # Count edge weights
        edge_weights = {}
        for source, target in comment_edges:
            if (source, target) not in edge_weights:
                edge_weights[(source, target)] = 0
            edge_weights[(source, target)] += 1
        
        # Add weighted edges
        for (source, target), weight in edge_weights.items():
            G.add_edge(source, target, weight=weight)
    
    else:
        # Content-based network
        # Extract shared content between authors
        import re
        from collections import defaultdict
        
        # Functions to extract different content types
        def extract_keywords(text):
            if not isinstance(text, str):
                return []
            # Simple keyword extraction - could be improved with NLP
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            # Filter out common words
            common_words = {'about', 'after', 'again', 'also', 'around', 'before', 'being', 'between',
                           'could', 'every', 'from', 'have', 'here', 'most', 'need', 'other', 'should',
                           'since', 'there', 'these', 'they', 'this', 'those', 'through', 'using',
                           'very', 'what', 'when', 'where', 'which', 'while', 'would', 'your'}
            return [w for w in words if w not in common_words]
        
        def extract_hashtags(text):
            if not isinstance(text, str):
                return []
            return re.findall(r'#[a-zA-Z0-9_]+', text.lower())
        
        def extract_urls(text):
            if not isinstance(text, str):
                return []
            return re.findall(r'https?://\S+', text.lower())
        
        # Extract content for each author
        author_content = defaultdict(lambda: {'keywords': set(), 'hashtags': set(), 'urls': set()})
        
        for _, row in filtered_data.iterrows():
            # Combine title and selftext
            full_text = f"{row['title']} {row.get('selftext', '')}"
            author = row['author']
            
            # Extract content based on requested type
            if content_type in ['all', 'keywords']:
                author_content[author]['keywords'].update(extract_keywords(full_text))
            
            if content_type in ['all', 'hashtags']:
                author_content[author]['hashtags'].update(extract_hashtags(full_text))
            
            if content_type in ['all', 'urls']:
                author_content[author]['urls'].update(extract_urls(full_text))
        
        # Find shared content between authors
        authors = list(author_content.keys())
        content_edges = []
        
        for i in range(len(authors)):
            for j in range(i+1, len(authors)):
                author1 = authors[i]
                author2 = authors[j]
                
                shared_content = {
                    'keywords': author_content[author1]['keywords'].intersection(author_content[author2]['keywords']),
                    'hashtags': author_content[author1]['hashtags'].intersection(author_content[author2]['hashtags']),
                    'urls': author_content[author1]['urls'].intersection(author_content[author2]['urls'])
                }
                
                # Calculate similarity score based on shared content
                total_shared = len(shared_content['keywords']) + len(shared_content['hashtags']) + len(shared_content['urls'])
                
                # Only create edges if there's meaningful shared content
                if total_shared > 0:
                    # Calculate similarity score 
                    author1_total = sum(len(author_content[author1][ct]) for ct in ['keywords', 'hashtags', 'urls'])
                    author2_total = sum(len(author_content[author2][ct]) for ct in ['keywords', 'hashtags', 'urls'])
                    
                    if author1_total > 0 and author2_total > 0:
                        # Jaccard similarity: intersection / union
                        similarity = total_shared / (author1_total + author2_total - total_shared)
                        
                        if similarity >= min_similarity:
                            # Create edge with shared content metadata
                            content_edges.append((
                                author1, 
                                author2, 
                                {
                                    'weight': total_shared,
                                    'similarity': similarity,
                                    'shared_keywords': list(shared_content['keywords'])[:10],  # Limit to top 10
                                    'shared_hashtags': list(shared_content['hashtags']),
                                    'shared_urls': list(shared_content['urls']),
                                    'total_shared': total_shared
                                }
                            ))
        
        # Add content-based edges to graph
        for source, target, attrs in content_edges:
            G.add_edge(source, target, **attrs)
            # Make the graph undirected for content sharing
            G.add_edge(target, source, **attrs)
    
    # Find communities using Louvain method
    if len(G.nodes()) > 0:
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G.to_undirected())
            nx.set_node_attributes(G, partition, 'group')
        except Exception as e:
            print(f"Community detection error: {str(e)}")
            # Fallback if community detection fails
            for node in G.nodes():
                G.nodes[node]['group'] = 0
    
    # Enhance node metadata with content info
    if network_type != 'interaction':
        for node in G.nodes():
            # Add content statistics to nodes
            if node in author_content:
                G.nodes[node]['keyword_count'] = len(author_content[node]['keywords'])
                G.nodes[node]['hashtag_count'] = len(author_content[node]['hashtags'])
                G.nodes[node]['url_count'] = len(author_content[node]['urls'])
                
                # Add top keywords to nodes (limited to 5)
                G.nodes[node]['top_keywords'] = list(author_content[node]['keywords'])[:5]
                G.nodes[node]['top_hashtags'] = list(author_content[node]['hashtags'])[:5]
    
    # Convert to D3.js format
    nodes = []
    for node in G.nodes():
        # Basic node info
        node_data = {
            'id': node, 
            'size': G.nodes[node].get('size', 10),
            'group': G.nodes[node].get('group', 0),
            'posts': G.nodes[node].get('posts', 1)
        }
        
        # Add content metadata if available
        if network_type != 'interaction':
            node_data.update({
                'keyword_count': G.nodes[node].get('keyword_count', 0),
                'hashtag_count': G.nodes[node].get('hashtag_count', 0),
                'url_count': G.nodes[node].get('url_count', 0),
                'top_keywords': G.nodes[node].get('top_keywords', []),
                'top_hashtags': G.nodes[node].get('top_hashtags', [])
            })
        
        nodes.append(node_data)
    
    # Convert links with enhanced metadata
    links = []
    for source, target in G.edges():
        link_data = {
            'source': source, 
            'target': target,
            'weight': G.edges[source, target].get('weight', 1)
        }
        
        # Add shared content info if available
        if network_type != 'interaction':
            link_data.update({
                'similarity': G.edges[source, target].get('similarity', 0),
                'shared_keywords': G.edges[source, target].get('shared_keywords', []),
                'shared_hashtags': G.edges[source, target].get('shared_hashtags', []),
                'shared_urls': G.edges[source, target].get('shared_urls', []),
                'total_shared': G.edges[source, target].get('total_shared', 0)
            })
        
        links.append(link_data)
    
    # Calculate network metrics
    metrics = {
        'node_count': len(nodes),
        'edge_count': len(links),
        'network_type': network_type,
        'content_type': content_type if network_type != 'interaction' else None,
        'density': nx.density(G) if len(G.nodes()) > 1 else 0,
        'avg_degree': sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0
    }
    
    return jsonify({
        'nodes': nodes, 
        'links': links,
        'metrics': metrics,
        'network_type': network_type
    })

@app.route('/api/topics', methods=['GET'])
def get_topics():

    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    n_topics = int(request.args.get('n_topics', 5))
    query = request.args.get('query', '')
    
    # Filter data if query is provided
    filtered_data = data
    if query:
        filtered_data = data[
            data['selftext'].str.contains(query, case=False, na=False) |
            data['title'].str.contains(query, case=False, na=False)
        ]
    
    if len(filtered_data) == 0:
        return jsonify([])
    
    # Create a copy to avoid SettingWithCopyWarning
    filtered_data = filtered_data.copy()
    # Prepare text data - combine title and selftext for better topic detection
    filtered_data.loc[:, 'combined_text'] = filtered_data['title'] + ' ' + filtered_data['selftext'].fillna('')
    
    # Prepare text data
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(filtered_data['combined_text'])
    
    # Apply LDA with improved parameters
    lda = LatentDirichletAllocation(
        n_components=n_topics, 
        random_state=42,
        learning_method='online',
        max_iter=50,
        learning_decay=0.7,
        evaluate_every=10
    )
    
    # Fit the model and transform the data to get document-topic distributions
    doc_topic_dists = lda.fit_transform(X)
    
    # Get top words for each topic with relevance scores
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    
    for topic_idx, topic in enumerate(lda.components_):
        # Get the top words with their weights
        sorted_indices = topic.argsort()[:-20-1:-1]
        top_words = [feature_names[i] for i in sorted_indices]
        top_weights = [float(topic[i]) for i in sorted_indices]
        
        # Normalize weights to percentages for easier interpretation
        total_weight = sum(top_weights)
        top_weights_normalized = [round((w / total_weight) * 100, 2) for w in top_weights]
        
        # Create topic object with enhanced metadata
        topic_obj = {
            'topic_id': topic_idx,
            'top_words': top_words[:15],  # Top 15 words
            'weights': top_weights_normalized[:15],  # Corresponding weights
            'word_weight_pairs': [{'word': w, 'weight': round(wt, 2)} 
                                  for w, wt in zip(top_words[:15], top_weights_normalized[:15])]
        }
        
        # Find representative documents for this topic
        topic_docs = []
        for i, dist in enumerate(doc_topic_dists):
            if np.argmax(dist) == topic_idx and dist[topic_idx] > 0.5:  # Strong topic alignment
                if len(topic_docs) < 3:  # Limit to 3 examples
                    doc = filtered_data.iloc[i]
                    topic_docs.append({
                        'title': doc['title'],
                        'author': doc['author'],
                        'subreddit': doc.get('subreddit', ''),
                        'created_utc': doc['created_utc'].isoformat(),
                        'topic_probability': float(dist[topic_idx])
                    })
        
        topic_obj['representative_docs'] = topic_docs
        topics.append(topic_obj)
    
    # Time-based topic distribution (how topics evolve over time)
    try:
        # Add topic assignments to the data
        filtered_data['dominant_topic'] = np.argmax(doc_topic_dists, axis=1)
        
        # Group by date and topic
        filtered_data['date'] = filtered_data['created_utc'].dt.date
        topic_evolution = {}
        
        # For each topic, get its frequency over time
        for topic_idx in range(n_topics):
            topic_docs = filtered_data[filtered_data['dominant_topic'] == topic_idx]
            if not topic_docs.empty:
                time_dist = topic_docs.groupby('date').size()
                topic_evolution[f'topic_{topic_idx}'] = {
                    str(date): int(count) for date, count in time_dist.items()
                }
        
        # Calculate overall coherence score
        coherence_score = sum(np.max(doc_topic_dists, axis=1)) / len(doc_topic_dists)
        
        return jsonify({
            'topics': topics,
            'topic_evolution': topic_evolution,
            'coherence_score': float(coherence_score),
            'n_docs_analyzed': len(filtered_data)
        })
    except Exception as e:
        # If time-based analysis fails, return just the topics
        return jsonify({
            'topics': topics,
            'error': str(e)
        })

@app.route('/api/coordinated', methods=['GET'])
def get_coordinated_behavior():

    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    time_window = int(request.args.get('time_window', 3600))  # Default to 1 hour in seconds
    similarity_threshold = float(request.args.get('similarity_threshold', 0.7))
    query = request.args.get('query', '')
    
    # Filter by query if specified
    filtered_data = data
    if query:
        filtered_data = data[
            data['selftext'].str.contains(query, case=False, na=False) |
            data['title'].str.contains(query, case=False, na=False)
        ]
    
    # Step 1: Sort data by timestamp
    sorted_data = filtered_data.sort_values('created_utc')
    
    # Step 2: Find posts with similar content in close time periods using improved similarity metrics
    coordinated_groups = []
    processed_indices = set()
    
    # Create a TF-IDF vectorizer for better similarity comparison
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        min_df=2,
        ngram_range=(1, 2)  # Include bigrams for better context
    )
    
    try:
        # Combine all available text for similarity analysis
        sorted_data['analysis_text'] = sorted_data['title']
        if 'selftext' in sorted_data.columns:
            sorted_data['analysis_text'] += ' ' + sorted_data['selftext'].fillna('')
        
        # Create matrix of TF-IDF features (might be sparse for large datasets)
        tfidf_matrix = tfidf_vectorizer.fit_transform(sorted_data['analysis_text'])
        
        # Enhanced coordinated group detection using vector similarity
        for i, row1 in sorted_data.iterrows():
            if i in processed_indices:
                continue
                
            group = [{
                'author': row1['author'],
                'id': row1.get('id', ''),
                'created_utc': row1['created_utc'].isoformat(),
                'title': row1['title'],
                'selftext': row1.get('selftext', '')[:200] + '...' if len(row1.get('selftext', '') or '') > 200 else row1.get('selftext', ''),
                'url': f"https://reddit.com/{row1.get('permalink', '')}"
            }]
            
            # Find posts within the time window
            time_limit = row1['created_utc'] + pd.Timedelta(seconds=time_window)
            window_posts = sorted_data[(sorted_data['created_utc'] <= time_limit) & 
                                      (sorted_data['created_utc'] >= row1['created_utc'])]
            
            # Find posts with similar content
            row1_vector = tfidf_matrix[sorted_data.index.get_loc(i)]
            
            for j, row2 in window_posts.iterrows():
                if i == j or j in processed_indices:
                    continue
                    
                # Calculate cosine similarity using TF-IDF vectors
                row2_vector = tfidf_matrix[sorted_data.index.get_loc(j)]
                similarity = cosine_similarity(row1_vector, row2_vector)[0][0]
                
                # Check for shared links, URLs or hashtags to improve detection
                shared_links = False
                shared_hashtags = False
                
                # Extract URLs and hashtags if available
                if 'selftext' in row1 and 'selftext' in row2:
                    # Simple regex to find URLs and hashtags (could be improved)
                    import re
                    urls1 = set(re.findall(r'https?://\S+', str(row1.get('selftext', ''))))
                    urls2 = set(re.findall(r'https?://\S+', str(row2.get('selftext', ''))))
                    hashtags1 = set(re.findall(r'#\w+', str(row1.get('selftext', ''))))
                    hashtags2 = set(re.findall(r'#\w+', str(row2.get('selftext', ''))))
                    
                    # Check for overlap
                    if urls1 and urls2 and urls1.intersection(urls2):
                        shared_links = True
                        similarity += 0.1  # Boost similarity score for shared links
                    
                    if hashtags1 and hashtags2 and hashtags1.intersection(hashtags2):
                        shared_hashtags = True
                        similarity += 0.1  # Boost similarity score for shared hashtags
                
                if similarity >= similarity_threshold:
                    group.append({
                        'author': row2['author'],
                        'id': row2.get('id', ''),
                        'created_utc': row2['created_utc'].isoformat(),
                        'title': row2['title'],
                        'selftext': row2.get('selftext', '')[:200] + '...' if len(row2.get('selftext', '') or '') > 200 else row2.get('selftext', ''),
                        'url': f"https://reddit.com/{row2.get('permalink', '')}",
                        'similarity_score': round(float(similarity), 3),
                        'shared_links': shared_links,
                        'shared_hashtags': shared_hashtags
                    })
                    processed_indices.add(j)
            
            if len(group) > 1:  # Only consider groups with at least 2 posts
                # Add metadata about the group
                group_metadata = {
                    'group_id': len(coordinated_groups),
                    'size': len(group),
                    'time_span': (max([pd.to_datetime(p['created_utc']) for p in group]) - 
                                 min([pd.to_datetime(p['created_utc']) for p in group])).total_seconds(),
                    'unique_authors': len(set([p['author'] for p in group])),
                    'shared_links_count': sum(1 for p in group if p.get('shared_links', False)),
                    'shared_hashtags_count': sum(1 for p in group if p.get('shared_hashtags', False)),
                    'posts': group
                }
                coordinated_groups.append(group_metadata)
                processed_indices.add(i)
    except Exception as e:
        # Fallback to simpler method if advanced method fails
        print(f"Advanced coordination detection failed: {str(e)}")
        # (Original simpler method would go here)
    
    # Step 3: Create network of coordinated authors
    author_links = []
    author_nodes = set()
    
    for group in coordinated_groups:
        authors = [post['author'] for post in group['posts']]
        author_nodes.update(authors)
        
        for i in range(len(authors)):
            for j in range(i+1, len(authors)):
                if authors[i] != authors[j]:  # Avoid self-loops
                    # Add weight based on frequency of coordination
                    author_links.append({
                        'source': authors[i], 
                        'target': authors[j],
                        'group_id': group['group_id'],
                        'weight': 1  # Could be enhanced to count multiple instances
                    })
    
    # Aggregate weights for duplicate links
    link_weights = defaultdict(int)
    for link in author_links:
        key = tuple(sorted([link['source'], link['target']]))
        link_weights[key] += link['weight']
    
    # Create final weighted links
    unique_links = [
        {'source': source, 'target': target, 'weight': weight}
        for (source, target), weight in link_weights.items()
    ]
    
    # Create nodes with metadata
    author_post_counts = filtered_data['author'].value_counts().to_dict()
    nodes = [
        {
            'id': author,
            'posts_count': author_post_counts.get(author, 0),
            'coordinated_groups_count': sum(1 for g in coordinated_groups if author in [p['author'] for p in g['posts']])
        }
        for author in author_nodes
    ]
    
    # Calculate network metrics
    network_metrics = {
        'total_groups': len(coordinated_groups),
        'total_authors': len(author_nodes),
        'total_connections': len(unique_links),
        'density': len(unique_links) / (len(author_nodes) * (len(author_nodes) - 1) / 2) if len(author_nodes) > 1 else 0,
        'avg_group_size': sum(g['size'] for g in coordinated_groups) / len(coordinated_groups) if coordinated_groups else 0,
        'authors_involved_percentage': len(author_nodes) / filtered_data['author'].nunique() * 100,
        'time_window_seconds': time_window,
        'similarity_threshold': similarity_threshold
    }
    
    return jsonify({
        'network': {'nodes': nodes, 'links': unique_links},
        'groups': coordinated_groups,
        'metrics': network_metrics
    })

def generate_structured_summary(text: str, query: str = None) -> str:
    """Generate a structured summary using NLTK-based extractive summarization."""
    try:
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        if not sentences:
            return "<div class='ai-summary-content'><p>No content to summarize.</p></div>"
        
        # Tokenize words and remove stopwords
        word_tokens = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            words = [lemmatizer.lemmatize(w) for w in words if w.isalnum() and w not in stop_words]
            word_tokens.extend(words)
        
        # Calculate word frequencies
        word_freq = Counter(word_tokens)
        
        # Score sentences based on word frequency and position
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = 0
            words = word_tokenize(sentence.lower())
            words = [lemmatizer.lemmatize(w) for w in words if w.isalnum() and w not in stop_words]
            
            # Word frequency score
            for word in words:
                score += word_freq[word]
            
            # Position score (favor sentences at the beginning)
            position_score = 1.0 / (i + 1)
            
            # Query relevance score if query is provided
            query_score = 0
            if query:
                query_words = set(word_tokenize(query.lower()))
                query_words = {lemmatizer.lemmatize(w) for w in query_words if w.isalnum() and w not in stop_words}
                common_words = query_words.intersection(set(words))
                query_score = len(common_words) * 2
            
            final_score = score + position_score + query_score
            sentence_scores.append((sentence, final_score))
        
        # Select top sentences
        num_sentences = min(5, len(sentences))
        top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]
        top_sentences = sorted(top_sentences, key=lambda x: sentences.index(x[0]))
        
        # Format the summary
        summary_html = "<div class='ai-summary-content'>"
        summary_html += "<h3>Key Points:</h3><ul>"
        
        for sentence, _ in top_sentences:
            # Clean and format the sentence
            clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
            if clean_sentence:
                summary_html += f"<li>{clean_sentence}</li>"
        
        summary_html += "</ul></div>"
        return summary_html
        
    except Exception as e:
        logging.error(f"Error in generate_structured_summary: {str(e)}")
        return "<div class='ai-summary-content'><p>Error generating summary.</p></div>"

def get_text_embeddings(texts):
    """Generate text embeddings using TF-IDF"""
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix

def semantic_search(query, texts, top_k=5):
    """Perform semantic search using TF-IDF"""
    # Get embeddings for all texts
    tfidf_matrix = get_text_embeddings(texts + [query])
    
    # Calculate cosine similarity
    query_vector = tfidf_matrix[-1]
    text_vectors = tfidf_matrix[:-1]
    
    similarities = (text_vectors * query_vector.T).toarray().flatten()
    
    # Get top k results
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [(texts[i], similarities[i]) for i in top_indices]

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    query = request.args.get('query', '')
    
    # Filter data based on query
    filtered_data = data[
        data['selftext'].str.contains(query, case=False, na=False) |
        data['title'].str.contains(query, case=False, na=False)
    ]
    
    # Calculate metrics
    metrics = {
        'total_posts': len(filtered_data),
        'unique_authors': filtered_data['author'].nunique(),
        'avg_comments': filtered_data['num_comments'].mean(),
        'time_span': (filtered_data['created_utc'].max() - filtered_data['created_utc'].min()).total_seconds() / (24 * 3600)
    }
    
    return jsonify(metrics)

@app.route('/api/ai_summary', methods=['GET'])
def get_ai_summary():
    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    query = request.args.get('query', '')
    
    # Filter data based on query
    filtered_data = data.copy()
    if query:
        filtered_data = filtered_data[
            filtered_data['selftext'].str.contains(query, case=False, na=False) |
            filtered_data['title'].str.contains(query, case=False, na=False)
        ]
    
    # Get top words and authors
    # Pass the query parameter to ensure filtering by ticker symbol
    top_words = get_common_words(query=query)
    top_authors = get_top_contributors(query=query)
    
    # Format the summary text
    # Calculate time span safely
    time_span = pd.Timedelta(filtered_data['created_utc'].max() - filtered_data['created_utc'].min())
    time_span_days = time_span.total_seconds() / (24 * 3600)
    
    summary_text = f"""
    Analysis Summary for query: {query}
    
    Total Posts: {len(filtered_data)}
    Unique Authors: {filtered_data['author'].nunique()}
    Average Comments: {filtered_data['num_comments'].mean():.2f}
    Time Span: {time_span_days:.1f} days
    
    Top Contributors:
    {', '.join([f"{author['author']} ({author['count']} posts)" for author in top_authors[:5]])}
    
    Most Common Words:
    {', '.join([f"{word['word']} ({word['count']})" for word in top_words[:10]])}
    """
    
    # Generate summary using the formatted text
    summary = generate_structured_summary(summary_text, query)
    
    return jsonify({
        'summary': summary,
        'metrics': {
        'total_posts': len(filtered_data),
        'unique_authors': filtered_data['author'].nunique(),
            'avg_comments': float(filtered_data['num_comments'].mean()),
            'time_span': float((filtered_data['created_utc'].max() - filtered_data['created_utc'].min()).total_seconds() / (24 * 3600)),
            'top_words': top_words,
            'top_authors': top_authors
        }
    })

@app.route('/api/common_words', methods=['GET'])
def get_common_words(query=None, limit=50):

    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    # If called as an API endpoint, get parameters from request
    if query is None:
        query = request.args.get('query', '')
        limit = int(request.args.get('limit', 50))
    
    # Filter data based on query
    filtered_data = data.copy()
    if query:
        filtered_data = filtered_data[
            filtered_data['selftext'].str.contains(query, case=False, na=False) |
            filtered_data['title'].str.contains(query, case=False, na=False)
        ]
    
    # Combine text data
    text_data = filtered_data['title'] + ' ' + filtered_data['selftext'].fillna('')
    
    # Tokenize and count words
    vectorizer = CountVectorizer(stop_words='english', max_features=limit)
    X = vectorizer.fit_transform(text_data)
    
    # Get word frequencies
    words = vectorizer.get_feature_names_out()
    freqs = X.sum(axis=0).A1
    
    # Sort by frequency
    sorted_indices = freqs.argsort()[::-1]
    result = [{'word': words[i], 'count': int(freqs[i])} for i in sorted_indices]
    
    # Return JSON if called as API endpoint, otherwise return the data
    if request.path == '/api/common_words':
        return jsonify(result)
    return result

@app.route('/api/dynamic_description', methods=['GET'])
def get_dynamic_description():
    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    if not has_groq or not GROQ_API_KEY:
        return jsonify({'error': 'GROQ API key not available'}), 400
    
    section = request.args.get('section', '')
    query = request.args.get('query', '')
    data_context = request.args.get('data_context', '{}')
    detail_level = request.args.get('detail_level', 'detailed')
    
    try:
        # Parse data context
        context_data = json.loads(data_context)
        
        # Prepare the prompt for GROQ
        prompt = f"""
        Generate a detailed description for a {section} visualization section.
        Query: {query}
        Data Context: {json.dumps(context_data)}
        Detail Level: {detail_level}
        
        The description should:
        1. Explain what the visualization shows
        2. Highlight key insights from the data
        3. Provide context about the analysis method
        4. Include relevant metrics and their interpretation
        
        Format the response in HTML with proper structure and styling.
        """
        
        # Make request to GROQ API
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "mixtral-8x7b-32768",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert data visualization analyst. Generate clear, concise, and informative descriptions of data visualizations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            description = result['choices'][0]['message']['content']
            return jsonify({'description': description})
        else:
            return jsonify({'error': 'Failed to generate description'}), 500
            
    except Exception as e:
        logging.error(f"Error in dynamic description generation: {str(e)}")
        return jsonify({'error': 'Error generating description'}), 500

@app.route('/api/semantic_map', methods=['GET'])
def get_semantic_map():

    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    query = request.args.get('query', '')
    max_points = min(int(request.args.get('max_points', 500)), 2000)  # Cap at 2000 for performance
    n_neighbors = int(request.args.get('n_neighbors', 15))
    min_dist = float(request.args.get('min_dist', 0.1))
    
    try:
        # Filter data based on query
        filtered_data = data
        if query:
            filtered_data = data[
                data['selftext'].str.contains(query, case=False, na=False) |
                data['title'].str.contains(query, case=False, na=False)
            ]
        
        if len(filtered_data) == 0:
            return jsonify({'error': 'No data found matching the query'}), 404
        
        # If we have too many posts, sample to improve performance
        if len(filtered_data) > max_points:
            filtered_data = filtered_data.sample(max_points, random_state=42)
        
        # Prepare text for embedding - combine title and selftext
        text_data = []
        for idx, row in filtered_data.iterrows():
            title = row['title'] if isinstance(row['title'], str) else ""
            selftext = row['selftext'] if isinstance(row['selftext'], str) and pd.notna(row['selftext']) else ""
            # Limit text length to avoid extremely long documents
            combined_text = (title + " " + selftext[:500]).strip()
            text_data.append(combined_text)
        
        # Load the sentence transformer model on demand
        global semantic_model
        if semantic_model is None:
            try:
                # Use a smaller, faster model for embeddings
                semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Loaded semantic embedding model successfully")
            except Exception as e:
                print(f"Error loading semantic model: {e}")
                return jsonify({'error': 'Failed to load semantic model'}), 500
        
        # Generate embeddings
        embeddings = semantic_model.encode(text_data, show_progress_bar=True)
        
        # Apply UMAP for dimensionality reduction
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric='cosine',
            random_state=42
        )
        
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        # Prepare result points with metadata
        points = []
        for i, (idx, row) in enumerate(filtered_data.iterrows()):
            points.append({
                'x': float(reduced_embeddings[i, 0]),
                'y': float(reduced_embeddings[i, 1]),
                'id': str(i),
                'title': row['title'],
                'author': row['author'],
                'subreddit': row.get('subreddit', ''),
                'created_utc': row['created_utc'].isoformat(),
                'num_comments': int(row.get('num_comments', 0)),
                'score': int(row.get('score', 0)),
                'preview_text': (row.get('selftext', '')[:100] + '...' if len(row.get('selftext', '') or '') > 100 
                                else row.get('selftext', ''))
            })
        
        # Try to extract topics from clusters of points
        topics = []
        try:
            from sklearn.cluster import KMeans
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Use K-means to find clusters in the embedding space
            n_clusters = min(10, max(3, len(points) // 50))  # Dynamic number of clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(reduced_embeddings)
            
            # Add cluster assignments to points
            for i, point in enumerate(points):
                point['cluster'] = int(clusters[i])
            
            # Extract keywords for each cluster
            for cluster_id in range(n_clusters):
                cluster_texts = [text_data[i] for i in range(len(text_data)) if clusters[i] == cluster_id]
                
                if cluster_texts:
                    # Use TF-IDF to find distinctive words for this cluster
                    tfidf = TfidfVectorizer(max_features=200, stop_words='english')
                    try:
                        cluster_tfidf = tfidf.fit_transform(cluster_texts)
                        
                        # Get top terms
                        feature_names = tfidf.get_feature_names_out()
                        top_indices = np.argsort(np.asarray(cluster_tfidf.mean(axis=0)).flatten())[-7:]
                        top_terms = [feature_names[i] for i in top_indices]
                        
                        # Calculate cluster center
                        cluster_points = reduced_embeddings[clusters == cluster_id]
                        center_x = float(np.mean(cluster_points[:, 0]))
                        center_y = float(np.mean(cluster_points[:, 1]))
                        
                        topics.append({
                            'id': int(cluster_id),
                            'terms': top_terms,
                            'size': int(np.sum(clusters == cluster_id)),
                            'center_x': center_x,
                            'center_y': center_y
                        })
                    except:
                        # Skip if TF-IDF fails for some reason
                        pass
        except Exception as e:
            print(f"Error extracting topics: {e}")
            # Continue without topics if clustering fails
        
        return jsonify({
            'points': points,
            'topics': topics,
            'total_posts': len(points),
            'umap_params': {
                'n_neighbors': n_neighbors,
                'min_dist': min_dist
            }
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Error generating semantic map: {str(e)}'}), 500

@app.route('/api/chatbot', methods=['POST'])
def chatbot_response():
    try:
        request_data = request.json
        query = request_data.get('query')
        history = request_data.get('history', [])
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
            
        # Load and filter relevant data based on query
        filtered_data = None
        global data  # Explicitly reference the global data variable
        if data is not None:
            # Use content_text instead of selftext for filtering
            if 'content_text' in data.columns:
                filtered_data = data[data['content_text'].str.contains(query, case=False, na=False) | 
                                  data['title'].str.contains(query, case=False, na=False)]
            else:
                # Fallback to selftext if content_text doesn't exist
                filtered_data = data[data['selftext'].str.contains(query, case=False, na=False) | 
                                  data['title'].str.contains(query, case=False, na=False)]
        
        # Prepare context from filtered data
        context = ''
        if filtered_data is not None and not filtered_data.empty:
            context += f"Based on analysis of {len(filtered_data)} relevant posts:\n"
            # Add key metrics
            context += f"- Total posts: {len(filtered_data)}\n"
            context += f"- Unique authors: {filtered_data['author'].nunique()}\n"
            context += f"- Average comments: {filtered_data['num_comments'].mean():.1f}\n"
            
            # Add time span
            if 'created_utc' in filtered_data.columns:
                time_span = (filtered_data['created_utc'].max() - filtered_data['created_utc'].min()).days
                context += f"- Time span: {time_span} days\n"
            
            # Add top keywords (using simple word frequency)
            # Use content_text instead of selftext if available
            content_column = 'content_text' if 'content_text' in filtered_data.columns else 'selftext'
            words = ' '.join(filtered_data[content_column].fillna('') + ' ' + filtered_data['title'].fillna('')).lower()
            words = re.findall(r'\b[a-z]{4,}\b', words)
            word_freq = Counter(words)
            # Remove common words
            for word in stop_words:
                word_freq.pop(word, None)
            top_keywords = [word for word, _ in word_freq.most_common(5)]
            if top_keywords:
                context += f"- Top keywords: {', '.join(top_keywords)}\n"
        
        if not has_gemini or not GEMINI_API_KEY:
            return jsonify({'error': 'Gemini API key not available'}), 400

        # Configure Gemini
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Prepare chat messages
        messages = [{
            "role": "system",
            "content": f"You are a helpful AI assistant analyzing social media data. Use the following context to answer questions:\n{context}"
        }]
        
        # Add chat history
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add user's current query
        messages.append({
            "role": "user",
            "content": query
        })
        
        # Generate response using Gemini
        chat = model.start_chat(history=[])
        response = chat.send_message(f"""
        Context: {context}
        
        User Query: {query}
        
        Please provide a detailed, well-structured response that:
        1. Directly addresses the user's query
        2. Uses data from the context when relevant
        3. Organizes information with appropriate headings and bullet points
        4. Maintains a professional and analytical tone
        5. Focuses on factual insights from the data
        
        Format the response in clean HTML with appropriate tags for structure.
        """)
        
        # Format the response with proper HTML structure
        formatted_response = f"<div class='chatbot-response'>{response.text}</div>"
        
        return jsonify({
            'response': formatted_response
        })
        
    except Exception as e:
        logging.error(f"Error in chatbot response: {str(e)}")
        return jsonify({
            'error': 'An error occurred while processing your request',
            'details': str(e)
        }), 500
        
        # Create the prompt
        prompt = f"""
        Context: {context_str}
        
        User Message: {user_message}
        
        Please provide a helpful response that:
        1. Addresses the user's question directly
        2. Uses the provided context to inform the response
        3. Maintains a professional and informative tone
        4. Includes relevant data insights when available
        """
        
        # Generate response
        response = model.generate_content(prompt)
        
        # Format the response
        formatted_response = {
            'response': response.text,
            'status': 'success'
        }
        
        return jsonify(formatted_response)
        
    except Exception as e:
        logging.error(f"Error in chatbot response: {str(e)}")
        return jsonify({'error': 'Error generating response'}), 500

@app.route('/api/events', methods=['GET'])
def get_historical_events():

    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    query = request.args.get('query', '')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    try:
        # Find events related to the query
        # First, look for exact matches in our event database
        matching_events = []
        
        # Try to find the most relevant event category based on the query
        best_match = None
        highest_match_score = 0
        
        for event_category in event_database.keys():
            # Calculate simple term overlap for matching
            category_terms = set(event_category.lower().split())
            query_terms = set(query.lower().split())
            common_terms = category_terms.intersection(query_terms)
            
            # Score based on proportion of matching terms
            if len(category_terms) > 0:
                match_score = len(common_terms) / len(category_terms)
                
                # Check if this is the best match so far
                if match_score > highest_match_score:
                    highest_match_score = match_score
                    best_match = event_category
        
        # Use best matching category if score is above threshold
        if highest_match_score >= 0.3 and best_match:
            matching_events = event_database[best_match]
        
        # If no matches, try to look for partial matches or use Groq to generate events
        if not matching_events and has_groq and GROQ_API_KEY:
            # Use Groq API to generate potential events
            groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Construct a prompt to generate historical events
            prompt = f"""
            I need a list of 3-5 major historical events related to "{query}".
            
            For each event, provide:
            1. The date (in YYYY-MM-DD format)
            2. A short title (5-7 words)
            3. A brief description (15-20 words)
            4. A reliable source
            
            Format the response as a JSON list of objects with fields: date, title, description, source.
            Do not include explanations or any text outside the JSON structure.
            """
            
            payload = {
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that provides accurate historical information."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 500
            }
            
            try:
                response = requests.post(groq_api_url, headers=headers, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    events_text = result["choices"][0]["message"]["content"].strip()
                    
                    # Extract JSON from response (it might be wrapped in markdown code blocks)
                    import re
                    json_match = re.search(r'```json(.*?)```', events_text, re.DOTALL)
                    if json_match:
                        events_text = json_match.group(1).strip()
                    
                    # Clean up any remaining markdown or text
                    events_text = re.sub(r'```.*?```', '', events_text, flags=re.DOTALL)
                    events_text = events_text.strip()
                    
                    # Try to parse the JSON
                    try:
                        import json
                        generated_events = json.loads(events_text)
                        matching_events = generated_events
                    except json.JSONDecodeError:
                        print(f"Error parsing generated events: {events_text}")
            except Exception as e:
                print(f"Error generating events with Groq: {e}")
        
        # If still no events, return an informative message
        if not matching_events:
            return jsonify({
                'events': [],
                'message': f"No historical events found for '{query}'. Try a more specific query related to major news events."
            })
        
        # Filter events by date range if provided
        if start_date and end_date:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            matching_events = [
                event for event in matching_events 
                if start <= pd.to_datetime(event['date']) <= end
            ]
        
        # If there are no events in the specified range, return appropriate message
        if not matching_events:
            return jsonify({
                'events': [],
                'message': f"No events found for '{query}' in the specified date range. Try expanding the date range or using a different query."
            })
        
        # Correlate events with social media activity
        # Filter data based on query for time series
        filtered_data = data[
            data['selftext'].str.contains(query, case=False, na=False) |
            data['title'].str.contains(query, case=False, na=False)
        ]
        
        # Group by date to get post counts
        filtered_data['date'] = filtered_data['created_utc'].dt.date
        post_counts = filtered_data.groupby('date').size()
        
        # Calculate rolling average for smoothing (7-day window)
        rolling_avg = post_counts.rolling(window=7, min_periods=1).mean()
        
        # Calculate standard deviation for detecting peaks
        std_dev = post_counts.std()
        mean_posts = post_counts.mean()
        
        # Find peaks (days with significantly higher activity)
        peak_dates = post_counts[post_counts > (mean_posts + 1.5 * std_dev)].index
        
        # Map events to their nearby activity and identify correlations
        correlated_events = []
        for event in matching_events:
            event_date = pd.to_datetime(event['date']).date()
            
            # Check if event data is in the post counts index
            if event_date in post_counts.index:
                posts_on_day = int(post_counts.loc[event_date])
                rolling_avg_on_day = float(rolling_avg.loc[event_date])
            else:
                # Find the closest date in the data
                closest_dates = post_counts.index.astype('datetime64[ns]').astype(object)
                closest_dates = [date for date in closest_dates]
                
                if not closest_dates:
                    # No post data available for comparison
                    posts_on_day = 0
                    rolling_avg_on_day = 0
                else:
                    # Find the date closest to the event date
                    closest_date = min(closest_dates, key=lambda x: abs(x - event_date))
                    posts_on_day = int(post_counts.loc[closest_date])
                    rolling_avg_on_day = float(rolling_avg.loc[closest_date])
            
            # Calculate days to nearest peak
            if len(peak_dates) > 0:
                days_to_nearest_peak = min([abs((event_date - peak_date).days) for peak_date in peak_dates])
            else:
                days_to_nearest_peak = None
            
            # Determine correlation type
            if days_to_nearest_peak is not None and days_to_nearest_peak <= 2:
                correlation = "strong"  # Event coincides with peak
            elif days_to_nearest_peak is not None and days_to_nearest_peak <= 7:
                correlation = "moderate"  # Event is close to peak
            else:
                correlation = "weak"  # No clear correlation
            
            # Calculate relative activity compared to average (how many times above average)
            if rolling_avg_on_day > 0:
                activity_ratio = posts_on_day / rolling_avg_on_day
            else:
                activity_ratio = 0
            
            # Add correlation data to the event
            correlated_event = {
                **event,  # Keep original event data
                'posts_on_day': posts_on_day,
                'average_posts': round(rolling_avg_on_day, 2),
                'activity_ratio': round(activity_ratio, 2),
                'correlation': correlation,
                'days_to_nearest_peak': days_to_nearest_peak
            }
            
            # Add user-friendly insights about correlation
            if correlation == "strong":
                correlated_event['insight'] = f"This event coincides with a significant spike in online discussions, with {posts_on_day} posts (about {round(activity_ratio, 1)}x the average)."
            elif correlation == "moderate":
                correlated_event['insight'] = f"This event is temporally close to increased online activity, with {posts_on_day} posts around this time."
            else:
                correlated_event['insight'] = f"This event didn't correspond with unusual online activity, with {posts_on_day} posts on this day."
            
            correlated_events.append(correlated_event)
        
        # Create a time series summary for context
        time_series_data = []
        for date, count in post_counts.items():
            time_series_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'count': int(count),
                'is_peak': date in peak_dates
            })
        
        # Return the correlated events with context
        return jsonify({
            'events': correlated_events,
            'time_series': time_series_data,
            'query': query,
            'total_posts': len(filtered_data),
            'peak_dates': [date.strftime('%Y-%m-%d') for date in peak_dates],
            'event_category': best_match if highest_match_score >= 0.3 else query
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Error retrieving events: {str(e)}'}), 500

@app.route('/api/semantic_search', methods=['GET'])
def semantic_search_endpoint():
    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    query = request.args.get('query', '')
    limit = int(request.args.get('limit', 5))
    
    # Get texts to search
    texts = data['title'] + ' ' + data['selftext'].fillna('')
    
    # Perform semantic search
    results = semantic_search(query, texts.tolist(), top_k=limit)
    
    # Format results
    formatted_results = []
    for text, score in results:
        formatted_results.append({
            'text': text,
            'similarity_score': float(score)
        })
    
    return jsonify(formatted_results)

@app.route('/api/semantic_query', methods=['GET'])
def semantic_query():

    global platform_manager, data, semantic_model
    
    # Check if data is available
    if (platform_manager.integrated_data is None) and (data is None):
        return jsonify({'error': 'No data loaded'}), 400
    
    query = request.args.get('query', '')
    platform = request.args.get('platform', None)
    max_results = min(int(request.args.get('max_results', 20)), 100)  # Limit to max 100 for performance
    min_similarity = float(request.args.get('min_similarity', 0.5))  # Minimum similarity threshold
    
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400
    
    try:
        # Use platform manager if available
        if platform_manager.integrated_data is not None:
            # Get data from specific platform or all platforms
            all_data = platform_manager.get_platform_data(platform)
            
            # Use the normalized column names
            content_column = 'content_text'
            title_column = 'title'
        else:
            # Fallback to global data variable (backward compatibility)
            all_data = data
            content_column = 'selftext'
            title_column = 'title'
        
        # To avoid processing too many posts, take a reasonable sample
        sample_size = 1000  # Sample size for semantic processing
        if len(all_data) > sample_size:
            # Try to get a balanced sample across platforms
            if 'platform' in all_data.columns and platform is None:
                # Try to get a balanced sample across platforms
                platforms = all_data['platform'].unique()
                per_platform = max(50, sample_size // len(platforms))
                sample_frames = []
                
                for p in platforms:
                    platform_data = all_data[all_data['platform'] == p]
                    if len(platform_data) > per_platform:
                        platform_sample = platform_data.sample(per_platform, random_state=42)
                        sample_frames.append(platform_sample)
                    else:
                        sample_frames.append(platform_data)
                
                sampled_data = pd.concat(sample_frames)
            else:
                # Simple random sample
                sampled_data = all_data.sample(sample_size, random_state=42)
        else:
            sampled_data = all_data
        
        # Load the sentence transformer model on demand
        if semantic_model is None:
            try:
                semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Loaded semantic embedding model successfully")
            except Exception as e:
                print(f"Error loading semantic model: {e}")
                return jsonify({'error': 'Failed to load semantic model'}), 500
        
        # Generate embedding for the query
        query_embedding = semantic_model.encode(query, convert_to_tensor=True)
        
        # Prepare texts from posts for embedding
        post_texts = []
        post_indices = []
        
        for idx, post in sampled_data.iterrows():
            # Combine title and content for better semantic matching
            title = post.get(title_column, '')
            content = post.get(content_column, '')
            combined = f"{title} {content[:500]}"  # Limit length for efficiency
            post_texts.append(combined)
            post_indices.append(idx)
        
        # Generate embeddings for all posts
        post_embeddings = semantic_model.encode(post_texts, convert_to_tensor=True)
        
        # Compute semantic similarity
        from torch.nn import functional as F
        similarities = F.cosine_similarity(query_embedding.unsqueeze(0), post_embeddings).cpu().numpy()
        
        # Create a list of (index, similarity) pairs
        ranked_indices = [(post_indices[i], similarities[i]) for i in range(len(similarities))]
        
        # Filter by minimum similarity threshold and sort
        ranked_indices = [pair for pair in ranked_indices if pair[1] >= min_similarity]
        ranked_indices.sort(key=lambda x: x[1], reverse=True)
        
        # Format the top results
        results = []
        for idx, similarity in ranked_indices[:max_results]:
            post = all_data.loc[idx]
            result = {
                'title': post.get(title_column, ''),
                'content': post.get(content_column, '')[:300] + '...' if len(post.get(content_column, '') or '') > 300 else post.get(content_column, ''),
                'author': post.get('author', ''),
                'created_at': post.get('created_at', post.get('created_utc', '')).isoformat(),
                'platform': post.get('platform', 'reddit'),
                'similarity': float(similarity)
            }
            if 'community' in post:
                result['community'] = post['community']
            elif 'subreddit' in post:
                result['community'] = post['subreddit']
            
            results.append(result)
        
        # Extract key terms from the query for explanation
        keywords = []
        try:
            # Simple extraction of nouns and important words
            import re
            words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
            stop_words = {'the', 'and', 'for', 'with', 'about', 'what', 'how', 'when', 'who', 'why', 'where', 'which'}
            keywords = [word for word in words if word not in stop_words][:5]  # Top 5 keywords
        except:
            pass
        
        return jsonify({
            'results': results,
            'total_results': len(results),
            'sample_size': len(sampled_data),
            'query': query,
            'key_terms': keywords,
            'message': f"Found {len(results)} semantically relevant posts matching your query"
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Error during semantic query: {str(e)}'}), 500

if __name__ == '__main__':
    # Load dataset on startup
    load_success = load_dataset()
    if not load_success:
        print("WARNING: Failed to load dataset. Make sure data file exists at ./data/data.jsonl")
        # Create data directory if it doesn't exist
        os.makedirs("./data", exist_ok=True)
        print("Created data directory. Please place your data.jsonl file in the ./data folder.")
    
    app.run(debug=True)