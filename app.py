#!/usr/bin/env python3
"""
Streamlit Client for Product Recommendation System (AWS Bedrock Version)

A web-based interface for uploading sales data and chatting with the AI recommendation agent.
Modified to use AWS Bedrock instead of Ollama for LLM capabilities.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime
import io
import duckdb
import re
import os

# Import AWS Bedrock
import boto3
import json
from botocore.exceptions import ClientError


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit-recommender")

# Page configuration
st.set_page_config(
    page_title="AI Product Recommender (Bedrock)",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state FIRST (before any access)
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "data_loaded": False,
        "df": None,
        "user_item_matrix": None,
        "user_sim": None,
        "user_encoder": None,
        "product_encoder": None,
        "df_processed": None,
        "chat_history": [],
        "bedrock_region": "ap-northeast-1",
        "bedrock_model": "openai.gpt-oss-20b-1:0",
        "bedrock_profile": "default",
        "bedrock_credentials_path": "./bedrock/.aws/credentials",
        "bedrock_connected": False,
        "duckdb_conn": None,
        "agent": None,
        "last_uploaded_file": None,
        "processing_message": False,
        "bedrock_connection_tested": False,
        "bedrock_client": None,
        "rfm_data": None,
        "rfm_calculated": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Enhanced Custom CSS
st.markdown("""
<style>
    
    /* Hide footer but keep menu visible */
    #MainMenu {visibility: visible;}
    footer {visibility: hidden;}
    
    
    /* Hide deploy button specifically */
    .stDeployButton {display: none;}
    
    /* Show hamburger menu button */
    button[kind="header"] {display: block !important;}
    
    /* Light blue color for Load Sample Data button */
    [data-testid="stSidebar"] button[kind="primary"] {
        background-color: #4A90E2 !important;
        border-color: #4A90E2 !important;
        color: white !important;
    }
    
    [data-testid="stSidebar"] button[kind="primary"]:hover {
        background-color: #357ABD !important;
        border-color: #357ABD !important;
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.4) !important;
    }
    
    [data-testid="stSidebar"] button[kind="primary"]:active {
        background-color: #2868A8 !important;
        border-color: #2868A8 !important;
    }
    
    /* Keep toolbar but hide specific elements */
    header[data-testid="stHeader"] {
        background: transparent;
        height: 2.5rem;
    }
    
    /* Make sure sidebar is visible and toggle works */
    [data-testid="stSidebar"] {
        display: block !important;
    }
    
    /* Ensure sidebar collapse button is visible */
    [data-testid="collapsedControl"] {
        display: block !important;
        visibility: visible !important;
        position: fixed !important;
        left: 0 !important;
        top: 0.5rem !important;
        z-index: 999999 !important;
        background: #667eea !important;
        color: white !important;
        padding: 0.5rem !important;
        border-radius: 0 0.5rem 0.5rem 0 !important;
    }
    
    /* Reduce top padding since header is now hidden */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
    }
    
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        padding: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: inherit;
    }
    
    /* Light mode specific */
    @media (prefers-color-scheme: light) {
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 0.8rem 0;
        animation: fadeIn 0.3s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        margin-left: 0rem;
    }
    .assistant-message {
        background-color: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        margin-right: 2rem;
        color: inherit;
    }
    
    /* Dark mode support - auto mode */
    @media (prefers-color-scheme: dark) {
        .assistant-message {
            background-color: rgba(102, 126, 234, 0.15);
        }
        .metric-card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        }
        .recommendation-card {
            background: rgba(255, 255, 255, 0.05);
            color: inherit;
        }
    }
    
    /* Dark mode forced styles */
    .stApp[data-theme="dark"] .assistant-message,
    :root[style*="color-scheme: dark"] .assistant-message {
        background-color: rgba(102, 126, 234, 0.15);
    }
    
    .stApp[data-theme="dark"] .metric-card,
    :root[style*="color-scheme: dark"] .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
    }
    
    .stApp[data-theme="dark"] .recommendation-card,
    :root[style*="color-scheme: dark"] .recommendation-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.25) 0%, rgba(118, 75, 162, 0.25) 100%);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
        border-left-color: #c4b5fd;
    }
    
    .stApp[data-theme="dark"] .recommendation-card:hover,
    :root[style*="color-scheme: dark"] .recommendation-card:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.35) 0%, rgba(118, 75, 162, 0.35) 100%);
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.5);
    }
    
    .stApp[data-theme="dark"] .recommendation-card strong,
    :root[style*="color-scheme: dark"] .recommendation-card strong {
        color: #ffffff !important;
        font-weight: 700;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }
    
    .stApp[data-theme="dark"] .recommendation-card small,
    :root[style*="color-scheme: dark"] .recommendation-card small {
        color: #e8e8ff !important;
        opacity: 1;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    /* Light mode forced styles */
    .stApp[data-theme="light"] .recommendation-card,
    :root[style*="color-scheme: light"] .recommendation-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.12) 0%, rgba(118, 75, 162, 0.12) 100%);
        border-left-color: #667eea;
    }
    
    .stApp[data-theme="light"] .recommendation-card strong,
    :root[style*="color-scheme: light"] .recommendation-card strong {
        color: #2c3e50;
    }
    
    .stApp[data-theme="light"] .recommendation-card small,
    :root[style*="color-scheme: light"] .recommendation-card small {
        color: #5a6c7d;
    }
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-success {
        background-color: rgba(40, 167, 69, 0.2);
        color: #28a745;
    }
    .status-warning {
        background-color: rgba(255, 193, 7, 0.2);
        color: #ffc107;
    }
    .status-error {
        background-color: rgba(220, 53, 69, 0.2);
        color: #dc3545;
    }
    
    /* Dark mode adjustments for badges - auto mode */
    @media (prefers-color-scheme: dark) {
        .status-success {
            background-color: rgba(40, 167, 69, 0.3);
            color: #5cd67e;
        }
        .status-warning {
            background-color: rgba(255, 193, 7, 0.3);
            color: #ffd96a;
        }
        .status-error {
            background-color: rgba(220, 53, 69, 0.3);
            color: #f58b9a;
        }
    }
    
    /* Dark mode forced - badges */
    .stApp[data-theme="dark"] .status-success,
    :root[style*="color-scheme: dark"] .status-success {
        background-color: rgba(40, 167, 69, 0.3);
        color: #5cd67e;
    }
    
    .stApp[data-theme="dark"] .status-warning,
    :root[style*="color-scheme: dark"] .status-warning {
        background-color: rgba(255, 193, 7, 0.3);
        color: #ffd96a;
    }
    
    .stApp[data-theme="dark"] .status-error,
    :root[style*="color-scheme: dark"] .status-error {
        background-color: rgba(220, 53, 69, 0.3);
        color: #f58b9a;
    }
    .recommendation-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
        padding: 0.6rem 0.85rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 0.3rem 0;
        box-shadow: 0 2px 6px rgba(102, 126, 234, 0.15);
        font-size: 0.9rem;
        line-height: 1.4;
        color: inherit;
        transition: all 0.2s ease;
    }
    
    .recommendation-card:hover {
        transform: translateX(4px);
        box-shadow: 0 3px 8px rgba(102, 126, 234, 0.25);
    }
    
    .recommendation-card strong {
        font-size: 0.95rem;
        color: inherit;
        font-weight: 600;
    }
    
    .recommendation-card small {
        font-size: 0.8rem;
        color: inherit;
        opacity: 0.75;
    }
    
    /* Light mode specific */
    @media (prefers-color-scheme: light) {
        .recommendation-card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.12) 0%, rgba(118, 75, 162, 0.12) 100%);
            border-left-color: #667eea;
        }
        .recommendation-card strong {
            color: #2c3e50;
        }
        .recommendation-card small {
            color: #5a6c7d;
        }
    }
    
    /* Dark mode for recommendation cards - auto mode */
    @media (prefers-color-scheme: dark) {
        .recommendation-card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.25) 0%, rgba(118, 75, 162, 0.25) 100%);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
            border-left-color: #c4b5fd;
        }
        .recommendation-card:hover {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.35) 0%, rgba(118, 75, 162, 0.35) 100%);
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.5);
        }
        .recommendation-card strong {
            color: #ffffff !important;
            font-weight: 700;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
        }
        .recommendation-card small {
            color: #e8e8ff !important;
            opacity: 1;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
        }
    }
    
    .score-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 0.25rem 0.6rem;
        border-radius: 0.5rem;
        font-weight: bold;
        font-size: 0.75rem;
        box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3);
    }
    
    /* Left pane/column styling */
    [data-testid="column"] {
        padding: 0.5rem;
    }
    
    /* Tab styling for better dark/light mode support */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: inherit;
    }
    
    /* Dataframe styling for dark/light modes */
    .stDataFrame {
        color: inherit;
    }
    
    /* Light mode dataframe */
    @media (prefers-color-scheme: light) {
        .stDataFrame {
            background: rgba(255, 255, 255, 0.9);
        }
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(0, 0, 0, 0.1);
        }
    }
    
    /* Dark mode dataframe */
    @media (prefers-color-scheme: dark) {
        .stDataFrame {
            background: rgba(255, 255, 255, 0.05);
        }
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        /* Dataframe text color */
        .stDataFrame table {
            color: #fafafa !important;
        }
    }
    
    /* Reduce tab margins */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        padding-top: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 0.5rem;
    }
    
    /* Reduce header spacing */
    h1, h2, h3 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Add padding to columns to prevent text cutoff */
    [data-testid="column"] > div {
        padding-top: 0.5rem;
    }
    
    /* Ensure info boxes have proper padding */
    .stAlert {
        padding: 0.75rem 1rem !important;
    }
</style>
""", unsafe_allow_html=True)


def copy_to_clipboard(text):
    """Create a button that copies text to clipboard using JavaScript."""
    # Escape special characters for JavaScript
    escaped_text = text.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'").replace('\n', '\\n')
    
    # Create unique button ID
    button_id = f"copy_btn_{hash(text) % 10000}"
    
    html_code = f"""
    <div style="margin: 10px 0;">
        <button id="{button_id}" 
                style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                       color: white;
                       border: none;
                       padding: 10px 20px;
                       border-radius: 5px;
                       cursor: pointer;
                       font-size: 14px;
                       font-weight: 500;
                       display: flex;
                       align-items: center;
                       gap: 8px;">
            📋 Click to Copy Message
        </button>
        <p id="status_{button_id}" style="margin-top: 8px; font-size: 12px; color: #666;"></p>
    </div>
    <script>
        document.getElementById('{button_id}').addEventListener('click', function() {{
            const text = "{escaped_text}";
            
            // Try modern clipboard API
            if (navigator.clipboard && navigator.clipboard.writeText) {{
                navigator.clipboard.writeText(text).then(function() {{
                    document.getElementById('status_{button_id}').innerHTML = '✅ Copied to clipboard!';
                    document.getElementById('status_{button_id}').style.color = '#28a745';
                    setTimeout(function() {{
                        document.getElementById('status_{button_id}').innerHTML = '';
                    }}, 2000);
                }}).catch(function(err) {{
                    document.getElementById('status_{button_id}').innerHTML = '❌ Failed to copy';
                    document.getElementById('status_{button_id}').style.color = '#dc3545';
                }});
            }} else {{
                // Fallback for older browsers
                const textarea = document.createElement('textarea');
                textarea.value = text;
                textarea.style.position = 'fixed';
                textarea.style.opacity = '0';
                document.body.appendChild(textarea);
                textarea.select();
                try {{
                    document.execCommand('copy');
                    document.getElementById('status_{button_id}').innerHTML = '✅ Copied to clipboard!';
                    document.getElementById('status_{button_id}').style.color = '#28a745';
                    setTimeout(function() {{
                        document.getElementById('status_{button_id}').innerHTML = '';
                    }}, 2000);
                }} catch (err) {{
                    document.getElementById('status_{button_id}').innerHTML = '❌ Failed to copy';
                    document.getElementById('status_{button_id}').style.color = '#dc3545';
                }}
                document.body.removeChild(textarea);
            }}
        }});
    </script>
    """
    
    components.html(html_code, height=80)


def initialize_bedrock_client():
    """Initialize AWS Bedrock client with credentials."""
    try:
        # Set custom credentials path
        if os.path.exists(st.session_state.bedrock_credentials_path):
            os.environ["AWS_SHARED_CREDENTIALS_FILE"] = st.session_state.bedrock_credentials_path
        
        # Initialize session and client
        session = boto3.Session(profile_name=st.session_state.bedrock_profile)
        client = session.client("bedrock-runtime", region_name=st.session_state.bedrock_region)
        
        st.session_state.bedrock_client = client
        logger.info("Bedrock client initialized successfully")
        return client, None
    except Exception as e:
        logger.error(f"Error initializing Bedrock client: {e}")
        return None, str(e)


def test_bedrock_connection():
    """Test connection to AWS Bedrock with detailed feedback."""
    try:
        client, error = initialize_bedrock_client()
        if error:
            return False, error
        
        # Try a simple test call
        test_payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ]
        }
        
        response = client.invoke_model(
            modelId=st.session_state.bedrock_model,
            body=json.dumps(test_payload),
            contentType="application/json"
        )
        
        # If we get here, connection is successful
        st.session_state.bedrock_client = client
        return True, None
        
    except ClientError as e:
        error_msg = f"AWS Error: {e.response['Error']['Code']} - {e.response['Error']['Message']}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Connection test error: {e}")
        return False, error_msg


def load_and_process_data(df):
    """Load and process the retail data with enhanced validation and feedback."""
    try:
        # Validate required columns
        required_cols = ['CustomerID', 'StockCode', 'Quantity', 'Description']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.info("💡 **Tip:** Make sure your CSV/Excel has these columns:\n" + 
                   "\n".join([f"- {col}" for col in required_cols]))
            return False
        
        # Data cleaning with progress feedback
        with st.spinner("🧹 Cleaning data..."):
            original_len = len(df)
            
            # Remove null values
            df = df.dropna(subset=['CustomerID', 'StockCode', 'Quantity'])
            after_null = len(df)
            
            # Remove returns/refunds
            df = df[df['Quantity'] > 0]
            cleaned_len = len(df)
            
            if cleaned_len == 0:
                st.error("❌ No valid data after cleaning!")
                return False
            
            # Show cleaning summary
            st.info(f"🧹 **Data Cleaning Summary:**\n"
                   f"- Removed {original_len - after_null:,} rows with null values\n"
                   f"- Removed {after_null - cleaned_len:,} returns/refunds\n"
                   f"- **{cleaned_len:,} valid transactions remaining**")
        
        # Build recommendation model
        with st.spinner("🤖 Building recommendation model..."):
            user_encoder = LabelEncoder()
            product_encoder = LabelEncoder()
            
            df_copy = df.copy()
            df_copy["user_idx"] = user_encoder.fit_transform(df_copy["CustomerID"])
            df_copy["product_idx"] = product_encoder.fit_transform(df_copy["StockCode"])
            
            n_users = df_copy["user_idx"].nunique()
            n_products = df_copy["product_idx"].nunique()
            
            # Build User-Item matrix
            aggregated = df_copy.groupby(['user_idx', 'product_idx'])['Quantity'].sum().reset_index()
            user_item_matrix = aggregated.pivot(
                index='user_idx', 
                columns='product_idx', 
                values='Quantity'
            ).fillna(0).values
            
            # Compute user similarity
            user_sim = cosine_similarity(user_item_matrix)
            
            sparsity = 1 - (np.count_nonzero(user_item_matrix) / (n_users * n_products))
        
        # Store in session state
        st.session_state.df = df
        st.session_state.user_item_matrix = user_item_matrix
        st.session_state.user_sim = user_sim
        st.session_state.user_encoder = user_encoder
        st.session_state.product_encoder = product_encoder
        st.session_state.df_processed = df_copy
        st.session_state.data_loaded = True
        
        # Reset RFM cache for new data
        st.session_state.rfm_data = None
        st.session_state.rfm_calculated = False
        
        # Initialize DuckDB connection
        with st.spinner("💾 Initializing database..."):
            try:
                conn = duckdb.connect(database='sales_data.duckdb')
                conn.execute("DROP TABLE IF EXISTS sales_data")
                conn.execute("CREATE TABLE sales_data AS SELECT * FROM df")
                st.session_state.duckdb_conn = conn
                
                row_count = conn.execute("SELECT COUNT(*) FROM sales_data").fetchone()[0]
                logger.info(f"DuckDB initialized with {row_count} rows")
            except Exception as e:
                logger.error(f"Error initializing DuckDB: {e}")
                st.error(f"⚠️ Database initialization failed: {e}")
        
        # Success - data loaded silently
        
        return True
        
    except Exception as e:
        st.error(f"❌ **Error loading data:** {e}")
        logger.error(f"Error loading data: {e}", exc_info=True)
        return False


def generate_personalized_message(customer_id, history_df, rec_df):
    """Generate a personalized recommendation message using AWS Bedrock."""
    try:
        # Prepare history summary
        history_items = history_df.head(10)['Description'].tolist()
        history_summary = "\n".join([f"- {item}" for item in history_items])
        
        if len(history_df) > 10:
            history_summary += f"\n- ...and {len(history_df) - 10} more items"
        
        # Create the prompt
        prompt = f"""You are a friendly shopping assistant. Based on this customer's purchase history and recommended products, write a warm, personalized 3-4 sentence recommendation message.

**Customer ID:** {int(customer_id)}

**Previous Purchases:**
{history_summary}

**Recommended Products:**
{rec_df[["Description", "Score"]].head(5).to_string(index=False)}

Write a friendly 3-4 sentence message that:
1. Acknowledges their past purchases
2. Explain why these new products would be great for them
3. Sounds natural and conversational (not robotic)
4. Creates excitement about the recommendations

CRITICAL INSTRUCTIONS:
- Do NOT include ANY reasoning, thinking process, explanations, or meta-commentary
- Do NOT use tags like <reasoning>, <thinking>, or any XML-style tags
- Do NOT include phrases like "Here's the message:", "Reasoning:", or "Let me think"
- ONLY provide the final personalized message directly
- Start with a greeting like "Hi there!" or "Hello!" and write the message naturally
- Your entire response should be ONLY the customer-facing message, nothing else"""
        
        # Get or initialize Bedrock client
        if st.session_state.bedrock_client is None:
            client, error = initialize_bedrock_client()
            if error:
                return f"❌ **Bedrock Connection Error:** {error}"
        else:
            client = st.session_state.bedrock_client
        
        # Call Bedrock API
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful shopping assistant."},
                {"role": "user", "content": prompt}
            ]
        }
        
        response = client.invoke_model(
            modelId=st.session_state.bedrock_model,
            body=json.dumps(payload),
            contentType="application/json"
        )
        
        # Parse Bedrock response
        body = json.loads(response["body"].read().decode("utf-8"))
        
        # Extract the assistant reply
        answer = body["choices"][0]["message"]["content"]
        
        # Remove reasoning sections if present
        answer = answer.strip()
        
        # Remove XML-style reasoning tags: <reasoning>...</reasoning>
        answer = re.sub(r'<reasoning>.*?</reasoning>', '', answer, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove any remaining reasoning patterns
        # Pattern 1: "Reasoning: ... \n\n Actual message"
        if "reasoning:" in answer.lower():
            parts = re.split(r'reasoning:\s*', answer, flags=re.IGNORECASE)
            if len(parts) > 1:
                # Look for the actual message after reasoning
                remaining = parts[-1]
                # Find where the actual message starts (after reasoning content)
                lines = remaining.split('\n')
                message_lines = []
                in_message = False
                for line in lines:
                    line_lower = line.lower().strip()
                    # Detect start of actual message
                    if not in_message and (
                        line_lower.startswith(('dear', 'hi ', 'hello', 'we', 'thank', 'based on', 'i see', 'you'))
                        or len(line.strip()) > 50
                    ):
                        in_message = True
                    if in_message:
                        message_lines.append(line)
                if message_lines:
                    answer = '\n'.join(message_lines).strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "Here's the message:",
            "Here is the message:",
            "Here's a personalized message:",
            "Here is a personalized message:",
            "Message:",
            "Personalized message:",
            "Final message:"
        ]
        
        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        
        # Clean up extra whitespace
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        answer = answer.strip()
        
        return answer
    
    except ClientError as e:
        error_msg = f"AWS Error: {e.response['Error']['Code']}"
        logger.error(f"Bedrock API error: {e}")
        return f"❌ **Bedrock API Error:** {error_msg}"
    except Exception as e:
        logger.error(f"Error generating message: {e}")
        return f"❌ **Error:** {str(e)}"


def get_recommendations(customer_id, num_recommendations=5):
    """Get product recommendations for a customer."""
    try:
            df = st.session_state.df
            user_item_matrix = st.session_state.user_item_matrix
            user_sim = st.session_state.user_sim
            user_encoder = st.session_state.user_encoder
            product_encoder = st.session_state.product_encoder
            
            # Check if customer exists
            if customer_id not in df["CustomerID"].values:
                return None, f"❌ Customer ID {int(customer_id)} not found in the dataset"
            
            # Get user index
            target_idx = user_encoder.transform([customer_id])[0]
            
            # Get products already purchased
            purchased_products = user_item_matrix[target_idx]
            
            # Predict scores
            scores = user_sim[target_idx] @ user_item_matrix
            
            # Exclude already purchased products
            scores[purchased_products > 0] = 0
            
            # Get top N recommendations
            top_idx = np.argsort(-scores)[:num_recommendations]
            top_products = product_encoder.inverse_transform(top_idx)
            top_scores = scores[top_idx]
            
            # Create recommendation dataframe
            rec_df = pd.DataFrame({
                "StockCode": top_products,
                "Score": top_scores
            }).merge(df[["StockCode", "Description"]].drop_duplicates(), 
                    on="StockCode", how="left")
            
            return rec_df, None
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return None, f"Error: {str(e)}"


def get_customer_history(customer_id):
    """Get purchase history for a customer."""
    try:
        df = st.session_state.df
        
        if customer_id not in df["CustomerID"].values:
            return None, f"❌ Customer ID {int(customer_id)} not found"
        
        history = df[df["CustomerID"] == customer_id][
            ["StockCode", "Description", "Quantity"]
        ].groupby(["StockCode", "Description"], as_index=False)["Quantity"].sum()
        
        history = history.sort_values("Quantity", ascending=False)
        
        return history, None
        
    except Exception as e:
        logger.error(f"Error getting customer history: {e}")
        return None, str(e)


def calculate_rfm():
    """Calculate RFM (Recency, Frequency, Monetary) scores for customers."""
    try:
        df = st.session_state.df
        
        # Check if required columns exist
        if 'InvoiceDate' not in df.columns:
            return None, "InvoiceDate column not found. RFM analysis requires date information."
        
        # Convert InvoiceDate to datetime if it's not already
        df_rfm = df.copy()
        df_rfm['InvoiceDate'] = pd.to_datetime(df_rfm['InvoiceDate'], errors='coerce')
        
        # Remove rows with invalid dates
        df_rfm = df_rfm.dropna(subset=['InvoiceDate'])
        
        if len(df_rfm) == 0:
            return None, "No valid date data available for RFM analysis."
        
        # Calculate Monetary value
        if 'UnitPrice' in df_rfm.columns:
            df_rfm['TotalPrice'] = df_rfm['Quantity'] * df_rfm['UnitPrice']
        else:
            # Use Quantity as proxy if UnitPrice not available
            df_rfm['TotalPrice'] = df_rfm['Quantity']
        
        # Get the most recent date in the dataset
        snapshot_date = df_rfm['InvoiceDate'].max() + pd.Timedelta(days=1)
        
        # Calculate RFM metrics
        rfm = df_rfm.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
            'CustomerID': 'count',  # Frequency
            'TotalPrice': 'sum'  # Monetary
        })
        
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        # Calculate RFM scores (1-5 scale, 5 being best)
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        
        # Convert scores to integers
        rfm['R_Score'] = rfm['R_Score'].astype(int)
        rfm['F_Score'] = rfm['F_Score'].astype(int)
        rfm['M_Score'] = rfm['M_Score'].astype(int)
        
        # Calculate RFM Score
        rfm['RFM_Score'] = rfm['R_Score'] * 100 + rfm['F_Score'] * 10 + rfm['M_Score']
        
        # Segment customers
        def segment_customer(row):
            if row['R_Score'] >= 4 and row['F_Score'] >= 4 and row['M_Score'] >= 4:
                return 'Champions'
            elif row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
                return 'Loyal Customers'
            elif row['R_Score'] >= 4 and row['F_Score'] <= 2:
                return 'Promising'
            elif row['R_Score'] >= 3 and row['F_Score'] <= 2 and row['M_Score'] <= 2:
                return 'Need Attention'
            elif row['R_Score'] <= 2 and row['F_Score'] >= 3:
                return 'At Risk'
            elif row['R_Score'] <= 2 and row['F_Score'] <= 2 and row['M_Score'] >= 4:
                return 'Cant Lose Them'
            elif row['R_Score'] <= 2 and row['F_Score'] <= 2 and row['M_Score'] <= 2:
                return 'Lost'
            else:
                return 'Others'
        
        rfm['Segment'] = rfm.apply(segment_customer, axis=1)
        
        # Reset index to make CustomerID a column
        rfm = rfm.reset_index()
        
        return rfm, None
        
    except Exception as e:
        logger.error(f"Error calculating RFM: {e}")
        return None, str(e)


def render_rfm_analysis():
    """Render RFM analysis interface."""
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("📊 RFM Customer Segmentation")
    with col2:
        if st.button("🔄 Refresh RFM", help="Recalculate RFM analysis"):
            st.session_state.rfm_calculated = False
            st.session_state.rfm_data = None
            st.rerun()
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first")
        return
    
    # Calculate RFM only if not already calculated
    if not st.session_state.rfm_calculated or st.session_state.rfm_data is None:
        with st.spinner("Calculating RFM scores..."):
            rfm_df, error = calculate_rfm()
        
        if error:
            st.error(f"❌ {error}")
            st.info("💡 **Tip:** RFM analysis requires 'InvoiceDate' column in your data.")
            return
        
        # Cache the result
        st.session_state.rfm_data = rfm_df
        st.session_state.rfm_calculated = True
    else:
        # Use cached RFM data
        rfm_df = st.session_state.rfm_data
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(rfm_df):,}")
    with col2:
        avg_recency = rfm_df['Recency'].mean()
        st.metric("Avg Recency", f"{avg_recency:.0f} days")
    with col3:
        avg_frequency = rfm_df['Frequency'].mean()
        st.metric("Avg Frequency", f"{avg_frequency:.1f}")
    with col4:
        avg_monetary = rfm_df['Monetary'].mean()
        st.metric("Avg Monetary", f"${avg_monetary:,.0f}")
    
    st.markdown("---")
    
    # Two-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer Segments Distribution
        segment_counts = rfm_df['Segment'].value_counts()
        
        fig1 = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Customer Segments Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1, use_container_width=True)
        
        # Segment statistics table
        st.markdown("**📋 Segment Statistics**")
        segment_stats = rfm_df.groupby('Segment').agg({
            'CustomerID': 'count',
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        }).round(2)
        segment_stats.columns = ['Count', 'Avg Recency', 'Avg Frequency', 'Avg Monetary']
        segment_stats = segment_stats.sort_values('Count', ascending=False)
        st.dataframe(segment_stats, use_container_width=True)
    
    with col2:
        # RFM 3D Scatter Plot
        fig2 = px.scatter_3d(
            rfm_df,
            x='Recency',
            y='Frequency',
            z='Monetary',
            color='Segment',
            title='RFM 3D Distribution',
            labels={'Recency': 'Recency (days)', 'Frequency': 'Frequency', 'Monetary': 'Monetary ($)'},
            color_discrete_sequence=px.colors.qualitative.Set3,
            height=500
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # RFM Score Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig3 = px.histogram(
            rfm_df,
            x='R_Score',
            title='Recency Score Distribution',
            nbins=5,
            color_discrete_sequence=['#667eea'],
            labels={'R_Score': 'Recency Score (1=Worst, 5=Best)'}
        )
        fig3.update_xaxes(range=[0.5, 5.5], dtick=1)
        fig3.update_layout(bargap=0.1)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        fig4 = px.histogram(
            rfm_df,
            x='F_Score',
            title='Frequency Score Distribution',
            nbins=5,
            color_discrete_sequence=['#764ba2'],
            labels={'F_Score': 'Frequency Score (1=Worst, 5=Best)'}
        )
        fig4.update_xaxes(range=[0.5, 5.5], dtick=1)
        fig4.update_layout(bargap=0.1)
        st.plotly_chart(fig4, use_container_width=True)
    
    # Top customers table with segment filter
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 🏆 Top 20 Customers by RFM Score")
    with col2:
        # Segment selector
        segments = ['All Segments'] + sorted(rfm_df['Segment'].unique().tolist())
        selected_segment = st.selectbox(
            "Filter by Segment:",
            segments,
            key="segment_filter"
        )
    
    # Filter by selected segment
    if selected_segment == 'All Segments':
        filtered_rfm = rfm_df
        segment_info = "All Segments"
    else:
        filtered_rfm = rfm_df[rfm_df['Segment'] == selected_segment]
        segment_info = selected_segment
    
    # Get top 20 from filtered data
    top_customers = filtered_rfm.nlargest(20, 'RFM_Score')[
        ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'R_Score', 'F_Score', 'M_Score', 'RFM_Score', 'Segment']
    ].copy()
    
    # Format CustomerID as string
    top_customers['CustomerID'] = top_customers['CustomerID'].apply(lambda x: f"Customer {int(x)}")
    
    # Show count info
    st.caption(f"Showing top {len(top_customers)} customers from **{segment_info}** (Total: {len(filtered_rfm)} customers)")
    
    st.dataframe(
        top_customers.style.background_gradient(subset=['RFM_Score'], cmap='Blues'),
        use_container_width=True
    )
    
    # Download button
    csv = rfm_df.to_csv(index=False)
    st.download_button(
        label="📥 Download RFM Analysis (CSV)",
        data=csv,
        file_name="rfm_analysis.csv",
        mime="text/csv"
    )


def create_visualizations():
    """Create enhanced data visualizations."""
    df = st.session_state.df
    
    st.subheader("📈 Sales Analytics Dashboard")
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = df['CustomerID'].nunique()
        st.metric("👥 Total Customers", f"{total_customers:,}")
    
    with col2:
        total_products = df['StockCode'].nunique()
        st.metric("📦 Total Products", f"{total_products:,}")
    
    with col3:
        total_transactions = len(df)
        st.metric("🛒 Transactions", f"{total_transactions:,}")
    
    with col4:
        if 'UnitPrice' in df.columns and 'Quantity' in df.columns:
            total_revenue = (df['UnitPrice'] * df['Quantity']).sum()
            st.metric("💰 Revenue", f"${total_revenue:,.0f}")
        else:
            avg_quantity = df['Quantity'].mean()
            st.metric("📊 Avg Quantity", f"{avg_quantity:.1f}")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 customers
        top_customers = df.groupby('CustomerID').size().sort_values(ascending=False).head(10)
        customer_ids_str = [f"Customer {int(cid)}" for cid in top_customers.index]
        
        # Custom colorscale with darker minimum for better visibility
        custom_blues = [[0, '#4A90E2'], [1, '#001F54']]
        
        fig1 = go.Figure(data=[
            go.Bar(
                x=customer_ids_str,
                y=top_customers.values,
                marker=dict(
                    color=top_customers.values,
                    colorscale=custom_blues,
                    showscale=False
                ),
                text=top_customers.values,
                textposition='outside'
            )
        ])
        fig1.update_layout(
            title="🏆 Top 10 Customers by Purchase Count",
            xaxis_title="Customer",
            yaxis_title="Number of Purchases",
            height=400,
            showlegend=False,
            hovermode='x'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Top 10 products
        top_products = df.groupby('Description').size().sort_values(ascending=False).head(10)
        
        # Custom colorscale with darker minimum for better visibility
        custom_greens = [[0, '#52C41A'], [1, '#135200']]
        
        fig2 = go.Figure(data=[
            go.Bar(
                x=top_products.values,
                y=[desc[:40] + '...' if len(desc) > 40 else desc for desc in top_products.index],
                orientation='h',
                marker=dict(
                    color=top_products.values,
                    colorscale=custom_greens,
                    showscale=False
                ),
                text=top_products.values,
                textposition='outside'
            )
        ])
        fig2.update_layout(
            title="🎁 Top 10 Most Popular Products",
            xaxis_title="Purchase Count",
            yaxis_title="Product",
            height=400,
            showlegend=False,
            hovermode='y'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Quantity distribution
    quantity_95th = df['Quantity'].quantile(0.95)
    max_quantity = min(quantity_95th, 100)
    df_filtered = df[df['Quantity'] <= max_quantity]
    
    fig3 = px.histogram(
        df_filtered,
        x='Quantity',
        title=f'📊 Purchase Quantity Distribution (showing 0-{int(max_quantity)})',
        nbins=50,
        labels={'Quantity': 'Quantity Purchased'},
        color_discrete_sequence=['#667eea']
    )
    fig3.update_layout(height=350, showlegend=False)
    
    if len(df_filtered) < len(df):
        st.caption(f"ℹ️ Showing {len(df_filtered):,} of {len(df):,} transactions (filtered for better visualization)")
    
    st.plotly_chart(fig3, use_container_width=True)


def render_sidebar():
    """Render enhanced sidebar."""
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # Show required headers first
        st.markdown("### 📋 Required CSV Headers")
        st.markdown("""
        Your CSV file must contain these exact column names:
        
        **Required Fields:**
        - `InvoiceNo` - Invoice number
        - `StockCode` - Product stock code  
        - `Description` - Product description
        - `Quantity` - Quantity purchased
        - `InvoiceDate` - Date of invoice
        - `UnitPrice` - Price per unit
        - `CustomerID` - Customer identifier
        - `Country` - Country of sale
        
        **Note:** Column names are case-sensitive!
        """)
        
        st.markdown("---")
        
        # Simple file uploader
        uploaded_file = st.file_uploader(
            "Upload sales data",
            type=['csv'],
            help="CSV file with the required headers above"
        )
        
        # Simple checkbox for sample data
        use_sample = st.checkbox("📦 Use sample data instead", 
                                help="Load OnlineRetail.csv demo dataset")
        
        if use_sample:
            sample_already_loaded = (st.session_state.data_loaded and 
                                   st.session_state.last_uploaded_file == "sample_OnlineRetail.csv")
            
            if not sample_already_loaded:
                try:
                    sample_path = "OnlineRetail.csv"
                    
                    if os.path.exists(sample_path):
                        with st.spinner("Loading..."):
                            df = pd.read_csv(sample_path, encoding='ISO-8859-1')
                            
                            if load_and_process_data(df):
                                st.session_state.last_uploaded_file = "sample_OnlineRetail.csv"
                                st.rerun()
                    else:
                        st.error("Sample file not found")
                
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.error(f"Sample data load error: {e}", exc_info=True)
            
            uploaded_file = None  # Ignore uploaded file if sample is selected
        
        if uploaded_file is not None:
            try:
                file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                is_new_file = (st.session_state.last_uploaded_file != file_id)
                
                if is_new_file:
                    # Load file
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    # Process
                    if load_and_process_data(df):
                        st.session_state.last_uploaded_file = file_id
                else:
                    pass
                
            except Exception as e:
                st.error(f"Error: {e}")
                logger.error(f"File load error: {e}", exc_info=True)
        
        # Status and Data Preview
        if st.session_state.data_loaded:
            st.markdown("---")
            st.caption("✅ Data ready")
            
            # Toggle button for data preview
            show_preview = st.checkbox("📊 Show Data Preview", value=False, help="Toggle to show/hide data preview")
            
            if show_preview:
                df = st.session_state.df
                
                # Show basic stats
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rows", f"{len(df):,}")
                with col2:
                    st.metric("Columns", len(df.columns))
                
                # Show first few rows
                st.markdown("**First 5 rows:**")
                preview_df = df.head(5)
                
                # Truncate long text for better display
                if 'Description' in preview_df.columns:
                    preview_df = preview_df.copy()
                    preview_df['Description'] = preview_df['Description'].apply(
                        lambda x: x[:30] + '...' if len(str(x)) > 30 else x
                    )
                
                st.dataframe(
                    preview_df,
                    use_container_width=True,
                    hide_index=True,
                    height=200
                )
        
        if st.session_state.bedrock_connected:
            st.caption("🟢 AWS Bedrock connected")
        
        st.markdown("---")
        st.caption("Built with Streamlit & AWS Bedrock")

def process_chat_message_with_bedrock(user_input: str) -> str:
    """Wrapper that removes reasoning tags from the response."""
    result = process_chat_message_with_bedrock_withReasoning(user_input)
    
    # Remove <reasoning> tags if present in the final output
    if "<reasoning>" in result.lower():
        result = re.sub(r'<reasoning>.*?</reasoning>', '', result, flags=re.DOTALL | re.IGNORECASE)
        result = result.strip()
    
    return result

def process_chat_message_with_bedrock_withReasoning(user_input: str) -> str:
    """Process chat message using AWS Bedrock with SQL generation capability."""
    try:
        if st.session_state.bedrock_client is None:
            client, error = initialize_bedrock_client()
            if error:
                return f"⚠️ Bedrock not connected. Error: {error}"
        else:
            client = st.session_state.bedrock_client
        
        # Check if question requires data query
        if st.session_state.data_loaded and st.session_state.duckdb_conn is not None:
            # First, check if this needs SQL
            sql_check_prompt = f"""Analyze this question: "{user_input}"

Does this question require querying a database to answer? Answer ONLY "YES" or "NO".

Examples that need SQL:
- "show me top 10 customers"
- "what are the most popular products"
- "list customers from UK"
- "who spent the most"

Examples that DON'T need SQL:
- "how does the recommendation system work"
- "what is RFM analysis"
- "explain collaborative filtering"

Answer (YES or NO):"""
            
            payload = {
                "messages": [
                    {"role": "user", "content": sql_check_prompt}
                ]
            }
            
            response = client.invoke_model(
                modelId=st.session_state.bedrock_model,
                body=json.dumps(payload),
                contentType="application/json"
            )
            
            body = json.loads(response["body"].read().decode("utf-8"))
            needs_sql = body["choices"][0]["message"]["content"].strip().upper()
            
            # If needs SQL, generate and execute it
            if "YES" in needs_sql:
                try:
                    # Get actual schema from DuckDB (like DuckDBAzureSample.py)
                    schema_result = st.session_state.duckdb_conn.execute("DESCRIBE sales_data").fetchall()
                    schema_str = "\n".join([f"  {col[0]} ({col[1]})" for col in schema_result])
                    
                    # Get table stats
                    row_count = st.session_state.duckdb_conn.execute("SELECT COUNT(*) FROM sales_data").fetchone()[0]
                    
                    # Generate SQL prompt (following DuckDBAzureSample.py pattern)
                    sql_gen_prompt = f"""You are a DuckDB SQL expert.
Based on the following table structure and English question, generate one executable SQL query for DuckDB.

【Available Table】
Table: sales_data
Rows: {row_count:,}
Schema:
{schema_str}

【User Question】
{user_input}

【Important Rules】
- Output only the SQL query, no comments or explanations.
- Filter NULL CustomerIDs: WHERE CustomerID IS NOT NULL
- Calculate revenue as: UnitPrice * Quantity  
- Filter positive quantities only: WHERE Quantity > 0
- Use CAST(CustomerID AS INTEGER) to show clean IDs
- DuckDB does not support CREATE CHART or EXECUTE IMMEDIATE
- No need <reasoning> tags in the output.
"""
                    
                    # Generate SQL
                    payload = {
                        "messages": [
                            {"role": "user", "content": sql_gen_prompt}
                        ]
                    }
                    
                    response = client.invoke_model(
                        modelId=st.session_state.bedrock_model,
                        body=json.dumps(payload),
                        contentType="application/json"
                    )
                    
                    body = json.loads(response["body"].read().decode("utf-8"))
                    sql_query = body["choices"][0]["message"]["content"].strip()
                    
                    # Remove <reasoning> tags if present
                    if "<reasoning>" in sql_query.lower():
                        sql_query = re.sub(r'<reasoning>.*?</reasoning>', '', sql_query, flags=re.DOTALL | re.IGNORECASE)
                    
                    # Clean SQL (remove markdown if present)
                    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
                    
                    # Execute SQL (following DuckDBAzureSample.py pattern)
                    result_df = st.session_state.duckdb_conn.execute(sql_query).fetchdf()
                    
                    # Check if results are empty
                    if len(result_df) == 0:
                        return f"📊 **SQL Query:**\n```sql\n{sql_query}\n```\n\n❌ No results found."
                    
                    # Format results
                    result_text = result_df.head(20).to_string(index=False)
                    if len(result_df) > 20:
                        result_text += f"\n\n_Showing 20 of {len(result_df)} results_"
                    
                    # Ask AI to interpret results
                    interpret_prompt = f"""User asked: "{user_input}"

Query results:
{result_text}

Provide a friendly 2-3 sentence summary. Be conversational."""
                    
                    payload = {
                        "messages": [
                            {"role": "user", "content": interpret_prompt}
                        ]
                    }
                    
                    response = client.invoke_model(
                        modelId=st.session_state.bedrock_model,
                        body=json.dumps(payload),
                        contentType="application/json"
                    )
                    
                    body = json.loads(response["body"].read().decode("utf-8"))
                    interpretation = body["choices"][0]["message"]["content"].strip()
                    
                    # Return formatted response with SQL query shown
                    return f"{interpretation}\n\n📊 **SQL Query:**\n```sql\n{sql_query}\n```\n\n📈 **Results:**\n```\n{result_text}\n"
                    
                except Exception as sql_error:
                    logger.error(f"SQL execution error: {sql_error}")
                    return f"❌ Error executing SQL: {str(sql_error)}\n\nTry rephrasing your question or ask about the data structure."
        
        # Fall back to regular chat (no SQL needed)
        df = st.session_state.df
        context = f"""You are a friendly AI shopping assistant.

**Dataset Summary:**
- Customers: {df['CustomerID'].nunique():,}
- Products: {df['StockCode'].nunique():,}
- Transactions: {len(df):,}

Answer questions about the recommendation system, explain features, or guide users."""
        
        system_prompt = """You are a helpful AI assistant. Be friendly and concise."""
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{context}\n\nQuestion: {user_input}"}
            ]
        }
        
        response = client.invoke_model(
            modelId=st.session_state.bedrock_model,
            body=json.dumps(payload),
            contentType="application/json"
        )
        
        body = json.loads(response["body"].read().decode("utf-8"))
        answer = body["choices"][0]["message"]["content"]
        
        return answer
        
    except ClientError as e:
        error_msg = f"AWS Error: {e.response['Error']['Code']}"
        logger.error(f"Bedrock chat error: {e}")
        return f"❌ Bedrock API Error: {error_msg}\n\n💡 Please check your connection in the sidebar."
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return f"❌ Error: {str(e)}\n\n💡 Try rephrasing your question or check the sidebar for connection status."


def render_chat_tab():
    """Render chat interface with AWS Bedrock."""
    st.subheader("💬 AI Shopping Assistant")
    
    # Status checks
    if not st.session_state.data_loaded:
        st.warning("⚠️ **No Data Loaded**")
        st.info("👈 Upload your CSV/Excel file in the sidebar to get started!")
        return
    
    if not st.session_state.bedrock_connected:
        st.error("⚠️ **AWS Bedrock Not Connected**")
        st.info("👈 Configure and test your AWS Bedrock connection in the sidebar")
        return
    
    # Ready indicator
    st.success("✅ Ready to chat! Ask me anything about your data.")
    
    # Chat history
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.chat_history:
            st.info("""
            👋 **Welcome!** I can query your data and provide insights:
            
            **Try asking:**
            - "Show me top 10 customers by revenue"
            - "What are the most popular products?"
            - "List customers from UK"
            - "Which products generate the most revenue?"
            - "How many customers bought more than 100 items?"
            
            I'll automatically generate and run SQL queries to give you real answers! 🚀
            """)
        
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="chat-message user-message">👤 **You:** {message["content"]}</div>', 
                    unsafe_allow_html=True
                )
            else:
                with st.container():
                    st.markdown(f'🤖 **Assistant:**')
                    st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input("💭 Type your question here...")
    
    if user_input and not st.session_state.processing_message:
        st.session_state.processing_message = True
        
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # Process with Bedrock
        with st.spinner("🤖 Thinking..."):
            response = process_chat_message_with_bedrock(user_input)
        
        # Add response
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now()
        })
        
        st.session_state.processing_message = False
        st.rerun()
    
    # Clear button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🗑️ Clear"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Example queries
    with st.expander("💡 Example Questions", expanded=False):
        st.markdown("""
        **📊 Data Analysis:**
        - "What are my top 10 customers?"
        - "What are the 5 most popular products?"
        - "How many countries do we sell to?"
        
        **🎯 Recommendations:**
        - "How does the recommendation system work?"
        - "What factors determine product recommendations?"
        
        **🛍️ Customer Insights:**
        - "Tell me about the customer segments"
        - "What defines a 'Champion' customer?"
        
        **📈 Business Insights:**
        - "Give me an overview of the data"
        - "What insights can you provide?"
        """)


def render_customer_lookup():
    """Render enhanced customer lookup interface with sub-tabs."""
    st.subheader("🔍 Customer Lookup & Recommendations")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first")
        return
    
    # Customer selection with search
    customers = sorted(st.session_state.df["CustomerID"].unique())
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_customer = st.selectbox(
            "🔎 Select Customer ID:",
            customers,
            format_func=lambda x: f"Customer {int(x)}",
            help="Choose a customer to view their history and recommendations"
        )
    
    with col2:
        st.metric("Total Customers", f"{len(customers):,}")
    
    # Main layout: Left side with sub-tabs, Right side with AI message
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        # Create sub-tabs for Recommendations and Purchase History
        subtab1, subtab2 = st.tabs(["🎯 Recommendations", "🛍️ Purchase History"])
        
        with subtab1:
            num_recs = st.slider(
                "Number of recommendations:",
                min_value=1,
                max_value=20,
                value=5,
                help="Adjust to see more or fewer recommendations"
            )
            
            rec_df, error = get_recommendations(selected_customer, num_recs)
            
            if error:
                st.error(error)
            else:
                # Display recommendations with same styling as purchase history
                st.dataframe(
                    rec_df.style.format({"Score": "{:.2f}"}),
                    use_container_width=True,
                    height=400
                )
        
        with subtab2:
            history_df, error = get_customer_history(selected_customer)
            
            if error:
                st.error(error)
            else:
                # Display with better formatting
                st.dataframe(
                    history_df.style.format({"Quantity": "{:.0f}"}),
                    use_container_width=True,
                    height=400
                )
                
                total_items = len(history_df)
                total_qty = history_df['Quantity'].sum()
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("📦 Unique Products", total_items)
                with metric_col2:
                    st.metric("🔢 Total Quantity", f"{int(total_qty)}")
    
    with right_col:
        # Add top margin to prevent text being covered
        st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
        
        st.subheader("✨ AI Personalized Message")
        
        # Auto-generate AI message using Bedrock
        if st.session_state.bedrock_connected:
            # Get history and recommendations
            history_df, hist_error = get_customer_history(selected_customer)
            rec_df, rec_error = get_recommendations(selected_customer, 5)
            
            if not hist_error and not rec_error:
                with st.spinner("💬 Generating personalized message with AWS Bedrock..."):
                    message = generate_personalized_message(
                        selected_customer,
                        history_df,
                        rec_df
                    )
                
                # Display the message
                st.info(message)
                
                # Copy button with JavaScript clipboard functionality
                copy_to_clipboard(message)
            else:
                st.error("❌ Unable to generate message - data error")
        else:
            st.warning("⚠️ **AWS Bedrock Not Connected**")
            st.info("Connect to AWS Bedrock in the sidebar to automatically generate personalized recommendation messages.")
            st.markdown("""
            **Why connect?**
            - Get AI-powered personalized messages
            - Automatic generation based on customer data
            - Professional recommendation text
            """)


def main():
    """Main application."""
    
    # Auto-test Bedrock connection on startup
    if not st.session_state.bedrock_connection_tested:
        try:
            connected, error = test_bedrock_connection()
            st.session_state.bedrock_connected = connected
            st.session_state.bedrock_connection_tested = True
            if connected:
                logger.info("Bedrock connection successful on startup")
            else:
                logger.warning(f"Bedrock connection failed on startup: {error}")
        except Exception as e:
            logger.error(f"Error testing Bedrock connection on startup: {e}")
            st.session_state.bedrock_connection_tested = True
    
    # Header
    st.header("🛒 SmartSales AI")
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    if not st.session_state.data_loaded:
        # Welcome screen
        st.info("👆 **Get Started:** Upload your data or use sample data in the sidebar")
        
        st.markdown("""
        ### Features
        - 📊 Interactive visualizations
        - 🤖 AI-powered recommendations  
        - 📈 Customer segmentation (RFM)
        - 💬 Chat with AWS Bedrock AI
        """)
        
    else:
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Visualizations", 
            "📈 RFM Analysis",
            "🔍 Customer Lookup",
            "💬 Chat"
        ])
        
        with tab1:
            create_visualizations()
        
        with tab2:
            render_rfm_analysis()
        
        with tab3:
            render_customer_lookup()
        
        with tab4:
            render_chat_tab()


if __name__ == "__main__":
    main()

