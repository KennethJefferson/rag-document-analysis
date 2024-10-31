import streamlit as st
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
import PyPDF2
from io import BytesIO
import atexit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Remove all CUDA checks and force CPU mode for consistency
torch.set_num_threads(4)  # Optimize CPU threading

# Add performance optimization through caching
@st.cache_resource
def load_sentence_transformer(model_name: str):
    return SentenceTransformer(model_name)

@st.cache_data
def get_embeddings(texts: List[str], _model):
    return _model.encode(texts)

class DocumentProcessor:
    def __init__(self):
        self.supported_formats = ['pdf', 'txt']
        logger.info("Document processor initialized")
    
    def process_document(self, content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process uploaded document content.
        """
        try:
            # Get file extension
            file_ext = filename.split('.')[-1].lower()
            
            if file_ext not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Process based on file type
            if file_ext == 'txt':
                text_content = content.decode('utf-8')
            elif file_ext == 'pdf':
                text_content = self._process_pdf(content)
            
            # Create metadata
            metadata = {
                'filename': filename,
                'format': file_ext,
                'processed_timestamp': datetime.now().isoformat(),
                'size_bytes': len(content)
            }
            
            logger.info(f"Processed document {filename} with {len(text_content)} characters")
            
            return {
                'page_content': text_content,  # Ensure we use page_content consistently
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {str(e)}")
            raise
    
    def _process_pdf(self, content: bytes) -> str:
        """
        Process PDF content using PyPDF2
        """
        try:
            # Create PDF reader object
            pdf_file = BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n\n"
            
            return text_content
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise

class RelevanceFeedbackSystem:
    def __init__(self):
        # Initialize with dummy data if it doesn't exist in session state
        if 'feedback_data' not in st.session_state:
            # Create dummy data with minute intervals over the last hour
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(hours=1)
            dates = pd.date_range(start=start_date, end=end_date, freq='1min')
            
            st.session_state.feedback_data = [
                {
                    'timestamp': date,
                    'query': f'Sample Query {i}',
                    'response': f'Sample Response {i}',
                    'relevance_score': np.random.uniform(0.3, 0.9)
                }
                for i, date in enumerate(dates)
            ]
        
        self.feedback_data = st.session_state.feedback_data
        self.feedback_df = pd.DataFrame(self.feedback_data)
        logger.info("Relevance feedback system initialized")
    
    def create_trend_chart(self) -> go.Figure:
        """
        Create relevance score trend visualization with minute-level granularity
        """
        # Sort by timestamp to ensure proper ordering
        df_sorted = self.feedback_df.sort_values('timestamp')
        
        fig = px.line(
            df_sorted,
            x='timestamp',
            y='relevance_score',
            title='Relevance Score Trend (Last Hour)',
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Relevance Score",
            yaxis_range=[0, 1],
            hovermode='x unified',
            xaxis_tickformat='%H:%M:%S',  # Show hours:minutes:seconds
        )
        
        # Add more granular gridlines and adjust tick spacing
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            dtick=60000,  # Show gridlines every minute (in milliseconds)
            tickmode='linear'
        )
        
        return fig

    def collect_feedback(self, query: str, response: str, relevance_score: float):
        """
        Collect user feedback with precise timestamp
        """
        new_feedback = {
            'timestamp': datetime.now(),
            'query': query,
            'response': response,
            'relevance_score': relevance_score
        }
        
        # Add new feedback and remove old data (keep last hour only)
        self.feedback_data.append(new_feedback)
        
        # Filter to keep only last hour of data
        cutoff_time = datetime.now() - pd.Timedelta(hours=1)
        self.feedback_data = [
            fb for fb in self.feedback_data 
            if fb['timestamp'] > cutoff_time
        ]
        
        # Update session state
        st.session_state.feedback_data = self.feedback_data
        self.feedback_df = pd.DataFrame(self.feedback_data)
        
        logger.info(f"Feedback collected: score={relevance_score}")

class RAGApplication:
    def __init__(self):
        try:
            # Force GPU usage if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
            if device == "cuda":
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("GPU not available, using CPU")
            
            self.vector_store = None
            self.gemini = None
            self.document_processor = DocumentProcessor()
            self.query_history = []
            self.feedback_system = RelevanceFeedbackSystem()
            self.api_key = None
            
            # Initialize visualizations data
            self.response_times = []
            self.relevance_scores = []
            
            # Initialize response time tracking
            if 'response_time_data' not in st.session_state:
                # Create dummy data with minute intervals over the last hour
                end_date = datetime.now()
                start_date = end_date - pd.Timedelta(hours=1)
                dates = pd.date_range(start=start_date, end=end_date, freq='1min')
                
                st.session_state.response_time_data = [
                    {
                        'timestamp': date,
                        'query': f'Sample Query {i}',
                        'response_time': np.random.uniform(0.5, 3.0)  # Random times between 0.5 and 3 seconds
                    }
                    for i, date in enumerate(dates)
                ]
            
            self.response_time_data = st.session_state.response_time_data
            self.response_time_df = pd.DataFrame(self.response_time_data)
            
            # Initialize query volume tracking
            if 'query_volume_data' not in st.session_state:
                # Create dummy data with hourly intervals for the past week
                end_date = datetime.now()
                start_date = end_date - pd.Timedelta(days=7)
                dates = pd.date_range(start=start_date, end=end_date, freq='h')
                
                st.session_state.query_volume_data = [
                    {
                        'timestamp': date,
                        'day': date.strftime('%A'),
                        'hour': date.hour,
                        'count': np.random.randint(1, 10)  # Random query counts
                    }
                    for date in dates
                ]
            
            self.query_volume_data = st.session_state.query_volume_data
            self.query_volume_df = pd.DataFrame(self.query_volume_data)
            
            # Add cleanup handler
            atexit.register(self.cleanup)
            
            logger.info("RAG Application initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG Application: {str(e)}")
            self.cleanup()
            raise

    def setup_gemini(self, api_key: str):
        """
        Setup Gemini API with the provided API key.
        
        Args:
            api_key: Google API key for Gemini
        """
        try:
            # Only configure if API key has changed
            if self.api_key != api_key:
                logger.info("Configuring Gemini API with new key")
                genai.configure(api_key=api_key)
                self.gemini = genai.GenerativeModel('gemini-pro')
                self.api_key = api_key
                logger.info("Gemini API configured successfully")
                
                # Verify initialization
                if self.gemini is None:
                    raise ValueError("Gemini model initialization failed")
                
        except Exception as e:
            logger.error(f"Error configuring Gemini API: {str(e)}")
            raise

    def create_visualizations(self):
        """
        Create analytics visualizations.
        
        Returns:
            Tuple of three Plotly figures
        """
        try:
            # Generate some dummy data for demonstration
            dates = pd.date_range(start='2024-01-01', end='2024-03-15', freq='H')
            dummy_data = pd.DataFrame({
                'timestamp': dates,
                'response_time': np.random.normal(2, 0.5, len(dates)),  # Random response times around 2 seconds
                'relevance_score': np.random.uniform(0.5, 1.0, len(dates)),  # Random relevance scores
                'hour': dates.hour,
                'day': dates.day_name()
            })

            # 1. Response Time Trend
            fig1 = px.line(dummy_data, x='timestamp', y='response_time',
                          title='Response Time Trend')
            fig1.update_layout(
                xaxis_title="Time",
                yaxis_title="Response Time (seconds)",
                hovermode='x unified'
            )
            # Add range slider
            fig1.update_xaxes(rangeslider_visible=True)

            # 2. Relevance Score Distribution
            fig2 = px.histogram(dummy_data, x='relevance_score',
                                title='Relevance Score Distribution',
                                nbins=20)
            fig2.update_layout(
                xaxis_title="Relevance Score",
                yaxis_title="Count",
                bargap=0.1
            )

            # 3. Query Volume Heatmap
            # Create pivot table for heatmap
            heatmap_data = dummy_data.pivot_table(
                values='response_time',  # Using response_time as a proxy for query count
                index='day',
                columns='hour',
                aggfunc='count'
            )
            
            fig3 = px.imshow(heatmap_data,
                            title='Query Volume Heatmap',
                            labels=dict(x="Hour of Day", y="Day of Week", color="Query Volume"),
                            aspect='auto')
            fig3.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Day of Week",
            )

            return fig1, fig2, fig3
                
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            raise

    def process_uploaded_file(self, uploaded_file) -> Dict[str, Any]:
        try:
            start_time = time.time()
            
            # Read file content
            file_content = uploaded_file.read()
            logger.info(f"Processing file: {uploaded_file.name} ({len(file_content)} bytes)")
            
            # Process document
            result = self.document_processor.process_document(
                file_content,
                uploaded_file.name
            )
            
            # Initialize vector store if needed
            if self.vector_store is None:
                # Force GPU usage for embeddings
                device = "cuda" if torch.cuda.is_available() else "cpu"
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2",
                    model_kwargs={'device': device}
                )
                
                if device == "cuda":
                    logger.info("Using GPU for embeddings")
                else:
                    logger.warning("GPU not available for embeddings")
                
                # Use page_content instead of content
                self.vector_store = FAISS.from_texts(
                    [result['page_content']], 
                    embeddings
                )
            else:
                # Add to existing vector store
                self.vector_store.add_texts([result['page_content']])  # Changed from content to page_content
            
            processing_time = time.time() - start_time
            logger.info(f"Document {uploaded_file.name} processed in {processing_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing file {uploaded_file.name}: {str(e)}")
            logger.error(f"Document result: {result}")  # Add this for debugging
            raise

    def handle_query(self, query: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            start_time = time.time()
            logger.info(f"Processing query: {query}")
            
            # Verify Gemini is initialized
            if self.gemini is None:
                raise ValueError("Gemini API not initialized. Please check your API key.")
            
            # Extract and verify context
            if not context_docs:
                logger.warning("No context documents provided")
                context = "No relevant context found."
            else:
                # Extract text content from context docs
                context = "\n\n".join([doc.page_content for doc in context_docs])
                logger.info(f"Using context with {len(context)} characters")
            
            # Create a more structured prompt
            prompt = f"""
            Based on the following context, please answer the question. If the context doesn't contain relevant information, please indicate that.

            Context:
            {context}

            Question: {query}
            """
            
            # Add rate limiting/retry logic
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    # Generate response
                    logger.info("Generating response with Gemini")
                    response = self.gemini.generate_content(prompt)
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        logger.warning(f"Rate limit hit, attempt {attempt + 1}/{max_retries}. Waiting {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise
            
            # Calculate metrics
            processing_time = time.time() - start_time
            logger.info(f"Query processed in {processing_time:.2f} seconds")
            
            return {
                'response': response.text,
                'processing_time': processing_time,
                'context_docs': context_docs
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            logger.error(f"Gemini initialized: {self.gemini is not None}")
            logger.error(f"Context docs: {context_docs}")
            raise

    def create_response_time_chart(self) -> go.Figure:
        """
        Create processing time trend visualization with multiple chart types
        """
        # Sort by timestamp to ensure proper ordering
        df_sorted = self.response_time_df.sort_values('timestamp')
        
        # Get chart type from session state (default to 'bar')
        chart_type = st.session_state.get('chart_type', 'bar')
        
        if chart_type == 'bar':
            fig = px.bar(
                df_sorted,
                x='timestamp',
                y='response_time',
                title='Processing Time Trend (Last Hour)',
                template='plotly_white'
            )
            
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Processing Time (seconds)",
                hovermode='x unified',
                bargap=0.1
            )
            
            # Add range slider for bar chart
            fig.update_xaxes(rangeslider_visible=True)
            
        elif chart_type == 'pie':
            # Create time buckets for pie chart
            df_sorted['time_bucket'] = pd.qcut(df_sorted['response_time'], 
                                             q=4, 
                                             labels=['Fast', 'Medium', 'Slow', 'Very Slow'])
            
            bucket_counts = df_sorted['time_bucket'].value_counts()
            
            fig = px.pie(
                values=bucket_counts.values,
                names=bucket_counts.index,
                title='Processing Time Distribution',
                template='plotly_white'
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label'
            )
        
        return fig

    def update_response_time_data(self, query: str, response_time: float):
        """
        Update response time tracking with new data point
        """
        new_data = {
            'timestamp': datetime.now(),
            'query': query,
            'response_time': response_time
        }
        
        # Add new data and remove old data (keep last hour only)
        self.response_time_data.append(new_data)
        
        # Filter to keep only last hour of data
        cutoff_time = datetime.now() - pd.Timedelta(hours=1)
        self.response_time_data = [
            data for data in self.response_time_data 
            if data['timestamp'] > cutoff_time
        ]
        
        # Update session state
        st.session_state.response_time_data = self.response_time_data
        self.response_time_df = pd.DataFrame(self.response_time_data)
        
        logger.info(f"Response time data updated: {response_time:.2f} seconds")

    def create_query_volume_visualization(self) -> go.Figure:
        """
        Create an interactive sunburst visualization of query patterns with real-time updates
        """
        # Force refresh of data
        self.query_volume_df = pd.DataFrame(self.query_volume_data)
        
        # Prepare data for sunburst chart
        df = self.query_volume_df.copy()
        
        # Create hierarchical data structure
        sunburst_data = []
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Calculate totals for percentage tracking
        total_queries = df['count'].sum()
        
        for day in day_order:
            day_data = df[df['day'] == day]
            day_total = day_data['count'].sum()
            
            # Add day level with updated statistics
            sunburst_data.append({
                'ids': day,
                'labels': f"{day}<br>({day_total} queries)",
                'parents': '',
                'values': day_total,
                'text': f'Total: {day_total}',
            })
            
            # Add hour level with detailed statistics
            for hour in range(24):
                hour_data = day_data[day_data['hour'] == hour]
                hour_count = hour_data['count'].sum()
                
                # Always add hour entries for consistent visualization
                hour_id = f"{day}-{hour}"
                sunburst_data.append({
                    'ids': hour_id,
                    'labels': f"{hour:02d}:00",
                    'parents': day,
                    'values': max(hour_count, 0.1),  # Use small value instead of 0 for visibility
                    'text': f'Queries: {hour_count}',
                })
        
        # Convert to DataFrame for plotly
        sunburst_df = pd.DataFrame(sunburst_data)
        
        # Create sunburst chart with enhanced interactivity
        fig = go.Figure(go.Sunburst(
            ids=sunburst_df['ids'],
            labels=sunburst_df['labels'],
            parents=sunburst_df['parents'],
            values=sunburst_df['values'],
            text=sunburst_df['text'],
            branchvalues='total',
            hovertemplate="""
            <b>%{label}</b><br>
            Queries: %{value}<br>
            % of Total: %{percentRoot:.1f}%<br>
            <extra></extra>
            """,
            maxdepth=2,
            insidetextorientation='radial'
        ))
        
        # Update layout with real-time information
        current_time = datetime.now()
        fig.update_layout(
            title={
                'text': f'Query Volume Distribution<br>' +
                       f'<sup>Total Queries: {int(total_queries)} | ' +
                       f'Last Updated: {current_time.strftime("%H:%M:%S")}</sup>',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            width=800,
            height=800,
            margin=dict(t=100, l=0, r=0, b=0)
        )
        
        return fig

    def update_query_volume_data(self):
        """
        Update query volume tracking with new query
        """
        current_time = datetime.now()
        new_data = {
            'timestamp': current_time,
            'day': current_time.strftime('%A'),
            'hour': current_time.hour,
            'count': 1
        }
        
        # Add new data point
        self.query_volume_data.append(new_data)
        
        # Keep only last week of data
        cutoff_time = datetime.now() - pd.Timedelta(days=7)
        self.query_volume_data = [
            data for data in self.query_volume_data 
            if data['timestamp'] > cutoff_time
        ]
        
        # Update session state
        st.session_state.query_volume_data = self.query_volume_data
        self.query_volume_df = pd.DataFrame(self.query_volume_data)

    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'model'):
                # Clear CUDA cache if using GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            logger.info("Cleaned up RAG Application resources")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

def main():
    st.title("📚 RAG-based Document Analysis System")
    
    try:
        # Initialize all session state variables first
        if 'rag_app' not in st.session_state:
            st.session_state.rag_app = RAGApplication()
        if 'current_query_result' not in st.session_state:
            st.session_state.current_query_result = None
        if 'last_query' not in st.session_state:
            st.session_state.last_query = None
        
        # Sidebar configuration
        with st.sidebar:
            st.header("Configuration")
            api_key = st.text_input("Enter Gemini API Key", type="password")
            
            if api_key:
                try:
                    st.session_state.rag_app.setup_gemini(api_key)
                    st.success("Gemini API configured successfully!")
                except Exception as e:
                    st.error(f"Error configuring Gemini API: {str(e)}")
            
            # File upload section
            uploaded_files = st.file_uploader(
                "Upload Documents",
                accept_multiple_files=True,
                type=['pdf', 'txt']
            )
            
            # Process uploaded files
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    for file in uploaded_files:
                        try:
                            result = st.session_state.rag_app.process_uploaded_file(file)
                            st.success(f"Processed {file.name}")
                            
                            # Show document metadata
                            with st.expander(f"Document Details: {file.name}"):
                                st.json(result['metadata'])
                                
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")
        
        # Query section
        st.header("Ask Questions")
        query = st.text_input("Enter your question:")
        
        # Only process query if it's new or intentionally resubmitted
        if query and (query != st.session_state.last_query):
            with st.spinner("Generating response..."):
                try:
                    # Get similar documents
                    context_docs = st.session_state.rag_app.vector_store.similarity_search(query)
                    
                    # Get response
                    result = st.session_state.rag_app.handle_query(query, context_docs)
                    
                    # Update query volume tracking
                    st.session_state.rag_app.update_query_volume_data()
                    
                    # Store results in session state
                    st.session_state.current_query_result = result
                    st.session_state.last_query = query
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
        
        # Display current response if it exists
        if st.session_state.current_query_result:
            result = st.session_state.current_query_result
            
            # Display response and processing time
            st.markdown("### Response:")
            st.markdown(result['response'])
            st.info(f"Processing time: {result['processing_time']:.2f} seconds")
            
            # Add divider before processing time section
            st.divider()
            
            # Display processing time trend
            st.subheader("Processing Time Trend")
            response_time_fig = st.session_state.rag_app.create_response_time_chart()
            st.plotly_chart(response_time_fig, use_container_width=True)
            
            # Add chart type selector beneath the graph
            chart_type = st.select_slider(
                "Select Chart Type",
                options=['bar', 'pie'],
                value=st.session_state.get('chart_type', 'bar')
            )
            st.session_state.chart_type = chart_type
            
            # Add test data button in a separate column
            if st.button("Add Test Data Point"):
                # Add random test data point
                test_time = np.random.uniform(0.5, 3.0)
                st.session_state.rag_app.update_response_time_data(
                    "Test Query",
                    test_time
                )
                st.rerun()
            
            # Add divider before feedback section
            st.divider()
            
            # Query Volume visualization section with real-time updates
            st.subheader("Query Volume Patterns")
            
            # Create single container and placeholder for the chart
            chart_container = st.empty()
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("### Simulation Controls")
                sim_count = st.number_input("Queries to add", 1, 20, 5, step=1)
                sim_hour = st.selectbox(
                    "Hour",
                    options=list(range(24)),
                    format_func=lambda x: f"{x:02d}:00"
                )
                
                if st.button("Add Queries", type="primary"):
                    current_time = datetime.now().replace(hour=sim_hour)
                    
                    # Show progress with visual feedback
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    for i in range(sim_count):
                        # Update progress
                        progress_text.text(f"Adding query {i+1} of {sim_count}...")
                        progress_bar.progress((i + 1) / sim_count)
                        
                        # Add new data
                        new_data = {
                            'timestamp': current_time,
                            'day': current_time.strftime('%A'),
                            'hour': sim_hour,
                            'count': np.random.randint(1, 6)
                        }
                        st.session_state.rag_app.query_volume_data.append(new_data)
                        
                        # Update chart in the single container
                        volume_fig = st.session_state.rag_app.create_query_volume_visualization()
                        chart_container.plotly_chart(volume_fig, use_container_width=True)
                        
                        time.sleep(0.2)  # Short delay for visual effect
                    
                    # Clear progress indicators
                    progress_text.empty()
                    progress_bar.empty()
                    
                    # Show success message
                    st.success(f"Added {sim_count} queries for {current_time.strftime('%A')} at {sim_hour:02d}:00")
            
            # Display initial chart in the single container
            with col1:
                if 'volume_fig' not in locals():  # Only create if not already created
                    volume_fig = st.session_state.rag_app.create_query_volume_visualization()
                    chart_container.plotly_chart(volume_fig, use_container_width=True)
                    
                st.markdown("""
                ### Insights
                - 🔄 **Click** segments to zoom in
                - 👆 **Double-click** to zoom out
                - 📊 **Hover** for detailed statistics
                - 🕒 Updates in real-time
                """)
            
            # Add divider before feedback section
            st.divider()
            
            # Feedback section (always at bottom)
            st.subheader("Response Feedback Trends (must submit feedback to alter)")
            trend_fig = st.session_state.rag_app.feedback_system.create_trend_chart()
            st.plotly_chart(trend_fig, use_container_width=True)
            
            # Only show feedback collection if there's a response
            if st.session_state.current_query_result:
                relevance = st.slider("How relevant was this response?", 0.0, 1.0, 0.5)
                if st.button("Submit Feedback"):
                    st.session_state.rag_app.feedback_system.collect_feedback(
                        st.session_state.last_query,
                        st.session_state.current_query_result['response'],
                        relevance
                    )
                    # Just update the feedback data without rerunning the query
                    st.rerun()
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
