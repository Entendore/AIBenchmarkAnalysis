# streamlit_ollama_benchmark_final.py
import streamlit as st
import ollama
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
import threading
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import queue
from datetime import datetime
import psutil
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Ollama Model Benchmark Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 5px;
    }
    .status-online { background-color: #00ff00; }
    .status-offline { background-color: #ff0000; }
</style>
""", unsafe_allow_html=True)

@dataclass
class BenchmarkResult:
    """Structured benchmark result data"""
    model_name: str
    benchmark_type: str
    timestamp: str
    latency_ms: float
    tokens_per_second: float
    total_tokens: int
    prompt_tokens: int
    response_tokens: int
    prompt: str
    response: str
    success: bool
    error_message: Optional[str] = None
    cpu_percent: float = 0.0
    memory_percent: float = 0.0

class OllamaBenchmark:
    """Core benchmarking engine"""
    
    def __init__(self):
        self.is_running = False
        self.progress_queue = queue.Queue()
        self.status_queue = queue.Queue()  # For status messages
        self.results_queue = queue.Queue()
        self.cached_models = None  # Cache models to avoid duplicate API calls
    
    def check_ollama_status(self) -> bool:
        """Check if Ollama service is running"""
        try:
            ollama.list()
            return True
        except Exception as e:
            logger.error(f"Ollama status check failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Fetch available models from Ollama - OPTIMIZED for newer library versions"""
        # Return cached models if available
        if self.cached_models is not None:
            return self.cached_models
        
        try:
            response = ollama.list()
            logger.info(f"Raw Ollama response type: {type(response)}")
            
            model_names = []
            
            # Handle new ListResponse format (most common in newer versions)
            if hasattr(response, 'models'):
                logger.info(f"Detected ListResponse format with {len(response.models)} models")
                for model in response.models:
                    # Direct access to model property for Model objects
                    if hasattr(model, 'model'):
                        model_names.append(model.model)
                        logger.info(f"Added model: {model.model}")
                    else:
                        logger.warning(f"Model object missing 'model' attribute: {model}")
            
            # Handle dictionary format (older versions)
            elif isinstance(response, dict) and 'models' in response:
                logger.info(f"Detected dictionary format with {len(response['models'])} models")
                for model in response['models']:
                    if isinstance(model, dict):
                        name = model.get('name') or model.get('model') or model.get('id')
                        if name:
                            model_names.append(name)
                        else:
                            logger.warning(f"Could not extract name from model dict: {model}")
                    elif isinstance(model, str):
                        model_names.append(model)
            
            # Handle direct list format
            elif isinstance(response, list):
                logger.info(f"Detected list format with {len(response)} models")
                for model in response:
                    if isinstance(model, dict):
                        name = model.get('name') or model.get('model') or model.get('id')
                        if name:
                            model_names.append(name)
                    elif hasattr(model, 'name'):
                        model_names.append(model.name)
                    elif hasattr(model, 'model'):
                        model_names.append(model.model)
                    elif isinstance(model, str):
                        model_names.append(model)
            
            else:
                logger.error(f"Unexpected response type: {type(response)}")
                st.error(f"Unexpected response format from Ollama API")
            
            # Remove duplicates while preserving order
            model_names = list(dict.fromkeys(model_names))
            logger.info(f"Final model list: {model_names}")
            
            # Cache the results
            self.cached_models = model_names
            
            if not model_names:
                logger.warning("No models found in response")
                fallback_models = ["stablelm2:latest", "llama3.2", "mistral", "qwen2.5"]
                st.warning("No models detected. Using fallback list.")
                return fallback_models
            
            return model_names
            
        except Exception as e:
            logger.exception("Failed to fetch models")
            st.error(f"Failed to fetch models: {e}")
            # Return fallback models on error
            return ["stablelm2:latest", "llama3:latest", "mistral:latest", "qwen2.5:latest"]
    
    def clear_model_cache(self):
        """Clear the cached model list"""
        self.cached_models = None
    
    def run_latency_benchmark(self, model: str, prompt: str, 
                            num_runs: int = 5, max_tokens: int = 100) -> List[BenchmarkResult]:
        """Run latency benchmark with multiple iterations"""
        results = []
        
        for i in range(num_runs):
            try:
                start_time = time.time()
                start_cpu = psutil.cpu_percent()
                start_memory = psutil.virtual_memory().percent
                
                response = ollama.generate(
                    model=model,
                    prompt=prompt,
                    options={"num_predict": max_tokens}
                )
                
                end_time = time.time()
                end_cpu = psutil.cpu_percent()
                end_memory = psutil.virtual_memory().percent
                
                latency_ms = (end_time - start_time) * 1000
                response_tokens = len(response['response'].split())
                prompt_tokens = len(prompt.split())
                total_tokens = prompt_tokens + response_tokens
                
                # Avoid division by zero
                tokens_per_sec = response_tokens / (end_time - start_time) if (end_time - start_time) > 0 else 0
                
                result = BenchmarkResult(
                    model_name=model,
                    benchmark_type="Latency",
                    timestamp=datetime.now().isoformat(),
                    latency_ms=latency_ms,
                    tokens_per_second=tokens_per_sec,
                    total_tokens=total_tokens,
                    prompt_tokens=prompt_tokens,
                    response_tokens=response_tokens,
                    prompt=prompt,
                    response=response['response'],
                    success=True,
                    cpu_percent=(start_cpu + end_cpu) / 2,
                    memory_percent=(start_memory + end_memory) / 2
                )
                results.append(result)
                
            except Exception as e:
                logger.exception(f"Latency benchmark failed for {model}")
                results.append(BenchmarkResult(
                    model_name=model,
                    benchmark_type="Latency",
                    timestamp=datetime.now().isoformat(),
                    latency_ms=0,
                    tokens_per_second=0,
                    total_tokens=0,
                    prompt_tokens=0,
                    response_tokens=0,
                    prompt=prompt,
                    response="",
                    success=False,
                    error_message=str(e)
                ))
            
            # Only update progress if queue exists
            if hasattr(self, 'progress_queue'):
                # Clear queue first to avoid buildup
                while not self.progress_queue.empty():
                    try:
                        self.progress_queue.get_nowait()
                    except queue.Empty:
                        break
                self.progress_queue.put((i + 1) / num_runs * 100)
        
        return results
    
    def run_throughput_benchmark(self, model: str, prompt: str, 
                               duration_seconds: int = 30) -> List[BenchmarkResult]:
        """Run throughput benchmark for a fixed duration"""
        results = []
        start_time = time.time()
        elapsed = 0
        
        try:
            while elapsed < duration_seconds:
                iter_start = time.time()
                try:
                    response = ollama.generate(
                        model=model,
                        prompt=prompt,
                        options={"num_predict": 50}
                    )
                    iter_end = time.time()
                    
                    latency_ms = (iter_end - iter_start) * 1000
                    response_tokens = len(response['response'].split())
                    tokens_per_sec = response_tokens / (iter_end - iter_start) if (iter_end - iter_start) > 0 else 0
                    
                    result = BenchmarkResult(
                        model_name=model,
                        benchmark_type="Throughput",
                        timestamp=datetime.now().isoformat(),
                        latency_ms=latency_ms,
                        tokens_per_second=tokens_per_sec,
                        total_tokens=response_tokens,
                        prompt_tokens=len(prompt.split()),
                        response_tokens=response_tokens,
                        prompt=prompt,
                        response=response['response'],
                        success=True
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Throughput iteration failed: {e}")
                    # Don't break on individual failures
                
                elapsed = time.time() - start_time
                progress = min(elapsed / duration_seconds * 100, 99)
                if hasattr(self, 'progress_queue'):
                    # Clear queue first to avoid buildup
                    while not self.progress_queue.empty():
                        try:
                            self.progress_queue.get_nowait()
                        except queue.Empty:
                            break
                    self.progress_queue.put(progress)
        
        finally:
            # Ensure we always signal 100% completion
            if hasattr(self, 'progress_queue'):
                while not self.progress_queue.empty():
                    try:
                        self.progress_queue.get_nowait()
                    except queue.Empty:
                        break
                self.progress_queue.put(100)
        
        return results

class ResultsManager:
    """Manage benchmark results persistence"""
    
    @staticmethod
    def save_results(results: List[BenchmarkResult], filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = [asdict(result) for result in results]
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        return filename
    
    @staticmethod
    def load_results(filename: str) -> List[BenchmarkResult]:
        """Load results from JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            return [BenchmarkResult(**item) for item in data]
        except FileNotFoundError:
            st.warning(f"File not found: {filename}")
            return []
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            st.error(f"Failed to load results file: {e}")
            return []
    
    @staticmethod
    def results_to_dataframe(results: List[BenchmarkResult]) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        if not results:
            return pd.DataFrame()
        return pd.DataFrame([asdict(result) for result in results])

def create_performance_charts(df: pd.DataFrame):
    """Create interactive Plotly charts"""
    if df.empty or df.shape[0] == 0:
        return None, None
    
    # Filter out failed results for meaningful charts
    successful_df = df[df['success']]
    if successful_df.empty:
        return None, None
    
    # Latency comparison chart
    fig_latency = go.Figure()
    for model in successful_df['model_name'].unique():
        model_data = successful_df[successful_df['model_name'] == model]
        if not model_data.empty:
            fig_latency.add_trace(go.Box(
                y=model_data['latency_ms'],
                name=model,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
    
    fig_latency.update_layout(
        title="Latency Distribution by Model",
        xaxis_title="Model",
        yaxis_title="Latency (ms)",
        hovermode='x unified'
    )
    
    # Throughput comparison
    throughput_df = successful_df.groupby('model_name').agg({
        'tokens_per_second': 'mean',
        'latency_ms': 'mean'
    }).reset_index()
    
    if throughput_df.empty:
        return fig_latency, None
        
    fig_throughput = make_subplots(
        specs=[[{"secondary_y": True}]]
    )
    
    fig_throughput.add_trace(
        go.Bar(x=throughput_df['model_name'], 
               y=throughput_df['tokens_per_second'],
               name="Tokens/sec"),
        secondary_y=False,
    )
    
    fig_throughput.add_trace(
        go.Scatter(x=throughput_df['model_name'], 
                   y=throughput_df['latency_ms'],
                   mode='lines+markers',
                   name="Avg Latency (ms)"),
        secondary_y=True,
    )
    
    fig_throughput.update_layout(
        title="Throughput vs Latency Comparison",
        xaxis_title="Model"
    )
    fig_throughput.update_yaxes(title_text="Tokens per Second", secondary_y=False)
    fig_throughput.update_yaxes(title_text="Average Latency (ms)", secondary_y=True)
    
    return fig_latency, fig_throughput

def main():
    """Main application"""
    st.markdown("<h1 class='main-header'>⚡ Ollama Model Benchmark Suite</h1>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'benchmark_results' not in st.session_state:
        st.session_state.benchmark_results = []
    if 'benchmark_in_progress' not in st.session_state:
        st.session_state.benchmark_in_progress = False
    if 'models_cache' not in st.session_state:
        st.session_state.models_cache = None
    if 'current_status' not in st.session_state:
        st.session_state.current_status = "Ready"
    if 'current_progress' not in st.session_state:
        st.session_state.current_progress = 0
    
    # Create benchmark instance
    if 'benchmark' not in st.session_state:
        st.session_state.benchmark = OllamaBenchmark()
    
    benchmark = st.session_state.benchmark
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Ollama status
        status_col = st.columns([0.2, 0.8])
        with status_col[0]:
            is_online = benchmark.check_ollama_status()
            st.markdown(f"<span class='status-indicator {'status-online' if is_online else 'status-offline'}'></span>", unsafe_allow_html=True)
        with status_col[1]:
            st.text("Ollama Status: Online" if is_online else "Offline")
        
        if not is_online:
            st.error("⚠️ Ollama service not detected. Please start Ollama first.")
            st.stop()
        
        # Model selection
        st.markdown("---")
        st.subheader("Available Models")
        
        # Refresh models button
        if st.button("🔄 Refresh Models"):
            benchmark.clear_model_cache()
            st.session_state.models_cache = None
            st.rerun()
        
        # Get models (with caching)
        if st.session_state.models_cache is None:
            models = benchmark.get_available_models()
            st.session_state.models_cache = models
        else:
            models = st.session_state.models_cache
        
        if models:
            st.success(f"Found {len(models)} model(s)")
        else:
            st.warning("No models detected - using fallback list")
        
        selected_models = st.multiselect(
            "Select Models to Benchmark",
            models,
            default=models[:2] if len(models) >= 2 else (models[0] if models else [])
        )
        
        # Benchmark parameters
        st.markdown("---")
        st.subheader("Parameters")
        benchmark_type = st.selectbox(
            "Benchmark Type",
            ["Latency Test", "Throughput Test", "Quality Benchmark", "Custom Prompt"]
        )
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
        max_tokens = st.slider("Max Tokens", 50, 2000, 200)
        
        # Benchmark-specific settings
        if benchmark_type == "Latency Test":
            num_runs = st.slider("Number of Runs", 1, 20, 5)
            prompt = st.text_area("Test Prompt", "Explain the concept of machine learning in simple terms.", height=100)
        elif benchmark_type == "Throughput Test":
            duration = st.slider("Test Duration (seconds)", 10, 120, 30)
            prompt = st.text_area("Test Prompt", "Count to 10.", height=100)
        elif benchmark_type == "Quality Benchmark":
            test_categories = {
                "Reasoning": "If a train travels 60 miles per hour for 2.5 hours, how far does it go?",
                "Creativity": "Write a haiku about artificial intelligence.",
                "Coding": "Write a Python function to reverse a string.",
                "Factual": "What is the capital of France?"
            }
            
            selected_tests = st.multiselect(
                "Select Test Categories",
                list(test_categories.keys()),
                default=list(test_categories.keys())
            )
            prompt_dict = {k: test_categories[k] for k in selected_tests}
        else:  # Custom Prompt
            prompt = st.text_area("Enter Custom Prompt", "Your prompt here...", height=150)
        
        # Run benchmark button - disable if in progress
        run_disabled = st.session_state.benchmark_in_progress or len(selected_models) == 0
        if run_disabled and st.session_state.benchmark_in_progress:
            st.info("Benchmark in progress... Please wait.")
        elif run_disabled and len(selected_models) == 0:
            st.warning("Please select at least one model to benchmark")
        
        run_benchmark = st.button("🚀 Run Benchmark", type="primary", width=True, disabled=run_disabled)
        
        # Save results
        if st.session_state.benchmark_results and not st.session_state.benchmark_in_progress:
            if st.button("💾 Save Results"):
                filename = ResultsManager.save_results(st.session_state.benchmark_results)
                st.success(f"Results saved to `{filename}`")
    
    # Main area
    tab1, tab2, tab3 = st.tabs(["📊 Benchmark", "📈 Analytics", "📋 History"])
    
    with tab1:
        # Status display
        status_container = st.empty()
        progress_bar = st.empty()
        
        # Only show progress elements if benchmark is in progress
        if st.session_state.benchmark_in_progress:
            status_container.info(f"Status: {st.session_state.current_status}")
            progress_bar.progress(st.session_state.current_progress)
        
        if run_benchmark and selected_models and not st.session_state.benchmark_in_progress:
            st.session_state.benchmark_in_progress = True
            st.session_state.current_status = f"Starting {benchmark_type} benchmark..."
            st.session_state.current_progress = 0
            
            # Update UI immediately
            status_container.info(f"Status: {st.session_state.current_status}")
            progress_bar.progress(st.session_state.current_progress)
            
            # Results container
            all_results = []
            
            # Create a new benchmark instance for this run (to avoid queue conflicts)
            benchmark_instance = OllamaBenchmark()
            
            # Thread-safe results collection
            results_queue = queue.Queue()
            
            # Thread function - NO STREAMLIT UI CALLS HERE
            def run_benchmarks():
                try:
                    for idx, model in enumerate(selected_models):
                        # Send status update through queue
                        benchmark_instance.status_queue.put(f"Testing {model} ({idx+1}/{len(selected_models)})...")
                        
                        if benchmark_type == "Latency Test":
                            results = benchmark_instance.run_latency_benchmark(
                                model=model,
                                prompt=prompt,
                                num_runs=num_runs,
                                max_tokens=max_tokens
                            )
                        elif benchmark_type == "Throughput Test":
                            results = benchmark_instance.run_throughput_benchmark(
                                model=model,
                                prompt=prompt,
                                duration_seconds=duration
                            )
                        elif benchmark_type == "Quality Benchmark":
                            model_results = []
                            for test_name, test_prompt in prompt_dict.items():
                                benchmark_instance.status_queue.put(f"  - {test_name} test for {model}")
                                test_results = benchmark_instance.run_latency_benchmark(
                                    model=model,
                                    prompt=test_prompt,
                                    num_runs=1,
                                    max_tokens=max_tokens
                                )
                                # Update benchmark type to include test name
                                for r in test_results:
                                    r.benchmark_type = f"Quality: {test_name}"
                                model_results.extend(test_results)
                            results = model_results
                        else:  # Custom Prompt
                            results = benchmark_instance.run_latency_benchmark(
                                model=model,
                                prompt=prompt,
                                num_runs=3,
                                max_tokens=max_tokens
                            )
                        
                        results_queue.put((model, results))
                    
                    # Signal completion
                    results_queue.put(None)
                except Exception as e:
                    logger.exception("Benchmark thread failed")
                    results_queue.put(f"ERROR: {str(e)}")
            
            # Start benchmark thread
            thread = threading.Thread(target=run_benchmarks, daemon=True)
            thread.start()
            
            # Monitor progress and collect results - ALL UI UPDATES HERE
            while thread.is_alive() or not results_queue.empty():
                try:
                    # Check for status updates
                    while not benchmark_instance.status_queue.empty():
                        try:
                            status = benchmark_instance.status_queue.get_nowait()
                            st.session_state.current_status = status
                            status_container.info(f"Status: {status}")
                        except queue.Empty:
                            break
                    
                    # Check for progress updates
                    while not benchmark_instance.progress_queue.empty():
                        try:
                            progress = benchmark_instance.progress_queue.get_nowait()
                            st.session_state.current_progress = min(int(progress), 100)
                            progress_bar.progress(st.session_state.current_progress)
                        except queue.Empty:
                            break
                    
                    # Check for results
                    while not results_queue.empty():
                        try:
                            result = results_queue.get_nowait()
                            if result is None:
                                continue
                            elif isinstance(result, str) and result.startswith("ERROR"):
                                st.error(result)
                            else:
                                model_name, model_results = result
                                all_results.extend(model_results)
                        except queue.Empty:
                            break
                    
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error in progress monitoring: {e}")
                    break
            
            # Final collection of any remaining results
            while not results_queue.empty():
                try:
                    result = results_queue.get_nowait()
                    if result is None:
                        continue
                    elif isinstance(result, str) and result.startswith("ERROR"):
                        st.error(result)
                    else:
                        _, model_results = result
                        all_results.extend(model_results)
                except queue.Empty:
                    break
            
            # Final progress update
            st.session_state.current_progress = 100
            st.session_state.current_status = "✅ Benchmark completed!"
            progress_bar.progress(100)
            status_container.success(f"Status: {st.session_state.current_status}")
            st.balloons()
            
            # Update session state in main thread
            st.session_state.benchmark_results = all_results
            st.session_state.benchmark_in_progress = False
            
            # Clear queues
            while not benchmark_instance.status_queue.empty():
                try:
                    benchmark_instance.status_queue.get_nowait()
                except queue.Empty:
                    break
            while not benchmark_instance.progress_queue.empty():
                try:
                    benchmark_instance.progress_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Force rerun to update UI
            st.rerun()
        
        # Display current results
        if st.session_state.benchmark_results:
            df = ResultsManager.results_to_dataframe(st.session_state.benchmark_results)
            
            # Summary metrics
            st.subheader("Summary Metrics")
            summary_cols = st.columns(4)
            
            successful_df = df[df['success']]
            if not successful_df.empty:
                with summary_cols[0]:
                    avg_latency = successful_df['latency_ms'].mean()
                    st.metric("Average Latency", f"{avg_latency:.2f} ms")
                
                with summary_cols[1]:
                    avg_tps = successful_df['tokens_per_second'].mean()
                    st.metric("Average Throughput", f"{avg_tps:.2f} tok/s")
                
                with summary_cols[2]:
                    total_tokens = successful_df['total_tokens'].sum()
                    st.metric("Total Tokens", f"{total_tokens:,}")
                
                with summary_cols[3]:
                    success_rate = (len(successful_df) / len(df)) * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")
            else:
                st.warning("No successful benchmark results to display")
            
            # Detailed results table
            st.subheader("Detailed Results")
            if not df.empty:
                display_df = df[['model_name', 'benchmark_type', 'latency_ms', 
                               'tokens_per_second', 'total_tokens', 'success']].copy()
                display_df['latency_ms'] = display_df['latency_ms'].round(2)
                display_df['tokens_per_second'] = display_df['tokens_per_second'].round(2)
                st.dataframe(display_df, width=True)
            
            # Response samples
            with st.expander("View Response Samples"):
                for result in st.session_state.benchmark_results[:3]:
                    if result.success:
                        st.markdown(f"**{result.model_name}** - {result.benchmark_type}")
                        preview = result.response[:500] + "..." if len(result.response) > 500 else result.response
                        st.code(preview)
                        st.divider()
        else:
            st.info("No benchmark results yet. Select models and click 'Run Benchmark' to start.")
    
    with tab2:
        if st.session_state.benchmark_results:
            df = ResultsManager.results_to_dataframe(st.session_state.benchmark_results)
            
            # Filter out failed results for charts
            successful_df = df[df['success']]
            
            if not successful_df.empty:
                # Create charts
                fig_latency, fig_throughput = create_performance_charts(successful_df)
                
                if fig_latency:
                    st.plotly_chart(fig_latency, width=True)
                if fig_throughput:
                    st.plotly_chart(fig_throughput, width=True)
                
                # Model comparison
                st.subheader("Model Performance Comparison")
                comparison_data = successful_df.groupby('model_name').agg({
                    'latency_ms': ['mean', 'std'],
                    'tokens_per_second': ['mean', 'std'],
                    'success': 'mean'
                }).round(2)
                comparison_data.columns = ['_'.join(col).strip() for col in comparison_data.columns.values]
                st.dataframe(comparison_data, width=True)
            else:
                st.warning("No successful results to analyze")
        else:
            st.info("No benchmark data available. Run a benchmark first.")
    
    with tab3:
        st.subheader("Historical Results")
        
        # Load previous results
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader("Load Results File", type=['json'])
        with col2:
            if st.button("📂 Browse Saved Files"):
                import glob
                files = glob.glob("benchmark_results_*.json")
                if files:
                    selected_file = st.selectbox("Select File", files, key="file_selector")
                    if st.button("Load Selected", key="load_button"):
                        try:
                            st.session_state.benchmark_results = ResultsManager.load_results(selected_file)
                            st.success(f"Loaded results from {selected_file}")
                        except Exception as e:
                            st.error(f"Failed to load file: {e}")
        
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                st.session_state.benchmark_results = [BenchmarkResult(**item) for item in data]
                st.success("Results loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load file: {e}")
        
        # Display history
        if st.session_state.benchmark_results:
            df = ResultsManager.results_to_dataframe(st.session_state.benchmark_results)
            
            if not df.empty:
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Filter by date
                min_date = df['timestamp'].min().date()
                max_date = df['timestamp'].max().date()
                date_range = st.date_input(
                    "Filter by Date",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                
                if len(date_range) == 2:
                    mask = (df['timestamp'].dt.date >= date_range[0]) & \
                           (df['timestamp'].dt.date <= date_range[1])
                    filtered_df = df.loc[mask]
                    
                    if not filtered_df.empty:
                        st.dataframe(
                            filtered_df[['model_name', 'benchmark_type', 'timestamp', 
                                       'latency_ms', 'tokens_per_second', 'success']].sort_values('timestamp', ascending=False),
                            width=True
                        )
                        
                        # Export filtered results
                        if st.button("📥 Export Filtered Results"):
                            csv = filtered_df.to_csv(index=False)
                            st.download_button(
                                "Download CSV",
                                csv,
                                "benchmark_export.csv",
                                "text/csv"
                            )
                    else:
                        st.warning("No results match the selected date range")
            else:
                st.info("No historical data available for display.")
        else:
            st.info("No historical data loaded. Upload a results file or run a new benchmark.")

if __name__ == "__main__":
    main()