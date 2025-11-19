# 🚀 Ollama Model Benchmark Suite

A professional benchmarking suite for evaluating and comparing the performance of AI models served by [Ollama](https://ollama.com/). This Streamlit application provides comprehensive latency, throughput, and quality analysis with beautiful visualizations and historical tracking capabilities.

## ✨ Features

✅ **Multi-Model Benchmarking** - Compare performance across multiple Ollama models simultaneously  
✅ **Comprehensive Metrics** - Measure latency, throughput (tokens/sec), and resource utilization  
✅ **Quality Assessment** - Evaluate models on reasoning, creativity, coding, and factual tasks  
✅ **Real-time Monitoring** - Visual progress tracking with CPU/memory usage monitoring  
✅ **Professional Visualizations** - Interactive Plotly charts for performance analysis  
✅ **Persistent Results** - Save/load benchmark results to/from JSON files  
✅ **Historical Analysis** - Filter and compare results across different time periods  
✅ **Responsive UI** - Professional dashboard with gradient styling and intuitive controls  

## 📸 Tab Overview

The application is organized into three specialized tabs, each serving a distinct purpose in the benchmarking workflow:

Here’s a **concise, clean, professional rewrite** of all three sections while keeping all key information:

---

## **1. 📊 Benchmark Tab — Real-Time Performance Testing**

**Purpose:** Configure, run, and monitor benchmarks with immediate visual feedback.

**Core Features:**

* **Live Status Indicator:** Real-time Ollama connectivity with color-coded status.
* **Progress Tracking:** Animated progress bars and percentage completion.
* **Instant Metrics:** Four metric cards showing:

  * Avg. Latency
  * Avg. Throughput
  * Total Tokens
  * Success Rate
* **Interactive Results Table:** Per-run metrics with pass/fail indicators.
* **Response Viewer:** Expandable samples to inspect model outputs.
* **Model Refresh:** One-click model list update from Ollama.
* **Config Sidebar:** Choose benchmark type and adjust parameters.

**Workflow:**

1. Confirm Ollama connection.
2. Select models.
3. Choose benchmark type:

   * Latency (runs + prompt)
   * Throughput (duration + prompt)
   * Quality (Reasoning, Creativity, Coding, Factual)
   * Custom Prompt
4. Click **Run Benchmark** (enabled only when valid).
5. Watch real-time execution.
6. Review metrics and run outcomes.

**Value:** A unified, reload-free workspace for configuring tests, running them, and viewing results instantly.

---

## **2. 📈 Analytics Tab — Insightful Performance Visualization**

**Purpose:** Turn benchmark data into clear visual insights.

**Core Features:**

* **Latency Distribution:** Box plots with medians, IQR, outliers, and individual run points.
* **Throughput vs Latency:** Combined bar+line chart showing correlation between speed and responsiveness.
* **Stat Summary Table:** Mean, standard deviation, and success rates for each model.
* **Model Ranking:** Automatic ordering by performance.
* **Interactive Charts:** Hover details, enable/disable models, zoom/pan, export as PNG/SVG.

**Analytical Benefits:**

* Spot outliers and performance variance.
* Compare speed vs responsiveness.
* Evaluate model stability.
* Understand statistical reliability across runs.

**Value:** Reveals trends and relationships that raw numbers alone can’t show—ideal for informed model selection.

---

## **3. 📋 History Tab — Long-Term Performance Tracking**

**Purpose:** Store and analyze benchmark results over time.

**Core Features:**

* **History Management:** Upload, browse, and load saved JSON result files.
* **Date Filtering:** Calendar range selection and timeline visualization.
* **Detailed Log Table:** Timestamped entries with metrics, formatting, and error indicators.
* **Export Tools:** Filtered CSV exports with proper naming.
* **Compare Results:** Side-by-side comparisons across dates or datasets.

**Use Cases:**

* Track improvements after updates.
* Measure effect of hardware changes.
* Maintain audit trail for model selection.
* Build long-term baselines.
* Detect performance drift.

**Value:** Turns benchmarking into a continuous monitoring system rather than a one-off test.

---

## ⚙️ Installation

### Prerequisites
- [Ollama](https://ollama.com/) installed and running (v0.1.27+)
- Python 3.8 or higher

### Setup Instructions
**Setup python Environment**
# Install dependencies
**pip install -r requirements.txt**
# Run the application
**streamlit run app.py**


## 🧪 Usage Guide

1. **Start Ollama Service**  
   Ensure Ollama is running before launching the application:
   ```bash
   ollama serve
   ```

2. **Launch the Application**  
   Run the Streamlit app:
   ```bash
   streamlit run streamlit_ollama_benchmark_final.py
   ```

3. **Configure Benchmark**  
   In the sidebar:
   - Verify Ollama status indicator (green = online)
   - Select models to benchmark
   - Choose benchmark type:
     - **Latency Test**: Multiple runs measuring response time
     - **Throughput Test**: Fixed duration measuring tokens/second
     - **Quality Benchmark**: Pre-defined tests for reasoning/creativity/coding
     - **Custom Prompt**: Your own test prompt
   - Adjust parameters (temperature, max tokens, etc.)

4. **Run Benchmark**  
   Click the "🚀 Run Benchmark" button and monitor progress in real-time

5. **Analyze Results**  
   - **📊 Benchmark Tab**: View immediate results and response samples
   - **📈 Analytics Tab**: Explore visualizations and performance comparisons
   - **📋 History Tab**: Load historical results and track performance trends

6. **Save Results**  
   Click "💾 Save Results" to export benchmark data to JSON for future analysis