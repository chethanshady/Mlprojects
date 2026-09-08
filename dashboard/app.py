import streamlit as st
import pandas as pd
import requests
import time
import os
import plotly.express as px
from PIL import Image

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
ARTIFACTS_DIR = os.getenv(
    "ARTIFACTS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
)

st.set_page_config(
    page_title="Fraud Detection Monitor",
    page_icon="🔍",
    layout="wide"
)

def check_api_health():
    """Check if the backend API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200 and response.json().get("model_loaded", False)
    except requests.exceptions.RequestException:
        return False

def get_model_info():
    """Fetch model metadata from API."""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=2)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    return None

def make_prediction(data_dict):
    """Call the API for a single prediction."""
    try:
        response = requests.post(f"{API_URL}/predict", json=data_dict, timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    return None

def make_batch_prediction(df):
    """Call the API for batch predictions."""
    try:
        payload = {"transactions": df.to_dict(orient="records")}
        response = requests.post(f"{API_URL}/predict/batch", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["predictions"]
    except requests.exceptions.RequestException:
        pass
    return None

# Sidebar navigation
st.sidebar.title("🔍 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio("Select View:", ["Live Monitor", "Model Performance", "Feature Importance", "Transaction Explorer"])

api_available = check_api_health()

if not api_available:
    st.sidebar.error("⚠️ Backend API is unavailable or model not loaded. Please start the FastAPI server.")
else:
    st.sidebar.success("✅ API Connected")

if page == "Live Monitor":
    st.title("🔴 Live Transaction Monitor")
    st.markdown("Simulate real-time incoming transactions and monitor fraud detection alerts.")
    
    # Initialize session state for live monitor
    if 'live_data' not in st.session_state:
        st.session_state.live_data = pd.DataFrame(columns=["Time", "Amount", "Fraud Probability", "Risk Level", "Status"])
        st.session_state.metrics = {"total": 0, "fraud": 0, "resp_time": []}
        
    col1, col2, col3, col4 = st.columns(4)
    metric_total = col1.empty()
    metric_fraud = col2.empty()
    metric_rate = col3.empty()
    metric_time = col4.empty()
    
    def update_metrics():
        total = st.session_state.metrics["total"]
        fraud = st.session_state.metrics["fraud"]
        rate = (fraud / total * 100) if total > 0 else 0.0
        avg_time = sum(st.session_state.metrics["resp_time"]) / len(st.session_state.metrics["resp_time"]) if st.session_state.metrics["resp_time"] else 0.0
        
        metric_total.metric("Total Transactions", total)
        metric_fraud.metric("Frauds Detected", fraud)
        metric_rate.metric("Detection Rate", f"{rate:.2f}%")
        metric_time.metric("Avg Response Time", f"{avg_time:.1f} ms")

    update_metrics()
    
    st.markdown("---")
    
    if api_available:
        if st.button("Start Simulation"):
            # Load some sample data for simulation (in a real app, this would be a stream)
            data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "creditcard.csv")
            try:
                df_sim = pd.read_csv(data_path).sample(100)
                st.write("Simulating transactions...")
                progress_bar = st.progress(0)
                
                table_placeholder = st.empty()
                
                for i, (_, row) in enumerate(df_sim.iterrows()):
                    # Call API
                    pred = make_prediction(row.to_dict())
                    if pred:
                        st.session_state.metrics["total"] += 1
                        if pred["is_fraud"]:
                            st.session_state.metrics["fraud"] += 1
                        st.session_state.metrics["resp_time"].append(pred["processing_time_ms"])
                        
                        new_row = {
                            "Time": row["Time"],
                            "Amount": row["Amount"],
                            "Fraud Probability": f"{pred['fraud_probability']:.4f}",
                            "Risk Level": pred["risk_level"],
                            "Status": "🔴 FRAUD" if pred["is_fraud"] else "🟢 LEGIT"
                        }
                        
                        # Add to session state dataframe
                        st.session_state.live_data = pd.concat(
                            [pd.DataFrame([new_row]), st.session_state.live_data]
                        ).head(20).reset_index(drop=True)
                        
                        table_placeholder.dataframe(st.session_state.live_data, use_container_width=True)
                        update_metrics()
                        
                    progress_bar.progress((i + 1) / len(df_sim))
                    time.sleep(0.1)  # Faster simulation
                
                st.success("Simulation Complete")
            except FileNotFoundError:
                st.error(f"Sample data not found at {data_path}. Please place a sample dataset there for simulation.")
    else:
        st.warning("Cannot start simulation without a connection to the API.")
        
    if not st.session_state.live_data.empty:
        st.subheader("Recent Transactions")
        def style_status_fallback(val):
            color = '#ff4b4b' if 'FRAUD' in str(val) else '#00cc96'
            return f'color: {color}'
        st.dataframe(st.session_state.live_data.style.map(style_status_fallback, subset=['Status']), use_container_width=True)

elif page == "Model Performance":
    st.title("📊 Model Performance")
    
    info = get_model_info()
    if info:
        with st.expander("Model Metadata", expanded=True):
            col1, col2 = st.columns(2)
            col1.write(f"**Model Type:** {info['model_type']}")
            col1.write(f"**Training Date:** {info['training_date']}")
            col2.write(f"**Classification Threshold:** {info['threshold']}")
            st.write("**Evaluation Metrics:**")
            st.json(info['metrics'])
    
    st.markdown("---")
    st.subheader("Evaluation Plots")
    
    # Check if plots exist in artifacts
    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    pr_path = os.path.join(ARTIFACTS_DIR, "pr_curve.png")
    roc_path = os.path.join(ARTIFACTS_DIR, "roc_curve.png")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists(cm_path):
            st.image(Image.open(cm_path), caption="Confusion Matrix", use_column_width=True)
        else:
            st.info("Confusion matrix plot not found in artifacts.")
            
        if os.path.exists(pr_path):
            st.image(Image.open(pr_path), caption="Precision-Recall Curve", use_column_width=True)
        else:
            st.info("PR curve plot not found in artifacts.")
            
    with col2:
        if os.path.exists(roc_path):
            st.image(Image.open(roc_path), caption="ROC Curve", use_column_width=True)
        else:
            st.info("ROC curve plot not found in artifacts.")

elif page == "Feature Importance":
    st.title("🎯 Feature Importance")
    st.markdown("Understand which features drive the model's fraud predictions.")
    
    fi_path = os.path.join(ARTIFACTS_DIR, "feature_importance.png")
    if os.path.exists(fi_path):
        st.image(Image.open(fi_path), caption="Top Features by Importance", use_column_width=True)
    else:
        st.info("Feature importance plot not found in artifacts.")

elif page == "Transaction Explorer":
    st.title("🔎 Transaction Explorer")
    st.markdown("Upload a batch of transactions (CSV) to analyze them for fraud.")
    
    uploaded_file = st.file_uploader("Upload Transactions CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} transactions.")
            
            if st.button("Run Batch Prediction") and api_available:
                with st.spinner("Analyzing transactions..."):
                    # We might need to handle large files by chunking, but for demo keep it simple
                    max_rows = min(1000, len(df))
                    df_subset = df.head(max_rows)
                    if len(df) > max_rows:
                        st.warning(f"File too large. Analyzing only the first {max_rows} rows.")
                        
                    predictions = make_batch_prediction(df_subset)
                    
                    if predictions:
                        st.success("Analysis Complete!")
                        
                        # Merge predictions with dataframe
                        results = []
                        for i, p in enumerate(predictions):
                            results.append({
                                "Fraud Probability": p["fraud_probability"],
                                "Is Fraud": p["is_fraud"],
                                "Risk Level": p["risk_level"]
                            })
                            
                        res_df = pd.concat([df_subset.reset_index(drop=True), pd.DataFrame(results)], axis=1)
                        
                        st.subheader("Results")
                        
                        # Filtering
                        risk_filter = st.selectbox("Filter by Risk Level", ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
                        
                        if risk_filter != "ALL":
                            display_df = res_df[res_df["Risk Level"] == risk_filter]
                        else:
                            display_df = res_df
                            
                        st.write(f"Showing {len(display_df)} transactions:")
                        
                        def highlight_fraud(s):
                            is_fraud = s["Is Fraud"]
                            return ['background-color: #ffcccc' if is_fraud else '' for _ in s]
                            
                        st.dataframe(display_df.style.apply(highlight_fraud, axis=1), use_container_width=True)
                        
                        # Plot breakdown
                        fig = px.pie(res_df, names="Risk Level", title="Risk Level Distribution", hole=0.4)
                        st.plotly_chart(fig)
                        
                    else:
                        st.error("Batch prediction failed. Check API logs.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
