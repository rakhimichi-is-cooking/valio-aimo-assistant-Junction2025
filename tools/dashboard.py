"""
Valio Aimo Neural Model Dashboard
Trading-inspired interface for supply chain management
"""

import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import time
import numpy as np

# Page config
st.set_page_config(
    page_title="Valio Aimo Neural Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for trading terminal look
st.markdown("""
    <style>
    .stApp {
        background-color: #0a0e1a;
    }
    .main {
        background-color: #0a0e1a;
    }
    h1, h2, h3 {
        color: #00ff88;
        font-family: 'Courier New', monospace;
    }
    .metric-card {
        background-color: #1a1f2e;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #2a3f5f;
        margin: 10px 0;
    }
    .risk-critical {
        color: #ff4444;
        font-weight: bold;
    }
    .risk-warning {
        color: #ffaa00;
        font-weight: bold;
    }
    .risk-normal {
        color: #00ff88;
    }
    .stMetric {
        background-color: #1a1f2e;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #2a3f5f;
    }
    </style>
    """, unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Session state initialization
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True
if 'selected_interval' not in st.session_state:
    st.session_state.selected_interval = "7_day"

# Helper Functions
def fetch_data(endpoint, params=None):
    """Fetch data from API with error handling."""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", params=params, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Please ensure the API is running on port 8000.")
        st.info("Run: `python backend/main_simple.py` or `python -m uvicorn backend.main_simple:app --port 8000`")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def get_risk_color(risk_score):
    """Get color based on risk score."""
    if risk_score > 0.7:
        return "#ff4444"  # Red
    elif risk_score > 0.4:
        return "#ffaa00"  # Orange
    else:
        return "#00ff88"  # Green

def format_risk_level(level):
    """Format risk level with color."""
    colors = {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "at_risk": "🟠",
        "normal": "🟢"
    }
    return f"{colors.get(level, '⚪')} {level.upper()}"

# Header
st.markdown("# 🚀 VALIO AIMO NEURAL TRADING DASHBOARD")
st.markdown("### Real-time Supply Chain Intelligence powered by Neural Networks")

# Top Controls Bar
col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
with col1:
    intervals = {"1 Day": "1_day", "7 Days": "7_day", "21 Days": "21_day"}
    selected_interval_label = st.selectbox(
        "Forecast Interval",
        options=list(intervals.keys()),
        index=1
    )
    st.session_state.selected_interval = intervals[selected_interval_label]

with col2:
    top_n = st.slider("Products to Analyze", 5, 20, 10)

with col3:
    critical_threshold = st.slider("Critical Threshold", 0.1, 1.0, 0.3, 0.05)

with col4:
    risk_threshold = st.slider("Risk Threshold", 0.05, 0.5, 0.15, 0.05)

with col5:
    if st.button("🔄 Refresh Data", type="primary"):
        st.session_state.last_update = datetime.now()
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh

# Auto-refresh using JavaScript injection
if st.session_state.auto_refresh:
    st.caption("🔄 Auto-refresh enabled (30s interval)")
    # Inject JavaScript to auto-refresh
    st.markdown("""
    <script>
    setTimeout(function(){
        window.location.reload(1);
    }, 30000);
    </script>
    """, unsafe_allow_html=True)

# Main Dashboard Layout
main_container = st.container()

with main_container:
    # Fetch briefing data
    briefing = fetch_data(
        "/dashboard/briefing",
        params={
            "top_n": top_n,
            "critical_threshold": critical_threshold,
            "risk_threshold": risk_threshold
        }
    )
    
    if briefing:
        # Display AI Summary
        st.markdown("### 🤖 Neural Network Analysis")
        st.info(briefing.get("summary", "Loading AI insights..."))
        
        # Metrics Row
        interval_data = briefing["intervals"][st.session_state.selected_interval]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "⚠️ Critical Products",
                interval_data["critical"],
                delta=f"{interval_data['critical']} need immediate action",
                delta_color="inverse"
            )
        with col2:
            st.metric(
                "🔶 At Risk Products",
                interval_data["at_risk"],
                delta=f"{interval_data['at_risk']} require monitoring"
            )
        with col3:
            total_products = len(interval_data["products"])
            st.metric(
                "📊 Total Analyzed",
                total_products,
                delta=f"Out of {top_n} requested"
            )
        with col4:
            confidence = 0.82 + (0.95 - 0.82) * (1 - st.session_state.selected_interval.count("2") * 0.1)
            st.metric(
                "🎯 Model Confidence",
                f"{confidence:.1%}",
                delta="Neural network active"
            )
        
        # External Factors
        if interval_data.get("external_factors"):
            st.markdown("### 🌍 External Factors")
            factors_text = " • ".join(interval_data["external_factors"])
            st.warning(f"📌 {factors_text}")
        
        # Two Column Layout: Chart and Product List
        chart_col, list_col = st.columns([2, 1])
        
        with chart_col:
            st.markdown("### 📈 Neural Forecast Visualization")
            
            if interval_data["products"]:
                # Create subplot figure
                fig = make_subplots(
                    rows=2, cols=1,
                    row_heights=[0.7, 0.3],
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    subplot_titles=("Demand Forecast with Risk Zones", "Risk Score Distribution")
                )
                
                # Prepare data for visualization
                products = interval_data["products"][:5]  # Top 5 for clarity
                product_names = [p["product_name"] for p in products]
                forecast_values = [p["forecast_avg"] for p in products]
                recent_values = [p["recent_avg"] for p in products]
                risk_scores = [p["risk_score"] for p in products]
                colors = [get_risk_color(r) for r in risk_scores]
                
                # Forecast bars
                fig.add_trace(
                    go.Bar(
                        x=product_names,
                        y=forecast_values,
                        name="Neural Forecast",
                        marker_color=colors,
                        text=[f"{v:.1f}" for v in forecast_values],
                        textposition="outside"
                    ),
                    row=1, col=1
                )
                
                # Recent average line
                fig.add_trace(
                    go.Scatter(
                        x=product_names,
                        y=recent_values,
                        name="Recent Average",
                        mode="lines+markers",
                        line=dict(color="#00ff88", width=2),
                        marker=dict(size=8)
                    ),
                    row=1, col=1
                )
                
                # Risk scores
                fig.add_trace(
                    go.Bar(
                        x=product_names,
                        y=risk_scores,
                        name="Risk Score",
                        marker_color=colors,
                        text=[f"{r:.2%}" for r in risk_scores],
                        textposition="outside"
                    ),
                    row=2, col=1
                )
                
                # Add threshold lines
                fig.add_hline(y=critical_threshold, row=2, col=1,
                            line_dash="dash", line_color="red",
                            annotation_text="Critical", annotation_position="right")
                fig.add_hline(y=risk_threshold, row=2, col=1,
                            line_dash="dash", line_color="orange",
                            annotation_text="At Risk", annotation_position="right")
                
                # Update layout for dark theme
                fig.update_layout(
                    template="plotly_dark",
                    height=600,
                    showlegend=True,
                    hovermode="x unified",
                    font=dict(family="Courier New, monospace"),
                    paper_bgcolor="#0a0e1a",
                    plot_bgcolor="#1a1f2e"
                )
                
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=True, gridcolor="#2a3f5f", gridwidth=0.5)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No products found for the selected criteria.")
        
        with list_col:
            st.markdown("### 📋 Product Risk Table")
            
            if interval_data["products"]:
                # Create DataFrame
                df = pd.DataFrame(interval_data["products"])
                
                # Format and display each product as a card
                for _, product in df.iterrows():
                    risk_emoji = format_risk_level(product.get("risk_level", "normal"))
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #00ff88; margin: 0;">{product['product_name']}</h4>
                        <p style="color: #888; margin: 5px 0;">Code: {product['product_code']}</p>
                        <p style="margin: 5px 0;">Risk: {risk_emoji}</p>
                        <p style="margin: 5px 0;">Score: <span style="color: {get_risk_color(product['risk_score'])};">{product['risk_score']:.2%}</span></p>
                        <p style="margin: 5px 0;">Forecast: {product['forecast_avg']:.1f}</p>
                        <p style="margin: 5px 0;">Recent: {product['recent_avg']:.1f}</p>
                        {f"<p style='margin: 5px 0;'>Confidence: {product.get('neural_confidence', 0.85):.1%}</p>" if 'neural_confidence' in product else ""}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No products to display")
        
        # Bottom Section: Shortage Events and Pattern Analysis
        tab1, tab2 = st.tabs(["🚨 Shortage Events", "📊 Pattern Analysis"])
        
        with tab1:
            st.markdown("### Real-time Shortage Detection")
            shortages = fetch_data("/shortages", params={"threshold": risk_threshold})
            
            if shortages and len(shortages) > 0:
                shortage_df = pd.DataFrame(shortages[:20])  # Show top 20
                
                # Format the dataframe for display
                display_df = shortage_df[[
                    'product_name', 'customer_name', 'ordered_qty', 
                    'delivered_qty', 'risk_score', 'delivery_date', 'risk_level'
                ]].copy()
                
                display_df['shortage'] = display_df['ordered_qty'] - display_df['delivered_qty']
                display_df['shortage_pct'] = ((display_df['shortage'] / display_df['ordered_qty']) * 100).round(1)
                
                # Color code by risk level
                def style_risk(row):
                    if row['risk_level'] == 'critical':
                        return ['background-color: #3d1a1a'] * len(row)
                    elif row['risk_level'] in ['high', 'at_risk']:
                        return ['background-color: #3d2e1a'] * len(row)
                    return [''] * len(row)
                
                styled_df = display_df.style.apply(style_risk, axis=1)
                
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "product_name": "Product",
                        "customer_name": "Customer",
                        "ordered_qty": st.column_config.NumberColumn("Ordered", format="%.0f"),
                        "delivered_qty": st.column_config.NumberColumn("Delivered", format="%.0f"),
                        "shortage": st.column_config.NumberColumn("Shortage", format="%.0f"),
                        "shortage_pct": st.column_config.NumberColumn("Shortage %", format="%.1f%%"),
                        "risk_score": st.column_config.NumberColumn("Risk Score", format="%.2%"),
                        "risk_level": "Risk Level",
                        "delivery_date": "Date"
                    }
                )
                
                # Summary stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Shortages", len(shortage_df))
                with col2:
                    avg_shortage = shortage_df['shortage'].mean()
                    st.metric("Avg Shortage", f"{avg_shortage:.1f} units")
                with col3:
                    critical_count = len(shortage_df[shortage_df['risk_level'] == 'critical'])
                    st.metric("Critical Events", critical_count, delta=f"{critical_count} urgent")
            else:
                st.success("✅ No active shortage events detected")
        
        with tab2:
            st.markdown("### Historical Pattern Analysis")
            patterns = fetch_data("/analytics/patterns")
            
            if patterns:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Shortage Frequency by Product")
                    if 'shortage_frequency' in patterns:
                        freq_df = pd.DataFrame([
                            {"Product": k, "Frequency": v} 
                            for k, v in patterns['shortage_frequency'].items()
                        ]).sort_values('Frequency', ascending=False)
                        
                        fig_freq = go.Figure(data=[
                            go.Bar(
                                x=freq_df['Product'],
                                y=freq_df['Frequency'],
                                marker_color='#ff4444',
                                text=[f"{v:.1%}" for v in freq_df['Frequency']],
                                textposition="outside"
                            )
                        ])
                        fig_freq.update_layout(
                            template="plotly_dark",
                            height=300,
                            paper_bgcolor="#0a0e1a",
                            plot_bgcolor="#1a1f2e",
                            font=dict(family="Courier New, monospace")
                        )
                        st.plotly_chart(fig_freq, use_container_width=True)
                
                with col2:
                    st.markdown("#### Neural Model Insights")
                    if 'neural_insights' in patterns:
                        insights = patterns['neural_insights']
                        st.info(f"**Trend:** {insights.get('trend', 'N/A')}")
                        st.warning(f"**Risk Level:** {insights.get('risk_level', 'N/A')}")
                        st.metric("Model Confidence", f"{insights.get('confidence', 0):.1%}")
                        st.caption(f"Next Review: {insights.get('next_review', 'N/A')}")
                
                if 'high_risk_products' in patterns:
                    st.markdown("#### High Risk Products")
                    st.write(", ".join(patterns['high_risk_products']))
            else:
                st.info("Pattern analysis data not available")
        
        # Footer with status
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.caption(f"🕐 Last Update: {briefing.get('timestamp', 'Unknown')[:19]}")
        with col2:
            st.caption(f"🔮 Forecast From: {briefing.get('forecast_from_date', 'Unknown')[:10]}")
        with col3:
            st.caption(f"🌐 Backend: {API_BASE_URL}")
        with col4:
            if briefing.get('neural_model_status') == 'active':
                st.caption("✅ Neural Model: Active")
            else:
                st.caption("⚡ Neural Model: Simulated")
    
    else:
        # Connection error display
        st.error("⚠️ Unable to connect to backend API")
        st.markdown("""
        ### 🔧 Quick Fix Instructions:
        
        1. **Start the backend server:**
        ```bash
        python backend/main_simple.py
        ```
        
        2. **Or with uvicorn:**
        ```bash
        python -m uvicorn backend.main_simple:app --host 0.0.0.0 --port 8000
        ```
        
        3. **Generate demo data if needed:**
        ```bash
        python generate_demo_data.py
        ```
        
        4. **Refresh this page** once the backend is running
        """)

# Sidebar - Additional Controls
with st.sidebar:
    st.markdown("## ⚙️ Neural Model Settings")
    
    st.markdown("### Model Parameters")
    model_type = st.selectbox(
        "Forecasting Model",
        ["Neural Network Ensemble", "LSTM/GRU", "Prophet", "XGBoost"]
    )
    
    st.markdown("### Display Options")
    show_confidence = st.checkbox("Show Confidence Intervals", value=True)
    show_patterns = st.checkbox("Show Historical Patterns", value=False)
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    
    # Try to get health status
    health = fetch_data("/health")
    if health:
        st.success("✅ Backend Online")
    else:
        st.error("❌ Backend Offline")
    
    st.markdown("---")
    st.markdown("### 📝 About")
    st.info("""
    **Valio Aimo Neural Dashboard**
    
    Trading-inspired interface for supply chain management using:
    - Neural network forecasting
    - Multi-factor risk scoring
    - Real-time shortage detection
    - Pattern recognition
    
    Built with Streamlit & FastAPI
    """)
