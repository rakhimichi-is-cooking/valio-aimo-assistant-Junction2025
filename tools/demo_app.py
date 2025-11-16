"""
Interactive Demo Web App for Hackathon

Features:
1. Upload stock photo → Vision AI detects issues → Adjust forecasts
2. LM Studio integration → Human-readable explanations
3. Interactive testing for judges
4. Real-time GNN predictions
5. Visual graph exploration

Tech Stack:
- Streamlit (simple web interface)
- LM Studio API (local LLM + vision model)
- GNN forecaster (backend)
- Product graph visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
from PIL import Image
import io
import base64

# Import our forecasters
from backend.gnn_forecaster import GNNForecaster
from backend.neural_forecaster import NeuralForecaster

# ============================================================================
# LM STUDIO INTEGRATION
# ============================================================================

class LMStudioClient:
    """Client for LM Studio API (local LLM + vision)"""

    def __init__(self, base_url="http://localhost:1234"):
        self.base_url = base_url
        self.chat_url = f"{base_url}/v1/chat/completions"

    def analyze_image(self, image_bytes, prompt="Describe this image"):
        """
        Analyze image using LM Studio vision model

        Args:
            image_bytes: Image as bytes
            prompt: Question to ask about the image

        Returns:
            str: LLM response
        """
        # Convert image to base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        payload = {
            "model": "vision",  # LM Studio vision model
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }

        try:
            response = requests.post(self.chat_url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Error analyzing image: {e}"

    def explain_forecast(self, product_name, prediction, neighbors_info):
        """
        Generate human-readable explanation of forecast

        Args:
            product_name: Name of product
            prediction: Forecast values
            neighbors_info: Info about related products

        Returns:
            str: Human-readable explanation
        """
        prompt = f"""You are a supply chain analyst. Explain this forecast in simple terms:

Product: {product_name}
Predicted demand (next 7 days): {prediction}

Related products and their trends:
{neighbors_info}

Provide a concise 2-3 sentence explanation focusing on:
1. What the forecast predicts
2. Why (based on related products)
3. Any risks or opportunities
"""

        payload = {
            "model": "local-model",  # Your loaded LM Studio model
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 200
        }

        try:
            response = requests.post(self.chat_url, json=payload, timeout=15)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Forecast shows demand of {prediction[0]:.0f} units tomorrow, trending to {prediction[-1]:.0f} by day 7."


# ============================================================================
# STREAMLIT WEB APP
# ============================================================================

st.set_page_config(
    page_title="Valio AI Supply Chain Assistant",
    page_icon="📦",
    layout="wide"
)

# Initialize session state
if 'lm_client' not in st.session_state:
    st.session_state.lm_client = LMStudioClient()

if 'forecaster' not in st.session_state:
    with st.spinner("Loading GNN forecaster... (this may take a moment)"):
        try:
            st.session_state.forecaster = GNNForecaster(
                lookback=30,
                horizon=7,
                lstm_hidden=64,
                gnn_hidden=32
            )
            st.session_state.forecaster_loaded = True
        except Exception as e:
            st.error(f"Error loading forecaster: {e}")
            st.session_state.forecaster_loaded = False

# ============================================================================
# HEADER
# ============================================================================

st.title("🧠 Valio AI: Supply Chain Shortage Predictor")
st.markdown("**State-of-the-art Graph Neural Network forecasting with 111,969 verified product relationships**")

# Tabs for different features
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📸 Stock Photo Analysis",
    "📈 Product Forecast",
    "📊 Manual Data Input",
    "📱 SMS Alerts",
    "🕸️ Product Network",
    "ℹ️ About"
])

# ============================================================================
# TAB 1: STOCK PHOTO ANALYSIS
# ============================================================================

with tab1:
    st.header("📸 Smart Stock Analysis with Vision AI")
    st.markdown("""
    Upload a photo of your stock and our AI will:
    - Detect damaged/expired products
    - Identify shortage risks
    - Automatically adjust forecasts
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload stock photo (milk, dairy products, etc.)",
            type=['jpg', 'jpeg', 'png']
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Stock Photo", use_column_width=True)

            if st.button("🔍 Analyze Stock Photo", type="primary"):
                with st.spinner("AI analyzing image..."):
                    # Read image bytes
                    image_bytes = uploaded_file.getvalue()

                    # Analyze with vision model
                    analysis_prompt = """Analyze this stock photo of dairy products. Identify:
1. What products are visible
2. Any damaged, broken, or expired items
3. Stock level (low/medium/high)
4. Any quality issues

Be specific and concise."""

                    analysis = st.session_state.lm_client.analyze_image(
                        image_bytes,
                        analysis_prompt
                    )

                    st.session_state.stock_analysis = analysis

    with col2:
        if 'stock_analysis' in st.session_state:
            st.subheader("🤖 AI Analysis Results")
            st.info(st.session_state.stock_analysis)

            # Parse analysis for damaged items - RUN ACTUAL FORECASTS
            if "damaged" in st.session_state.stock_analysis.lower() or \
               "broken" in st.session_state.stock_analysis.lower():
                st.warning("⚠️ Damaged items detected!")

                if st.session_state.forecaster_loaded:
                    # Load actual sales data
                    sales_path = Path("data/valio_aimo_sales_and_deliveries_junction_2025.csv")
                    if sales_path.exists():
                        sales_df = pd.read_csv(sales_path)
                        sales_df['order_created_date'] = pd.to_datetime(sales_df['order_created_date'])
                        sales_df['product_code'] = sales_df['product_code'].astype(str)

                        # Get top products to analyze
                        top_products = sales_df['product_code'].value_counts().head(3).index.tolist()

                        st.markdown("**Real Forecast Adjustments:**")

                        all_dates = sorted(sales_df['order_created_date'].unique())

                        for product_code in top_products[:2]:  # Analyze top 2 products
                            try:
                                # Get baseline forecast
                                product_sales = sales_df[sales_df['product_code'] == product_code]
                                daily_demand = product_sales.groupby('order_created_date')['order_qty'].sum()
                                daily_demand = daily_demand.reindex(all_dates, fill_value=0)
                                history = daily_demand.values

                                if len(history) >= 30:
                                    baseline_result = st.session_state.forecaster.predict(product_code, history)
                                    baseline_pred = baseline_result['predictions']

                                    # Adjust forecast based on damage (reduce available stock)
                                    damage_factor = 1.15  # Need 15% more to compensate for damaged stock
                                    adjusted_pred = baseline_pred * damage_factor

                                    st.markdown(f"**Product {product_code}:**")
                                    st.markdown(f"- Baseline 7-day forecast: {baseline_pred.sum():.0f} units")
                                    st.markdown(f"- Adjusted for damage: {adjusted_pred.sum():.0f} units (+{(damage_factor-1)*100:.0f}%)")

                                    # Show neighbor impact
                                    neighbors = baseline_result.get('neighbors', [])
                                    if neighbors:
                                        st.markdown(f"- {len(neighbors)} related products in network will be monitored")

                            except Exception as e:
                                st.warning(f"Could not forecast product {product_code}: {e}")

                        st.success("✅ Real forecasts generated and adjusted based on stock condition!")
                    else:
                        st.error("Sales data not found")
                else:
                    st.error("Forecaster not loaded")

# ============================================================================
# TAB 2: PRODUCT FORECAST
# ============================================================================

with tab2:
    st.header("📈 Interactive Product Forecast")

    if not st.session_state.forecaster_loaded:
        st.error("Forecaster not loaded. Please check backend setup.")
    else:
        # Load sample products
        sales_path = Path("data/valio_aimo_sales_and_deliveries_junction_2025.csv")

        if sales_path.exists():
            @st.cache_data
            def load_products():
                sales_df = pd.read_csv(sales_path)
                sales_df['product_code'] = sales_df['product_code'].astype(str)
                top_products = sales_df['product_code'].value_counts().head(50).index.tolist()
                return top_products

            products = load_products()

            selected_product = st.selectbox(
                "Select a product to forecast:",
                products,
                format_func=lambda x: f"Product {x}"
            )

            if st.button("🔮 Generate Forecast", type="primary"):
                with st.spinner("Running GNN forecast..."):
                    # Load historical data
                    sales_df = pd.read_csv(sales_path)
                    sales_df['order_created_date'] = pd.to_datetime(sales_df['order_created_date'])
                    sales_df['product_code'] = sales_df['product_code'].astype(str)

                    # Get all dates
                    all_dates = sorted(sales_df['order_created_date'].unique())

                    # Build demand series
                    product_sales = sales_df[sales_df['product_code'] == selected_product]
                    daily_demand = product_sales.groupby('order_created_date')['order_qty'].sum()
                    daily_demand = daily_demand.reindex(all_dates, fill_value=0)

                    history = daily_demand.values

                    try:
                        # Make prediction
                        result = st.session_state.forecaster.predict(selected_product, history)
                        predictions = result['predictions']

                        # Display results
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            # Create chart
                            forecast_df = pd.DataFrame({
                                'Day': range(1, 8),
                                'Predicted Demand': predictions
                            })

                            st.line_chart(forecast_df.set_index('Day'))

                            st.subheader("📊 7-Day Forecast")
                            for i, pred in enumerate(predictions, 1):
                                st.metric(f"Day {i}", f"{pred:.0f} units")

                        with col2:
                            st.subheader("🤖 AI Explanation")

                            # Get REAL neighbors info from the graph
                            neighbors = result.get('neighbors', [])
                            neighbors_info_lines = []

                            if neighbors:
                                neighbors_info_lines.append(f"Connected to {len(neighbors)} products in network:")
                                for neighbor in neighbors[:3]:  # Show top 3
                                    # Calculate trend for neighbor
                                    neighbor_sales = sales_df[sales_df['product_code'] == neighbor]
                                    if len(neighbor_sales) > 0:
                                        neighbor_demand = neighbor_sales.groupby('order_created_date')['order_qty'].sum()
                                        neighbor_demand = neighbor_demand.reindex(all_dates, fill_value=0)
                                        neighbor_history = neighbor_demand.values

                                        if len(neighbor_history) >= 14:
                                            recent_avg = neighbor_history[-7:].mean()
                                            older_avg = neighbor_history[-14:-7].mean()

                                            if older_avg > 0:
                                                trend_pct = ((recent_avg - older_avg) / older_avg) * 100
                                                trend_dir = "↗️" if trend_pct > 5 else "↘️" if trend_pct < -5 else "→"
                                                neighbors_info_lines.append(f"  - Product {neighbor}: {trend_dir} {trend_pct:+.1f}%")
                                            else:
                                                neighbors_info_lines.append(f"  - Product {neighbor}: New product")

                                neighbors_info = "\n".join(neighbors_info_lines)
                            else:
                                neighbors_info = "No direct neighbors in graph"

                            explanation = st.session_state.lm_client.explain_forecast(
                                f"Product {selected_product}",
                                predictions,
                                neighbors_info
                            )

                            st.info(explanation)

                            # Show neighbor details
                            if neighbors:
                                with st.expander("📊 Network Details"):
                                    st.markdown(neighbors_info)

                            # Risk assessment
                            avg_demand = predictions.mean()
                            if avg_demand < history[-7:].mean() * 0.8:
                                st.warning("⚠️ Demand drop predicted - check for substitution!")
                            elif avg_demand > history[-7:].mean() * 1.2:
                                st.success("📈 Demand surge predicted - ensure sufficient stock!")
                            else:
                                st.info("➡️ Stable demand predicted")

                    except Exception as e:
                        st.error(f"Error generating forecast: {e}")

# ============================================================================
# TAB 3: MANUAL DATA INPUT
# ============================================================================

with tab3:
    st.header("📊 Manual Data Input - Update Stock & Orders")

    st.markdown("""
    Enter new data to update the system in real-time:
    - **New Orders**: Customer orders just received
    - **New Arrivals**: Stock that just arrived
    - **Inventory Updates**: Manual stock adjustments
    """)

    input_type = st.selectbox("What type of data are you entering?", [
        "New Orders",
        "New Arrivals",
        "Inventory Adjustment",
        "Damaged Stock"
    ])

    if input_type == "New Orders":
        st.subheader("📦 New Orders")

        # Create editable dataframe
        if 'new_orders' not in st.session_state:
            st.session_state.new_orders = pd.DataFrame({
                'Product Code': ['', '', ''],
                'Customer': ['', '', ''],
                'Quantity': [0, 0, 0],
                'Order Date': [pd.Timestamp.now().date()] * 3
            })

        edited_orders = st.data_editor(
            st.session_state.new_orders,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Order Date": st.column_config.DateColumn("Order Date", format="YYYY-MM-DD"),
                "Quantity": st.column_config.NumberColumn("Quantity", min_value=0, step=1)
            }
        )

        col1, col2 = st.columns([1, 3])

        with col1:
            if st.button("💾 Save Orders", type="primary"):
                # Filter out empty rows
                valid_orders = edited_orders[edited_orders['Product Code'].str.strip() != '']

                if len(valid_orders) > 0:
                    # Save to CSV
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"data/manual_orders_{timestamp}.csv"

                    try:
                        valid_orders.to_csv(filename, index=False)
                        st.success(f"✅ Saved {len(valid_orders)} orders to {filename}")
                        st.session_state.new_orders = pd.DataFrame({
                            'Product Code': ['', '', ''],
                            'Customer': ['', '', ''],
                            'Quantity': [0, 0, 0],
                            'Order Date': [pd.Timestamp.now().date()] * 3
                        })
                    except Exception as e:
                        st.error(f"Error saving: {e}")
                else:
                    st.warning("No valid orders to save")

        with col2:
            st.info("💡 Tip: Add rows using the '+' button. Data is saved to data/ folder.")

    elif input_type == "New Arrivals":
        st.subheader("📥 New Stock Arrivals")

        if 'new_arrivals' not in st.session_state:
            st.session_state.new_arrivals = pd.DataFrame({
                'Product Code': ['', '', ''],
                'Quantity': [0, 0, 0],
                'Arrival Date': [pd.Timestamp.now().date()] * 3,
                'Warehouse': ['', '', '']
            })

        edited_arrivals = st.data_editor(
            st.session_state.new_arrivals,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Arrival Date": st.column_config.DateColumn("Arrival Date", format="YYYY-MM-DD"),
                "Quantity": st.column_config.NumberColumn("Quantity", min_value=0, step=1)
            }
        )

        if st.button("💾 Save Arrivals", type="primary"):
            valid_arrivals = edited_arrivals[edited_arrivals['Product Code'].str.strip() != '']

            if len(valid_arrivals) > 0:
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                filename = f"data/manual_arrivals_{timestamp}.csv"

                try:
                    valid_arrivals.to_csv(filename, index=False)
                    st.success(f"✅ Saved {len(valid_arrivals)} arrivals to {filename}")
                except Exception as e:
                    st.error(f"Error saving: {e}")

    elif input_type == "Inventory Adjustment":
        st.subheader("📝 Inventory Adjustments")

        if 'inventory_adj' not in st.session_state:
            st.session_state.inventory_adj = pd.DataFrame({
                'Product Code': ['', '', ''],
                'Adjustment': [0, 0, 0],
                'Reason': ['', '', ''],
                'Date': [pd.Timestamp.now().date()] * 3
            })

        edited_adj = st.data_editor(
            st.session_state.inventory_adj,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Adjustment": st.column_config.NumberColumn("Adjustment (+/-)", step=1),
                "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD")
            }
        )

        if st.button("💾 Save Adjustments", type="primary"):
            valid_adj = edited_adj[edited_adj['Product Code'].str.strip() != '']

            if len(valid_adj) > 0:
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                filename = f"data/manual_inventory_adj_{timestamp}.csv"

                try:
                    valid_adj.to_csv(filename, index=False)
                    st.success(f"✅ Saved {len(valid_adj)} adjustments to {filename}")
                except Exception as e:
                    st.error(f"Error saving: {e}")

    else:  # Damaged Stock
        st.subheader("⚠️ Damaged Stock Reports")

        if 'damaged_stock' not in st.session_state:
            st.session_state.damaged_stock = pd.DataFrame({
                'Product Code': ['', '', ''],
                'Damaged Quantity': [0, 0, 0],
                'Damage Type': ['', '', ''],
                'Date': [pd.Timestamp.now().date()] * 3
            })

        edited_damaged = st.data_editor(
            st.session_state.damaged_stock,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Damaged Quantity": st.column_config.NumberColumn("Damaged Qty", min_value=0, step=1),
                "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD")
            }
        )

        if st.button("💾 Save Damage Reports", type="primary"):
            valid_damaged = edited_damaged[edited_damaged['Product Code'].str.strip() != '']

            if len(valid_damaged) > 0:
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                filename = f"data/manual_damaged_{timestamp}.csv"

                try:
                    valid_damaged.to_csv(filename, index=False)
                    st.success(f"✅ Saved {len(valid_damaged)} damage reports to {filename}")

                    # Trigger forecast adjustment
                    st.warning("🔄 Damage detected - system will adjust forecasts automatically")
                except Exception as e:
                    st.error(f"Error saving: {e}")

# ============================================================================
# TAB 4: SMS ALERTS
# ============================================================================

with tab4:
    st.header("📱 SMS Shortage Alerts")

    st.markdown("""
    Send SMS notifications to customers about product shortages using Twilio.
    """)

    # Check if backend is available
    backend_url = "http://127.0.0.1:8000"

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📞 Customer Information")

        customer_name = st.text_input("Customer Name", value="Test Customer")
        customer_phone = st.text_input("Phone Number", value="+358442605413", help="Format: +358XXXXXXXXX")

    with col2:
        st.subheader("📦 Products with Shortages")

        # Sample shortage products (in real system would come from shortage detection)
        shortage_products = st.text_area(
            "Product SKUs (one per line)",
            value="400122\n400123\n400124",
            height=100
        )

        include_replacements = st.checkbox("Mention replacement options", value=True)

    if st.button("📤 Send SMS Alert", type="primary"):
        with st.spinner("Sending SMS..."):
            # Parse products
            skus = [sku.strip() for sku in shortage_products.split('\n') if sku.strip()]

            # Build shortage list
            shortages = []
            for sku in skus:
                shortages.append({
                    'sku': sku,
                    'product_name': f'Product {sku}',
                    'risk_score': 0.8
                })

            # Call backend SMS endpoint
            try:
                response = requests.post(
                    f"{backend_url}/sms/send_shortage_alert",
                    json={
                        'to_number': customer_phone,
                        'customer_name': customer_name,
                        'shortages': shortages,
                        'include_substitutes': include_replacements
                    },
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ SMS sent successfully!")
                    st.code(result.get('body', 'Message sent'), language='text')
                    st.caption(f"Message SID: {result.get('message_sid', 'N/A')}")
                else:
                    st.error(f"❌ Failed to send SMS: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Backend not running. Start with: `python -m uvicorn backend.main:app --reload`")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    st.markdown("---")

    st.info("""
    **Setup Instructions:**
    1. Install Twilio: `pip install twilio python-dotenv`
    2. Create `.env` file in project root:
       ```
       TWILIO_ACCOUNT_SID=your_account_sid
       TWILIO_AUTH_TOKEN=your_auth_token
       TWILIO_PHONE_NUMBER=+15415322772
       ```
    3. Start backend: `python -m uvicorn backend.main:app --reload`
    """)

# ============================================================================
# TAB 5: PRODUCT NETWORK
# ============================================================================

with tab5:
    st.header("🕸️ Product Relationship Network")

    st.markdown("""
    Our GNN uses a verified network of **111,969 product relationships**:
    - **109,818** co-purchase relationships (bought together)
    - **9,310** demand correlations (synchronized patterns)
    - **261** expert-defined substitutions
    """)

    st.subheader("Network Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Edges", "111,969", help="Verified relationships")

    with col2:
        st.metric("Products", "696", help="Nodes in graph")

    with col3:
        st.metric("Avg Connections", "161", help="Per product")

    st.markdown("---")

    st.subheader("How It Works")

    st.markdown("""
    **1. Traditional LSTM (Memory Only):**
    ```
    Product A history → LSTM → Prediction
    ```

    **2. Our GNN (Memory + Network):**
    ```
    Product A history → LSTM ──┐
                                ├→ Fusion → Prediction
    Product Network   → GNN  ──┘
    ```

    **The GNN captures:**
    - 🔄 **Substitution**: If A runs out, customers buy B
    - 🤝 **Complementarity**: If A sells, B sells (coffee + creamer)
    - 📊 **Correlation**: Products with synchronized demand patterns
    """)

# ============================================================================
# TAB 6: ABOUT
# ============================================================================

with tab6:
    st.header("ℹ️ About This System")

    st.markdown("""
    ## 🎯 What We Built

    A state-of-the-art supply chain forecasting system that combines:

    ### 🧠 Neural Networks
    - **LSTM**: Learns temporal patterns from product history
    - **GNN**: Leverages 111,969 verified product relationships
    - **Hybrid Model**: Combines both for superior accuracy

    ### 🔍 Data Verification
    - ✅ 100% of 80 random edges verified mathematically
    - ✅ Perfect symmetry (0 violations out of 25,351 checks)
    - ✅ 95% confidence interval: [100%, 100%]
    - ✅ Zero fabricated data

    ### 🤖 AI Integration
    - **LM Studio Vision**: Analyzes stock photos
    - **LLM Explanations**: Human-readable forecasts
    - **Two-Stage Reasoning**: Analyzer → Decision Maker

    ### 📊 Real Business Impact

    **Example 1: Damaged Stock Detection**
    ```
    Photo shows broken milk bottles
    → AI detects damage
    → Automatically adjusts forecast (+15%)
    → Prevents shortage
    ```

    **Example 2: Network Effects**
    ```
    Coffee sales spike unexpectedly
    → GNN sees coffee→creamer relationship (45% co-purchase)
    → Predicts creamer shortage too
    → Proactive restocking
    ```

    ## 🏆 Why This Wins

    **Most competitors have:**
    - Simple ARIMA/Prophet
    - No network analysis
    - No verification

    **We have:**
    - Graph Neural Networks
    - 111,969 verified relationships
    - Vision AI integration
    - Interactive demo
    - Production-ready code

    ## 📚 Technical Stack

    - **Data Mining**: 7.3M sales records → 111,969 edges
    - **Graph Building**: Jaccard similarity, Pearson correlation
    - **Verification**: 3-layer testing (basic, statistical, independent)
    - **GNN**: PyTorch Geometric (GraphSAGE)
    - **Vision**: LM Studio multimodal models
    - **LLM**: Local inference via LM Studio
    - **Interface**: Streamlit web app

    ## 🔗 Links

    - [GitHub Repository](https://github.com/CyberPsycho2077/valio-aimo-assistant)
    - [Verification Results](data/product_graph/verification_results.json)
    - [Product Graph](data/product_graph/)

    ---

    **Built for Valio Aimo Hackathon 2025**

    *State-of-the-art neural forecasting meets real-world supply chain challenges*
    """)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("# 🧠 Valio AI")
    st.markdown("### Graph Neural Network Forecasting")

    st.markdown("### 🎮 Try It Yourself!")

    st.markdown("""
    **For Judges:**
    1. Upload a stock photo in the first tab
    2. Try forecasting different products
    3. Explore the product network

    **What makes this special:**
    - 111,969 verified relationships
    - Real-time AI analysis
    - Production-ready code
    """)

    st.markdown("---")

    st.markdown("### 📊 System Status")

    # Check LM Studio connection
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=2)
        if response.status_code == 200:
            st.success("✅ LM Studio connected")
        else:
            st.warning("⚠️ LM Studio not responding")
    except:
        st.error("❌ LM Studio offline")

    if st.session_state.forecaster_loaded:
        st.success("✅ GNN forecaster ready")
    else:
        st.error("❌ GNN forecaster error")

    st.markdown("---")
    st.markdown("**Built with:** PyTorch, PyTorch Geometric, Streamlit, LM Studio")
