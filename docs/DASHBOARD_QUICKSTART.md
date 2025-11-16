# Python Dashboard Quick Start

## Installation

```bash
pip install -r requirements_dashboard.txt
```

## Running the Dashboard

### 1. Start Backend (Terminal 1)
```bash
python -m uvicorn backend.main_simple:app --host 0.0.0.0 --port 8000
```

### 2. Generate Demo Data (if needed)
```bash
python generate_demo_data.py
```

### 3. Start Dashboard (Terminal 2)
```bash
streamlit run dashboard.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

## Features

- **Real-time Neural Forecasts**: 1-day, 7-day, 21-day predictions
- **Interactive Charts**: Plotly visualizations with risk zones
- **Shortage Detection**: Real-time shortage events with risk scoring
- **Pattern Analysis**: Historical trends and neural insights
- **Trading Terminal UI**: Dark theme inspired by professional trading platforms
- **Auto-refresh**: Updates every 30 seconds automatically

## Dashboard Sections

1. **Top Controls**: Adjust forecast intervals, thresholds, and refresh settings
2. **AI Summary**: Neural network generated insights
3. **Metrics**: Critical/At-risk product counts and model confidence
4. **Forecast Charts**: Visual demand predictions with risk indicators
5. **Product Risk Table**: Detailed product-by-product analysis
6. **Shortage Events**: Real-time shortage detection table
7. **Pattern Analysis**: Historical patterns and trends

## Customization

Edit `dashboard.py` to:
- Change API endpoint: Modify `API_BASE_URL`
- Adjust refresh interval: Change the `time.sleep()` value
- Modify colors: Update CSS in the `st.markdown()` style section
- Add features: Extend with additional Streamlit components

## Troubleshooting

**Dashboard shows "Cannot connect to backend":**
- Ensure backend is running on port 8000
- Check `API_BASE_URL` matches your backend address
- Verify backend health: `curl http://localhost:8000/health`

**No data showing:**
- Run `python generate_demo_data.py` to create demo data
- Check backend logs for errors
- Verify data files exist in `data/` directory

**Charts not displaying:**
- Ensure Plotly is installed: `pip install plotly`
- Check browser console for JavaScript errors
- Try refreshing the page
