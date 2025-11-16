from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import requests
from .config import BACKEND_HOST, BACKEND_PORT, FRONTEND_ORIGIN
from .data_loader import (
    load_sales_and_deliveries,
    load_replacement_orders,
    load_purchases,
    load_product_data,
    load_product_dict,
)
from .shortage_engine import compute_shortage_events
from .replacement_engine import suggest_substitutes
from .aimo_agent import AimoAgent
from .forecasting_engine import ForecastingEngine
from .pattern_analysis import PatternAnalyzer
from .cache_manager import get_cache_manager, get_performance_monitor
from .sms_service import get_sms_service

app = FastAPI(title="Valio Aimo Shortage Assistant v0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ShortageEvent(BaseModel):
    customer_name: str
    customer_id: str
    sku: str
    product_name: str
    ordered_qty: float
    delivered_qty: float
    delivery_date: str
    risk_score: float
    suggested_substitutes: Optional[List[Dict[str, Any]]] = None

    # Advanced risk scoring fields (optional)
    composite_risk_score: Optional[float] = None
    severity_score: Optional[float] = None
    historical_score: Optional[float] = None
    customer_score: Optional[float] = None
    forecast_score: Optional[float] = None
    seasonal_score: Optional[float] = None
    risk_level: Optional[str] = None


class AimoResponse(BaseModel):
    summary: str
    internal_notes: str
    customer_message: str
    call_script: str
    language: str


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/shortages", response_model=List[ShortageEvent])
def list_shortages(
    threshold: float = Query(0.5, description="Minimum risk score threshold"),
    use_advanced_scoring: bool = Query(True, description="Use multi-factor risk scoring"),
    enable_forecasting: bool = Query(False, description="Enable demand forecasting (slower)"),
) -> List[ShortageEvent]:
    """
    Get list of shortage events with optional advanced risk scoring.

    Args:
        threshold: Minimum risk score to include (0-1)
        use_advanced_scoring: Enable advanced multi-factor risk scoring
        enable_forecasting: Enable demand forecasting (increases processing time)

    Returns:
        List of shortage events sorted by risk score
    """
    sales = load_sales_and_deliveries()
    replacements = load_replacement_orders()
    purchases = load_purchases()
    products = load_product_data()

    # Compute shortage events with optional advanced scoring
    base_events = compute_shortage_events(
        sales,
        replacements,
        purchases,
        use_advanced_scoring=use_advanced_scoring,
        enable_forecasting=enable_forecasting
    )

    enriched_events: List[ShortageEvent] = []
    for evt in base_events:
        # Use composite risk score if available, otherwise fall back to basic risk score
        current_risk = evt.get("composite_risk_score", evt.get("risk_score", 0))

        if current_risk < threshold:
            continue

        # Add suggested substitutes (with optional neural embeddings)
        substitutes = suggest_substitutes(
            evt["sku"],
            products,
            k=3,
            use_advanced_matching=True,
            use_neural_embeddings=False  # Can be made configurable via query param
        )
        evt["suggested_substitutes"] = substitutes

        enriched_events.append(ShortageEvent(**evt))

    return enriched_events


@app.post("/shortages/message", response_model=AimoResponse)
def generate_message(
    event: ShortageEvent,
    language: str = "en",
    use_two_stage: bool = Query(False, description="Use two-stage LLM reasoning (Analyzer + Decision Maker)")
) -> AimoResponse:
    """
    Generate AI-powered messages for a shortage event.

    Args:
        event: Shortage event
        language: Target language (en, fi, sv)
        use_two_stage: Enable two-stage LLM reasoning for improved accuracy

    Returns:
        AIMO response with summary, notes, customer message, and call script
    """
    agent = AimoAgent()
    try:
        if use_two_stage:
            # Two-stage reasoning: Analyzer → Decision Maker
            resp = agent.generate_for_event_two_stage(event.model_dump(), language=language)
        else:
            # Single-stage (traditional)
            resp = agent.generate_for_event(event.model_dump(), language=language)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AimoResponse(**resp)


@app.get("/analytics/patterns")
def get_patterns() -> Dict[str, Any]:
    """
    Get historical pattern analysis.

    Returns:
        Comprehensive pattern insights including:
        - Shortage frequency by product
        - Seasonal patterns
        - Customer behavior patterns
        - Trend analysis
        - High-risk combinations
    """
    try:
        sales = load_sales_and_deliveries()
        analyzer = PatternAnalyzer()
        insights = analyzer.generate_insights(sales)
        return insights
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pattern analysis failed: {str(e)}")


@app.get("/analytics/forecast/{sku}")
def get_forecast(
    sku: str,
    periods: int = Query(30, description="Forecast periods (days)"),
    method: str = Query("prophet", description="Forecasting method (prophet, arima, xgboost, ensemble)")
) -> Dict[str, Any]:
    """
    Get demand forecast for a specific product.

    Args:
        sku: Product SKU
        periods: Number of days to forecast
        method: Forecasting algorithm

    Returns:
        Forecast data with predictions and confidence intervals
    """
    try:
        sales = load_sales_and_deliveries()

        # Filter to product
        product_df = sales[sales['product_code'] == sku]

        if len(product_df) < 14:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for product {sku}. Need at least 14 days of history."
            )

        # Initialize forecasting engine
        engine = ForecastingEngine(method=method)

        # Prepare time series
        ts_data = engine.prepare_time_series(product_df)

        # Generate forecast
        forecast = engine.forecast(ts_data, periods=periods)

        # Convert to serializable format
        forecast_dict = forecast.to_dict(orient='records')

        return {
            "sku": sku,
            "method": method,
            "periods": periods,
            "forecast": forecast_dict,
            "historical_data_points": len(ts_data)
        }

    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Forecasting method '{method}' not available. Install required dependencies."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")


@app.get("/analytics/stats")
def get_stats() -> Dict[str, Any]:
    """
    Get performance statistics and cache info.

    Returns:
        Performance metrics and cache statistics
    """
    monitor = get_performance_monitor()
    cache = get_cache_manager()

    return {
        "performance": monitor.get_all_stats(),
        "cache_dir": str(cache.cache_dir)
    }


@app.post("/cache/clear")
def clear_cache() -> Dict[str, str]:
    """
    Clear all caches.

    Returns:
        Success message
    """
    try:
        cache = get_cache_manager()
        cache.clear()
        return {"status": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")


class SMSRequest(BaseModel):
    to_number: str = Field(..., description="Phone number (e.g., +358442605413)")
    customer_name: str = Field(..., description="Customer name")
    shortages: List[Dict[str, Any]] = Field(..., description="List of shortage events")
    include_substitutes: bool = Field(True, description="Mention replacements")


@app.post("/sms/send_shortage_alert")
def send_shortage_sms(request: SMSRequest) -> Dict[str, Any]:
    """
    Send SMS alert about product shortages via Twilio

    Args:
        request: SMS request with phone number, customer name, and shortages

    Returns:
        {'success': bool, 'message_sid': str, 'body': str, 'error': str}
    """
    sms_service = get_sms_service()

    result = sms_service.send_shortage_alert(
        to_number=request.to_number,
        customer_name=request.customer_name,
        shortages=request.shortages,
        include_substitutes=request.include_substitutes
    )

    if not result['success']:
        raise HTTPException(status_code=500, detail=result.get('error', 'SMS send failed'))

    return result


@app.get("/dashboard/briefing")
def get_dashboard_briefing(
    top_n: int = Query(15, description="Number of top products to analyze"),
    critical_threshold: float = Query(0.4, description="Critical risk threshold (0-1)"),
    risk_threshold: float = Query(0.15, description="At-risk threshold (0-1)")
) -> Dict[str, Any]:
    """
    Get dashboard briefing with 1, 7, 21 day interval forecasts.

    Fast version for hackathon demo - uses simple trend analysis with external factors.
    Factors in: Finland holidays, weekends, major events, market conditions, seasonality.

    Returns:
        Dashboard briefing with multi-interval forecasts and AI-generated summary
    """
    try:
        sales = load_sales_and_deliveries()
        products = load_product_dict()  # Use dict for easy lookup

        # Load external factors
        try:
            external_factors = pd.read_csv('data/finland_external_factors.csv')
            external_factors['date'] = pd.to_datetime(external_factors['date'])
        except:
            external_factors = None

        # Get top products by volume
        top_products = sales['product_code'].value_counts().head(top_n).index.tolist()

        # Calculate forecast start date (end of data + 1 day)
        sales['order_created_date'] = pd.to_datetime(sales['order_created_date'])
        forecast_start_date = sales['order_created_date'].max() + pd.Timedelta(days=1)

        briefing = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "forecast_from_date": forecast_start_date.isoformat(),
            "intervals": {
                "1_day": {"critical": 0, "at_risk": 0, "products": [], "external_factors": []},
                "7_day": {"critical": 0, "at_risk": 0, "products": [], "external_factors": []},
                "21_day": {"critical": 0, "at_risk": 0, "products": [], "external_factors": []}
            },
            "summary": ""
        }

        # Get external factors for each interval
        if external_factors is not None:
            for interval, days in [("1_day", 1), ("7_day", 7), ("21_day", 21)]:
                interval_end = forecast_start_date + pd.Timedelta(days=days-1)
                interval_factors = external_factors[
                    (external_factors['date'] >= forecast_start_date) &
                    (external_factors['date'] <= interval_end)
                ]

                # Identify key factors in this interval
                if len(interval_factors) > 0:
                    factors_list = []
                    if interval_factors['is_holiday'].sum() > 0:
                        holidays = interval_factors[interval_factors['is_holiday'] == 1]['holiday_name'].tolist()
                        factors_list.append(f"Holidays: {', '.join(holidays[:3])}")

                    if interval_factors['is_major_event'].sum() > 0:
                        events = interval_factors[interval_factors['is_major_event'] == 1]['major_event'].unique().tolist()
                        factors_list.append(f"Events: {', '.join(events[:2])}")

                    weekend_count = interval_factors['is_weekend'].sum()
                    factors_list.append(f"{weekend_count} weekend days")

                    briefing["intervals"][interval]["external_factors"] = factors_list

        # Fast trend-based analysis (no heavy forecasting)
        for product_code in top_products:
            product_df = sales[sales['product_code'] == product_code].copy()

            if len(product_df) < 7:
                continue

            # Sort by date
            product_df['order_created_date'] = pd.to_datetime(product_df['order_created_date'])
            product_df = product_df.sort_values('order_created_date')

            # Calculate daily demand
            daily_demand = product_df.groupby('order_created_date')['order_qty'].sum()

            if len(daily_demand) < 7:
                continue

            # Simple trend analysis
            recent_7d = daily_demand.tail(7).mean()
            recent_3d = daily_demand.tail(3).mean()
            overall_avg = daily_demand.mean()

            # Simple "forecast" based on recent trend
            trend = (recent_3d - recent_7d) / (recent_7d + 1)

            # Risk scoring for each interval
            for interval, weight, days in [("1_day", 1.0, 1), ("7_day", 0.7, 7), ("21_day", 0.5, 21)]:
                # Simulate forecast (recent avg adjusted by trend)
                forecast_avg = recent_7d * (1 + trend * weight)

                # Apply external factor modifiers if available
                if external_factors is not None:
                    interval_end = forecast_start_date + pd.Timedelta(days=days-1)
                    interval_factors = external_factors[
                        (external_factors['date'] >= forecast_start_date) &
                        (external_factors['date'] <= interval_end)
                    ]
                    if len(interval_factors) > 0:
                        avg_modifier = interval_factors['combined_demand_modifier'].mean()
                        forecast_avg = forecast_avg * avg_modifier

                # Risk: if trending down or forecast below overall avg
                shortage_risk = max(0, 1.0 - (forecast_avg / (overall_avg + 1)))

                # Use provided thresholds (adjustable via API)
                interval_critical = critical_threshold
                interval_risk = risk_threshold

                if shortage_risk > interval_critical:
                    briefing["intervals"][interval]["critical"] += 1
                    risk_level = "critical"
                elif shortage_risk > interval_risk:
                    briefing["intervals"][interval]["at_risk"] += 1
                    risk_level = "at_risk"
                else:
                    continue

                # Add product to interval
                # Get product name from dict
                product_info = products.get(str(product_code), {})
                product_name = product_info.get("name") or product_info.get("product_name") or f"Product {product_code}"

                briefing["intervals"][interval]["products"].append({
                    "product_code": str(product_code),
                    "product_name": product_name,
                    "forecast_avg": round(forecast_avg, 2),
                    "recent_avg": round(recent_7d, 2),
                    "risk_score": round(shortage_risk, 3),
                    "risk_level": risk_level
                })

        # Generate AI summary using AIMO agent
        external_context = ""
        if briefing["intervals"]["1_day"]["external_factors"]:
            external_context = f"\nExternal factors (next 1-7 days): {', '.join(briefing['intervals']['7_day']['external_factors'])}"

        summary_prompt = f"""Generate a concise executive briefing (2-3 sentences) for this Finland supply chain forecast:

Forecasting from: {forecast_start_date.strftime('%B %d, %Y')}

1 Day: {briefing['intervals']['1_day']['critical']} critical, {briefing['intervals']['1_day']['at_risk']} at risk
7 Day: {briefing['intervals']['7_day']['critical']} critical, {briefing['intervals']['7_day']['at_risk']} at risk
21 Day: {briefing['intervals']['21_day']['critical']} critical, {briefing['intervals']['21_day']['at_risk']} at risk
{external_context}

Provide brief actionable insights considering Finland market conditions and external factors."""

        try:
            agent = AimoAgent()
            response = requests.post(
                f"{agent.api_url}",
                json={
                    "model": agent.model,
                    "messages": [{"role": "user", "content": summary_prompt}],
                    "temperature": 0.5,
                    "max_tokens": 150
                },
                timeout=10
            )
            if response.status_code == 200:
                briefing["summary"] = response.json()["choices"][0]["message"]["content"]
            else:
                briefing["summary"] = "Supply chain monitoring active. Check critical items for immediate action."
        except Exception as e:
            print(f"AI summary failed: {e}")
            briefing["summary"] = "Supply chain monitoring active. Check critical items for immediate action."

        return briefing

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard briefing failed: {str(e)}")


class ChatQuery(BaseModel):
    query: str = Field(..., description="User's natural language query")


@app.post("/chat/query")
def chat_query(request: ChatQuery) -> Dict[str, Any]:
    """
    Process natural language query using conversational agent.
    
    Args:
        request: Chat query with user's question
        
    Returns:
        {'answer': str, 'code': str, 'data': Any}
    """
    # Quick check: Is LM Studio running?
    try:
        lm_check = requests.get("http://localhost:1234/v1/models", timeout=2)
        if lm_check.status_code != 200:
            return {
                "answer": "LM Studio is not responding. Please start LM Studio and load a model, then try again.",
                "code": "",
                "data": None,
                "error": "LM Studio not available"
            }
    except requests.exceptions.RequestException:
        return {
            "answer": "LM Studio is not running. Please start LM Studio on port 1234 and load a model (e.g., Qwen, Llama). The AI chat feature requires a local LLM.",
            "code": "",
            "data": None,
            "error": "LM Studio connection failed"
        }
    
    # LM Studio is available, proceed with agent
    try:
        from .conversational_agent import ConversationalAgent
        
        agent = ConversationalAgent()
        result = agent.answer_question(request.query)
        
        return {
            "answer": result.get('answer', 'No answer generated'),
            "code": result.get('code_generated', ''),
            "data": result.get('execution_result', {}).get('result'),
            "error": result.get('error')
        }
    except Exception as e:
        return {
            "answer": f"Error processing query: {str(e)}",
            "code": "",
            "data": None,
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True)
