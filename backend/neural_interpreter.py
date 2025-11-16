"""
Neural Model Data Interpreter using Qwen3-v1-30B
Converts complex neural forecasting outputs into human-understandable insights.

Pipeline:
1. Neural Data Extraction: Extract predictions, confidence intervals, training metrics
2. Context Enrichment: Add historical patterns, seasonal info, related products
3. LLM Interpretation: Use Qwen3-v1-30B to generate human insights
4. Human Output: Structured, actionable explanations for business users
"""

import json
import requests
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from .config import LMSTUDIO_BASE_URL

@dataclass
class NeuralModelOutput:
    """Structured representation of neural model predictions"""
    predictions: np.ndarray  # Future predictions
    confidence_lower: np.ndarray  # Lower confidence bound
    confidence_upper: np.ndarray  # Upper confidence bound
    historical_data: np.ndarray  # Recent historical values
    training_metrics: Dict[str, float]  # Loss, MAE, etc.
    model_type: str  # "LSTM+GRU", "GNN", etc.
    forecast_horizon: int  # Number of days predicted
    product_code: str  # Product identifier
    
@dataclass
class BusinessContext:
    """Business context for enriching neural output"""
    product_name: str
    category: str
    seasonal_patterns: Dict[str, Any]
    related_products: List[str]
    historical_volatility: float
    avg_daily_demand: float
    recent_trends: Dict[str, Any]

@dataclass 
class HumanInsight:
    """Human-readable interpretation of neural output"""
    executive_summary: str
    forecast_explanation: str
    confidence_assessment: str
    business_recommendations: List[str]
    risk_alerts: List[str]
    technical_notes: str
    
class NeuralDataExtractor:
    """Extracts and structures data from neural model outputs"""
    
    def extract_from_forecast_df(self, forecast_df: pd.DataFrame, 
                                product_code: str, model_type: str = "LSTM+GRU") -> NeuralModelOutput:
        """
        Extract structured data from forecast DataFrame
        
        Args:
            forecast_df: Output from neural_forecaster.forecast()
            product_code: Product identifier
            model_type: Type of neural model used
            
        Returns:
            Structured neural model output
        """
        return NeuralModelOutput(
            predictions=forecast_df['yhat'].values,
            confidence_lower=forecast_df['yhat_lower'].values,
            confidence_upper=forecast_df['yhat_upper'].values,
            historical_data=np.array([]),  # Will be filled by context enricher
            training_metrics={"loss": 0.0, "mae": 0.0},  # Will be filled if available
            model_type=model_type,
            forecast_horizon=len(forecast_df),
            product_code=product_code
        )
    
    def extract_from_gnn_output(self, gnn_result: Dict[str, Any]) -> NeuralModelOutput:
        """Extract data from GNN forecaster output"""
        predictions = gnn_result['predictions']
        product_code = gnn_result['product_code']
        
        # GNN doesn't provide confidence intervals by default, estimate them
        pred_std = np.std(predictions) if len(predictions) > 1 else predictions[0] * 0.1
        confidence_lower = predictions - 1.96 * pred_std
        confidence_upper = predictions + 1.96 * pred_std
        
        return NeuralModelOutput(
            predictions=predictions,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            historical_data=np.array([]),
            training_metrics={"method": "gnn"},
            model_type="Graph Neural Network",
            forecast_horizon=len(predictions),
            product_code=product_code
        )

class BusinessContextEnricher:
    """Enriches neural output with business context and historical patterns"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
    
    def enrich_neural_output(self, neural_output: NeuralModelOutput, 
                           historical_df: Optional[pd.DataFrame] = None) -> Tuple[NeuralModelOutput, BusinessContext]:
        """
        Enrich neural output with business context
        
        Args:
            neural_output: Raw neural model output
            historical_df: Historical sales data for context
            
        Returns:
            Enhanced neural output and business context
        """
        product_code = neural_output.product_code
        
        # Get business context
        context = self._get_business_context(product_code, historical_df)
        
        # Enrich neural output with historical data
        if historical_df is not None:
            product_history = historical_df[
                historical_df['product_code'] == product_code
            ]['order_qty'].values[-30:]  # Last 30 days
            
            neural_output.historical_data = product_history
        
        return neural_output, context
    
    def _get_business_context(self, product_code: str, 
                            historical_df: Optional[pd.DataFrame] = None) -> BusinessContext:
        """Get business context for a product"""
        
        if historical_df is not None:
            product_data = historical_df[historical_df['product_code'] == product_code]
            
            avg_demand = product_data['order_qty'].mean() if len(product_data) > 0 else 0
            volatility = product_data['order_qty'].std() / avg_demand if avg_demand > 0 else 0
            
            # Calculate recent trends (last 7 vs previous 7 days)
            recent_7 = product_data.tail(7)['order_qty'].mean() if len(product_data) >= 7 else avg_demand
            previous_7 = product_data.iloc[-14:-7]['order_qty'].mean() if len(product_data) >= 14 else avg_demand
            trend_change = (recent_7 - previous_7) / previous_7 if previous_7 > 0 else 0
            
            recent_trends = {
                "week_over_week_change": trend_change,
                "recent_avg": recent_7,
                "direction": "increasing" if trend_change > 0.05 else "decreasing" if trend_change < -0.05 else "stable"
            }
        else:
            avg_demand = 100  # Default
            volatility = 0.3
            recent_trends = {"direction": "stable", "recent_avg": avg_demand}
        
        # Try to load product information
        product_info = self._load_product_info(product_code)
        
        return BusinessContext(
            product_name=product_info.get("name", f"Product {product_code}"),
            category=product_info.get("category", "Dairy"),
            seasonal_patterns=self._detect_seasonal_patterns(historical_df, product_code),
            related_products=self._find_related_products(product_code),
            historical_volatility=volatility,
            avg_daily_demand=avg_demand,
            recent_trends=recent_trends
        )
    
    def _load_product_info(self, product_code: str) -> Dict[str, Any]:
        """Load product information from catalog"""
        catalog_path = self.data_dir / "product_features" / "product_catalog.json"
        if catalog_path.exists():
            try:
                with open(catalog_path, 'r') as f:
                    catalog = json.load(f)
                    return catalog.get(str(product_code), {})
            except:
                pass
        return {}
    
    def _detect_seasonal_patterns(self, historical_df: Optional[pd.DataFrame], 
                                 product_code: str) -> Dict[str, Any]:
        """Detect seasonal patterns in historical data"""
        if historical_df is None:
            return {"detected": False}
        
        product_data = historical_df[historical_df['product_code'] == product_code]
        if len(product_data) < 30:  # Need at least 30 days
            return {"detected": False}
        
        # Simple seasonality detection - day of week patterns
        if 'order_created_date' in product_data.columns:
            product_data['day_of_week'] = pd.to_datetime(product_data['order_created_date']).dt.dayofweek
            dow_avg = product_data.groupby('day_of_week')['order_qty'].mean()
            
            # Check if there's significant variation by day of week
            dow_std = dow_avg.std()
            dow_mean = dow_avg.mean()
            coefficient_of_variation = dow_std / dow_mean if dow_mean > 0 else 0
            
            if coefficient_of_variation > 0.3:  # More than 30% variation
                peak_day = dow_avg.idxmax()
                low_day = dow_avg.idxmin()
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                return {
                    "detected": True,
                    "type": "weekly",
                    "peak_day": days[peak_day],
                    "low_day": days[low_day],
                    "variation": f"{coefficient_of_variation:.1%}"
                }
        
        return {"detected": False}
    
    def _find_related_products(self, product_code: str) -> List[str]:
        """Find related products from product graph"""
        graph_path = self.data_dir / "product_graph" / "product_graph.pkl"
        if graph_path.exists():
            try:
                import pickle
                with open(graph_path, 'rb') as f:
                    product_graph = pickle.load(f)
                    related = list(product_graph.get(str(product_code), []))[:5]  # Top 5
                    return [str(p) for p in related]
            except:
                pass
        return []

class QwenNeuralInterpreter:
    """Interprets neural model output using Qwen3-v1-30B LLM"""
    
    def __init__(self, base_url: str = None, model_name: str = "qwen3-v1-30b"):
        self.base_url = base_url or LMSTUDIO_BASE_URL
        self.api_url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        self.model_name = model_name
    
    def interpret_neural_output(self, neural_output: NeuralModelOutput, 
                               context: BusinessContext) -> HumanInsight:
        """
        Generate human-readable interpretation of neural model output
        
        Args:
            neural_output: Structured neural model predictions
            context: Business context and enrichment data
            
        Returns:
            Human-readable insights and recommendations
        """
        
        # Prepare structured data for LLM
        llm_input = self._prepare_llm_input(neural_output, context)
        
        # Generate interpretation
        response = self._call_qwen(llm_input)
        
        # Parse and structure response
        return self._parse_llm_response(response)
    
    def _prepare_llm_input(self, neural_output: NeuralModelOutput, 
                          context: BusinessContext) -> str:
        """Prepare structured input for Qwen LLM"""
        
        # Calculate key statistics
        pred_mean = np.mean(neural_output.predictions)
        pred_trend = "increasing" if neural_output.predictions[-1] > neural_output.predictions[0] else \
                    "decreasing" if neural_output.predictions[-1] < neural_output.predictions[0] else "stable"
        
        confidence_width = np.mean(neural_output.confidence_upper - neural_output.confidence_lower)
        uncertainty_level = "high" if confidence_width > pred_mean * 0.5 else \
                          "moderate" if confidence_width > pred_mean * 0.2 else "low"
        
        historical_avg = np.mean(neural_output.historical_data) if len(neural_output.historical_data) > 0 else context.avg_daily_demand
        forecast_vs_historical = (pred_mean / historical_avg - 1) * 100 if historical_avg > 0 else 0
        
        # Prepare comprehensive input
        llm_input = f"""
NEURAL MODEL INTERPRETATION REQUEST

## Product Information
- Product Code: {neural_output.product_code}
- Product Name: {context.product_name}
- Category: {context.category}
- Model Used: {neural_output.model_type}

## Forecast Results
- Forecast Horizon: {neural_output.forecast_horizon} days
- Predictions: {neural_output.predictions.tolist()}
- Average Predicted Demand: {pred_mean:.1f} units/day
- Trend Direction: {pred_trend}
- Confidence Intervals: Lower={neural_output.confidence_lower.tolist()}, Upper={neural_output.confidence_upper.tolist()}
- Uncertainty Level: {uncertainty_level}

## Business Context
- Historical Average Demand: {historical_avg:.1f} units/day
- Forecast vs Historical: {forecast_vs_historical:+.1f}%
- Historical Volatility: {context.historical_volatility:.2f}
- Recent Trends: {context.recent_trends}
- Seasonal Patterns: {context.seasonal_patterns}
- Related Products: {context.related_products}

## Model Performance
- Training Metrics: {neural_output.training_metrics}

TASK:
As a supply chain expert, provide a comprehensive interpretation of this neural network forecast. Structure your response as JSON with these sections:

1. executive_summary: 2-3 sentence overview for executives
2. forecast_explanation: Detailed explanation of what the model predicts and why
3. confidence_assessment: How reliable are these predictions and what affects uncertainty
4. business_recommendations: Specific actionable recommendations (3-5 items)
5. risk_alerts: Potential issues or concerns to monitor
6. technical_notes: Brief technical context about the model and its approach

Focus on business impact, actionable insights, and risk management. Use clear, non-technical language for business sections.
"""
        return llm_input
    
    def _call_qwen(self, prompt: str) -> str:
        """Call Qwen3-v1-30B API"""
        
        system_prompt = """You are an expert supply chain analyst and data scientist specializing in interpreting neural network forecasts for business users. You translate complex AI predictions into clear, actionable business insights.

Your expertise includes:
- Neural network forecasting (LSTM, GRU, Graph Neural Networks)  
- Supply chain demand patterns and seasonality
- Risk assessment and uncertainty quantification
- Business impact analysis and recommendations

Provide practical, business-focused interpretations that help decision-makers understand:
1. What the AI predicts and why
2. How confident they should be in these predictions  
3. What actions they should take
4. What risks to monitor

Always structure responses as valid JSON and use clear, professional language."""

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 2048,
            "stream": False
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error calling Qwen API: {e}")
            return self._generate_fallback_response(prompt)
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate fallback response if LLM call fails"""
        return json.dumps({
            "executive_summary": "Neural network forecast generated successfully. Manual review recommended.",
            "forecast_explanation": "The model has generated predictions based on historical patterns and product relationships.",
            "confidence_assessment": "Confidence assessment requires manual review of model outputs.",
            "business_recommendations": [
                "Review forecast against business expectations",
                "Monitor actual demand vs predictions",
                "Adjust inventory planning accordingly"
            ],
            "risk_alerts": ["LLM interpretation service unavailable - manual review required"],
            "technical_notes": "Neural model executed successfully. LLM interpretation service experienced connectivity issues."
        })
    
    def _parse_llm_response(self, response: str) -> HumanInsight:
        """Parse LLM response into structured insights"""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                return HumanInsight(
                    executive_summary=parsed.get("executive_summary", ""),
                    forecast_explanation=parsed.get("forecast_explanation", ""), 
                    confidence_assessment=parsed.get("confidence_assessment", ""),
                    business_recommendations=parsed.get("business_recommendations", []),
                    risk_alerts=parsed.get("risk_alerts", []),
                    technical_notes=parsed.get("technical_notes", "")
                )
            else:
                # Fallback: treat entire response as explanation
                return HumanInsight(
                    executive_summary=response[:200] + "..." if len(response) > 200 else response,
                    forecast_explanation=response,
                    confidence_assessment="Assessment not available",
                    business_recommendations=["Review forecast manually"],
                    risk_alerts=["Manual review recommended"],
                    technical_notes="Response parsing failed"
                )
                
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return HumanInsight(
                executive_summary="Forecast generated, interpretation failed",
                forecast_explanation="Neural model completed successfully",
                confidence_assessment="Manual review required",
                business_recommendations=["Review forecast outputs"],
                risk_alerts=["Interpretation service error"],
                technical_notes=f"Parsing error: {str(e)}"
            )

class NeuralInterpretationPipeline:
    """Complete pipeline for neural model interpretation"""
    
    def __init__(self, data_dir: str = "data", llm_base_url: str = None):
        self.extractor = NeuralDataExtractor()
        self.enricher = BusinessContextEnricher(data_dir)
        self.interpreter = QwenNeuralInterpreter(llm_base_url)
    
    def interpret_forecast(self, forecast_df: pd.DataFrame, product_code: str,
                          model_type: str = "LSTM+GRU", 
                          historical_df: Optional[pd.DataFrame] = None) -> HumanInsight:
        """
        Complete interpretation pipeline for neural forecasts
        
        Args:
            forecast_df: Neural model forecast output
            product_code: Product identifier
            model_type: Type of neural model
            historical_df: Historical data for context
            
        Returns:
            Human-readable insights and recommendations
        """
        
        # Step 1: Extract structured data
        neural_output = self.extractor.extract_from_forecast_df(
            forecast_df, product_code, model_type
        )
        
        # Step 2: Enrich with business context
        neural_output, context = self.enricher.enrich_neural_output(
            neural_output, historical_df
        )
        
        # Step 3: Generate human insights
        insights = self.interpreter.interpret_neural_output(neural_output, context)
        
        return insights
    
    def interpret_gnn_result(self, gnn_result: Dict[str, Any], 
                            historical_df: Optional[pd.DataFrame] = None) -> HumanInsight:
        """Interpret GNN forecaster results"""
        
        # Step 1: Extract from GNN output
        neural_output = self.extractor.extract_from_gnn_output(gnn_result)
        
        # Step 2: Enrich with context
        neural_output, context = self.enricher.enrich_neural_output(
            neural_output, historical_df
        )
        
        # Step 3: Generate insights
        insights = self.interpreter.interpret_neural_output(neural_output, context)
        
        return insights

# Example usage functions
def interpret_neural_forecast(forecast_df: pd.DataFrame, product_code: str,
                            historical_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Quick function to interpret a neural forecast
    
    Args:
        forecast_df: Output from neural forecaster
        product_code: Product to interpret
        historical_df: Historical sales data for context
        
    Returns:
        Dictionary with human-readable insights
    """
    pipeline = NeuralInterpretationPipeline()
    insights = pipeline.interpret_forecast(forecast_df, product_code, historical_df=historical_df)
    
    return {
        "executive_summary": insights.executive_summary,
        "forecast_explanation": insights.forecast_explanation,
        "confidence_assessment": insights.confidence_assessment,
        "business_recommendations": insights.business_recommendations,
        "risk_alerts": insights.risk_alerts,
        "technical_notes": insights.technical_notes
    }

if __name__ == "__main__":
    print("Neural Model Interpreter Module")
    print("=" * 50)
    print()
    print("This module interprets neural network forecasts using Qwen3-v1-30B LLM")
    print("Pipeline: Neural Data → Business Context → LLM Interpretation → Human Insights")
    print()
    print("Usage:")
    print("  from backend.neural_interpreter import interpret_neural_forecast")
    print("  insights = interpret_neural_forecast(forecast_df, product_code)")
    print()