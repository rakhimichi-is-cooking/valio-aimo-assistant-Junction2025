from typing import Dict, Any, List
import requests
import json
import re
from pydantic import BaseModel, Field, ValidationError
from .config import LMSTUDIO_BASE_URL, LMSTUDIO_MODEL


class AnalysisStage(BaseModel):
    """Pydantic model for Stage 1: Data Analysis."""
    severity_assessment: str = Field(description="Assessment of shortage severity")
    historical_context: str = Field(description="Historical patterns for this product/customer")
    customer_impact: str = Field(description="Impact on customer operations")
    risk_factors: str = Field(description="Key risk factors identified")
    recommended_action: str = Field(description="Recommended course of action")


class AimoResponse(BaseModel):
    """Pydantic model for validated AIMO response."""
    summary: str = Field(description="Operations summary of the shortage event")
    internal_notes: str = Field(description="Internal risk and operational notes")
    customer_message: str = Field(description="Customer-facing message (SMS/email)")
    call_script: str = Field(description="Phone support call script")
    language: str = Field(default="en", description="Response language")

SYSTEM_PROMPT = """
You are AIMO OPS, an AI operations companion for Valio Aimo.

Your job is to help warehouse staff and customer support handle
delivery shortages and replacements. You:

- Explain predicted shortages in plain language.
- Suggest good replacements using the provided substitute list.
- Draft short, polite messages to customers (SMS or email).
- Draft short call scripts with greeting, issue, options, and closing.
- Stay professional, calm, and helpful.
- Never swear or insult anyone.
- Keep answers concise (2–6 sentences per section).
"""

class AimoAgent:
    def __init__(self) -> None:
        self.api_url = f"{LMSTUDIO_BASE_URL.rstrip('/')}/v1/chat/completions"
        self.model = LMSTUDIO_MODEL

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 512,
            "stream": False,
        }
        resp = requests.post(self.api_url, json=payload, timeout=60)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return content

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response with multiple strategies.

        Args:
            text: Raw LLM response text

        Returns:
            Extracted JSON dictionary

        Raises:
            ValueError: If no valid JSON found
        """
        # Strategy 1: Find JSON code block
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 2: Find first complete JSON object
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass

        # Strategy 3: Try to find each brace pair
        depth = 0
        start_idx = -1
        for i, char in enumerate(text):
            if char == '{':
                if depth == 0:
                    start_idx = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start_idx != -1:
                    try:
                        return json.loads(text[start_idx:i + 1])
                    except json.JSONDecodeError:
                        continue

        raise ValueError("No valid JSON found in response")

    def generate_for_event(
        self,
        event: Dict[str, Any],
        language: str = "en",
    ) -> Dict[str, str]:
        """
        Generate AIMO response for a shortage event with Pydantic validation.

        Args:
            event: Shortage event dictionary
            language: Target language (en, fi, sv)

        Returns:
            Validated response dictionary with keys:
                - summary: Operations summary
                - internal_notes: Risk details
                - customer_message: Customer-facing message
                - call_script: Phone support script
                - language: Response language
        """
        user_prompt = f"""
Shortage event (JSON):

{json.dumps(event, indent=2)}

Language: {language}

Tasks:
1) Summarize the situation for Valio operations staff in 2–4 sentences.
2) Provide internal notes with any relevant risk details (mention risk_score if present).
3) Draft a single SMS/email style message to the customer explaining the issue and offering the listed substitutes.
4) Draft a short call script for phone support staff.

IMPORTANT: Return ONLY valid JSON in this exact format (no markdown, no code blocks):

{{
  "summary": "...",
  "internal_notes": "...",
  "customer_message": "...",
  "call_script": "...",
  "language": "{language}"
}}
"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]

        try:
            raw = self._call_llm(messages)

            # Extract JSON from response
            json_obj = self._extract_json_from_text(raw)

            # Validate with Pydantic
            response = AimoResponse(**json_obj)

            # Return as dictionary
            return response.model_dump()

        except ValueError as e:
            # JSON extraction failed
            print(f"Failed to extract JSON from LLM response: {e}")
            return self._create_fallback_response(event, language, raw)

        except ValidationError as e:
            # Pydantic validation failed
            print(f"Pydantic validation failed: {e}")
            # Try to extract what we can
            try:
                json_obj = self._extract_json_from_text(raw)
                return self._create_partial_response(json_obj, language)
            except:
                return self._create_fallback_response(event, language, raw)

        except Exception as e:
            # General error
            print(f"Error generating AIMO response: {e}")
            return self._create_fallback_response(event, language, "")

    def _create_fallback_response(
        self,
        event: Dict[str, Any],
        language: str,
        raw_content: str
    ) -> Dict[str, str]:
        """
        Create a fallback response when LLM parsing fails.

        Args:
            event: Shortage event
            language: Target language
            raw_content: Raw LLM response

        Returns:
            Fallback response dictionary
        """
        customer_name = event.get("customer_name", "Customer")
        sku = event.get("sku", "product")
        ordered = event.get("ordered_qty", 0)
        delivered = event.get("delivered_qty", 0)
        shortage = ordered - delivered

        return {
            "summary": f"Shortage detected: {sku} for {customer_name}. Ordered: {ordered}, Delivered: {delivered}",
            "internal_notes": f"Manual review needed. Risk score: {event.get('risk_score', 'N/A')}",
            "customer_message": raw_content[:500] if raw_content else f"Dear customer, we have a shortage of {shortage} units for order {sku}. We apologize for the inconvenience.",
            "call_script": f"Hello, this is regarding your order for {sku}. We have a partial shortage and wanted to discuss alternatives with you.",
            "language": language,
        }

    def _create_partial_response(
        self,
        json_obj: Dict[str, Any],
        language: str
    ) -> Dict[str, str]:
        """
        Create response from partially valid JSON.

        Args:
            json_obj: Partially valid JSON object
            language: Target language

        Returns:
            Response dictionary with defaults for missing fields
        """
        return {
            "summary": json_obj.get("summary", "").strip() or "Shortage event detected.",
            "internal_notes": json_obj.get("internal_notes", "").strip() or "Review required.",
            "customer_message": json_obj.get("customer_message", "").strip() or "We have a shortage on your order.",
            "call_script": json_obj.get("call_script", "").strip() or "Call customer to discuss shortage.",
            "language": json_obj.get("language", language),
        }

    def _analyze_shortage_event(self, event: Dict[str, Any]) -> Dict[str, str]:
        """
        Stage 1: Analyze shortage event data (Analyzer LLM).

        This stage examines raw data and produces structured analysis without
        making customer-facing decisions.

        Args:
            event: Shortage event dictionary

        Returns:
            Analysis dictionary with severity, context, impact, and risks
        """
        analyzer_prompt = f"""
You are a data analyst for Valio supply chain operations.

Analyze this shortage event and provide structured insights:

Event Data:
{json.dumps(event, indent=2)}

Analyze the following aspects:
1. Severity Assessment: How severe is this shortage? (quantity, percentage, customer impact)
2. Historical Context: What do the scores tell us about past patterns?
3. Customer Impact: How will this affect the customer's operations?
4. Risk Factors: What are the key risk factors? (composite_risk_score, seasonal_score, forecast_score if present)
5. Recommended Action: What should operations prioritize?

Return ONLY valid JSON:
{{
  "severity_assessment": "...",
  "historical_context": "...",
  "customer_impact": "...",
  "risk_factors": "...",
  "recommended_action": "..."
}}
"""

        messages = [
            {"role": "system", "content": "You are an expert supply chain data analyst. Provide concise, factual analysis."},
            {"role": "user", "content": analyzer_prompt.strip()}
        ]

        try:
            raw = self._call_llm(messages)
            json_obj = self._extract_json_from_text(raw)
            analysis = AnalysisStage(**json_obj)
            return analysis.model_dump()
        except Exception as e:
            print(f"Analysis stage failed: {e}")
            # Fallback analysis
            return {
                "severity_assessment": f"Shortage: {event.get('ordered_qty', 0) - event.get('delivered_qty', 0)} units",
                "historical_context": f"Risk score: {event.get('risk_score', 'N/A')}",
                "customer_impact": f"Customer: {event.get('customer_name', 'Unknown')}",
                "risk_factors": "Manual review needed",
                "recommended_action": "Contact customer with substitutes"
            }

    def _generate_decision(
        self,
        event: Dict[str, Any],
        analysis: Dict[str, str],
        language: str = "en"
    ) -> Dict[str, str]:
        """
        Stage 2: Generate customer-facing decisions (Decision Maker LLM).

        This stage takes the analysis and produces actionable communications
        and scripts for customer interaction.

        Args:
            event: Original shortage event
            analysis: Analysis from Stage 1
            language: Target language

        Returns:
            Response dictionary with messages and scripts
        """
        decision_prompt = f"""
You are AIMO OPS customer communication specialist.

Based on this analysis, generate customer communications:

Analysis from Data Team:
{json.dumps(analysis, indent=2)}

Event Details:
- Customer: {event.get('customer_name', 'Unknown')}
- Product: {event.get('product_name', event.get('sku', 'Unknown'))} ({event.get('sku', '')})
- Ordered: {event.get('ordered_qty', 0)}
- Delivered: {event.get('delivered_qty', 0)}
- Substitutes: {json.dumps(event.get('suggested_substitutes', []), indent=2)}

Language: {language}

Generate:
1. Summary: Brief operational summary for internal team (2-3 sentences)
2. Internal Notes: Risk and operational notes based on analysis
3. Customer Message: Professional SMS/email explaining shortage and offering substitutes
4. Call Script: Short phone support script with greeting, issue explanation, substitute offer, and closing

Return ONLY valid JSON:
{{
  "summary": "...",
  "internal_notes": "...",
  "customer_message": "...",
  "call_script": "...",
  "language": "{language}"
}}
"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": decision_prompt.strip()}
        ]

        try:
            raw = self._call_llm(messages)
            json_obj = self._extract_json_from_text(raw)
            response = AimoResponse(**json_obj)
            return response.model_dump()
        except Exception as e:
            print(f"Decision stage failed: {e}")
            return self._create_fallback_response(event, language, "")

    def generate_for_event_two_stage(
        self,
        event: Dict[str, Any],
        language: str = "en"
    ) -> Dict[str, str]:
        """
        Two-stage LLM reasoning for shortage event handling.

        Stage 1 (Analyzer): Analyzes data → structured insights
        Stage 2 (Decision Maker): Insights → customer communications

        This approach improves accuracy by separating analysis from decision-making,
        inspired by research on multi-agent LLM systems for supply chains.

        Args:
            event: Shortage event dictionary
            language: Target language (en, fi, sv)

        Returns:
            Complete response with summary, notes, messages, and scripts
        """
        # Stage 1: Analyze the data
        analysis = self._analyze_shortage_event(event)

        # Stage 2: Generate decisions based on analysis
        response = self._generate_decision(event, analysis, language)

        # Enrich response with analysis data
        response['analysis'] = analysis

        return response
