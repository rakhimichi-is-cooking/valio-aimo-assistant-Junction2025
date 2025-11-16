"""
Conversational AI Agent for Valio Supply Chain System

User asks questions in natural language:
- "What replacements do we have for product 400122?"
- "What's the forecast for milk next week?"
- "Show me shortage events today"

LLM converts to Python code, executes it, returns results in natural language.
"""

import json
import requests
from typing import Dict, Any, List
import traceback
import sys
from io import StringIO

class ConversationalAgent:
    """
    Two-way conversational agent that can answer questions about the supply chain.

    Flow:
    1. User asks question in natural language
    2. LLM generates Python code to query system
    3. Code executes and retrieves data
    4. LLM formats results as natural language response
    """

    def __init__(self, lm_studio_url: str = "http://localhost:1234"):
        self.lm_studio_url = lm_studio_url
        self.conversation_history = []

        # System context - what the LLM knows about the system
        self.system_context = """You are Valio AI, an intelligent supply chain assistant.

You have access to a Python environment with the following modules:

AVAILABLE MODULES:
- backend.replacement_engine: suggest_substitutes(sku, product_data, k=5)
- backend.neural_interpreter: interpret_neural_forecast(forecast_df, product_code, historical_df)
- backend.gnn_forecaster: GNNForecaster (for demand forecasting)
- backend.data_loader: load_sales_and_deliveries(), load_product_catalog()
- pandas, numpy, json

AVAILABLE DATA FILES:
- data/product_features/product_catalog.json (17,546 products)
- data/valio_aimo_sales_and_deliveries_junction_2025.csv (sales data)
- data/product_graph/product_graph.pkl (product relationship graph)

COMMON TASKS:

1. Find product replacements:
```python
import json
from backend.replacement_engine import suggest_substitutes

with open('data/product_features/product_catalog.json', encoding='utf-8') as f:
    products = json.load(f)

# Format products
products_formatted = [{'sku': p['gtin'], 'name': p['name'], 'category': p['category']} for p in products]

# Get replacements
replacements = suggest_substitutes(sku='6409460002724', product_data=products_formatted, k=5)
result = replacements
```

2. Load and analyze sales data:
```python
import pandas as pd

sales = pd.read_csv('data/valio_aimo_sales_and_deliveries_junction_2025.csv')
result = sales.head(10).to_dict('records')
```

NOTE: Sales CSV columns are:
- product_code, customer_number, order_qty, delivered_qty, requested_delivery_date, order_created_date
- NO 'category' column - category is in product catalog (use GTIN to join)

3a. Join sales with product catalog for category analysis:
```python
import pandas as pd
import json

# Load sales
sales = pd.read_csv('data/valio_aimo_sales_and_deliveries_junction_2025.csv')

# Load product catalog
with open('data/product_features/product_catalog.json', encoding='utf-8') as f:
    products = json.load(f)

# Create product lookup
product_df = pd.DataFrame(products)
product_df['product_code'] = product_df['gtin']

# Join sales with products to get category
sales_with_category = sales.merge(product_df[['product_code', 'category', 'name']],
                                   on='product_code', how='left')

# Now can analyze by category
category_sales = sales_with_category.groupby('category')['order_qty'].sum().reset_index()
result = category_sales.sort_values('order_qty', ascending=False).head(5).to_dict('records')
```

3. Detect shortage events:
```python
import pandas as pd
from backend.shortage_engine import compute_shortage_events

# Load sales data
sales = pd.read_csv('data/valio_aimo_sales_and_deliveries_junction_2025.csv')

# Compute shortage events
shortage_events = compute_shortage_events(sales, use_advanced_scoring=True)

# Get top 5 by risk score
result = sorted(shortage_events, key=lambda x: x['risk_score'], reverse=True)[:5]
```

4. GNN Forecast for specific product:
```python
import pandas as pd
import numpy as np
from backend.gnn_forecaster import GNNForecaster

# Load sales data
sales = pd.read_csv('data/valio_aimo_sales_and_deliveries_junction_2025.csv')

# Get historical demand for a product
product_code = '400122'
product_data = sales[sales['product_code'] == product_code]
history = product_data.groupby('requested_delivery_date')['order_qty'].sum().values

# Initialize forecaster (requires trained model)
forecaster = GNNForecaster()
# forecaster.load_model('path/to/model.pth')

# Predict next 7 days (requires at least 30 days history)
if len(history) >= 30:
    forecast = forecaster.predict(product_code=product_code, history=history[-30:])
    result = forecast
else:
    result = {"error": "Not enough historical data"}
```

RULES:
1. Always store final result in variable called 'result'
2. Keep code concise and focused
3. Handle errors gracefully
4. For product SKUs, use GTIN from catalog
5. Limit results to reasonable amounts (top 5-10)
6. ALWAYS use encoding='utf-8' when opening files (e.g., open('file.json', encoding='utf-8'))
7. Use try/except for file operations

When user asks a question:
1. Generate Python code to answer it
2. Code should set 'result' variable with the answer
3. Return ONLY the Python code in a code block
4. After execution, format 'result' into natural language response
"""

    def call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Call LM Studio API"""
        try:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            response = requests.post(
                f"{self.lm_studio_url}/v1/chat/completions",
                json={
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 500,  # Reduced from 1000
                },
                timeout=20  # Reduced from 60
            )

            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            else:
                return f"Error: LLM API returned status {response.status_code}"

        except Exception as e:
            return f"Error calling LLM: {str(e)}"

    def extract_code_from_response(self, llm_response: str) -> str:
        """Extract Python code from LLM response"""
        # Look for code blocks
        if "```python" in llm_response:
            start = llm_response.find("```python") + 9
            end = llm_response.find("```", start)
            code = llm_response[start:end].strip()
        elif "```" in llm_response:
            start = llm_response.find("```") + 3
            end = llm_response.find("```", start)
            code = llm_response[start:end].strip()
        else:
            # Assume entire response is code
            code = llm_response.strip()
        
        # Clean up common issues
        # Remove any "python" keyword at the start
        if code.startswith("python\n"):
            code = code[7:]
        
        # Remove any leading/trailing quotes
        code = code.strip("'\"")
        
        return code

    def execute_code(self, code: str) -> Dict[str, Any]:
        """
        Safely execute Python code and capture result.

        Returns:
            dict with 'success', 'result', 'error', 'stdout'
        """
        # Create isolated namespace
        namespace = {
            '__builtins__': __builtins__,
            'result': None
        }

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            # Execute code
            exec(code, namespace)

            # Get result
            result = namespace.get('result')
            stdout = captured_output.getvalue()

            return {
                'success': True,
                'result': result,
                'error': None,
                'stdout': stdout
            }

        except Exception as e:
            return {
                'success': False,
                'result': None,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'stdout': captured_output.getvalue()
            }
        finally:
            sys.stdout = old_stdout

    def format_result_as_text(self, result: Any, original_question: str) -> str:
        """Use LLM to format result as natural language"""

        # Convert result to string representation
        if isinstance(result, (list, dict)):
            result_str = json.dumps(result, indent=2, default=str)[:2000]  # Reduced from 5000
        else:
            result_str = str(result)[:2000]

        prompt = f"""The user asked: "{original_question}"

The system returned this data:
{result_str}

Provide a clear, 2-sentence response summarizing the key findings."""

        return self.call_llm(prompt)

    def answer_question(self, user_question: str) -> Dict[str, Any]:
        """
        Main method: Answer user's question through code generation and execution.

        Args:
            user_question: Natural language question

        Returns:
            dict with 'answer', 'code_generated', 'execution_result', 'error'
        """
        print(f"\nUser Question: {user_question}")
        print("-" * 70)

        # Step 1: Generate Python code from question
        print("Step 1: Generating Python code...")
        code_prompt = f"""User question: "{user_question}"

Generate ONLY Python code to answer this. No explanations, no markdown, just executable code.
Store the final answer in a variable called 'result'.

Example format:
import pandas as pd
sales = pd.read_csv('data/valio_aimo_sales_and_deliveries_junction_2025.csv')
result = sales['product_code'].value_counts().head(5).to_dict()
"""

        llm_response = self.call_llm(code_prompt, system_prompt=self.system_context)
        code = self.extract_code_from_response(llm_response)

        print("Generated code:")
        print(code)
        print()

        # Step 2: Execute the code
        print("Step 2: Executing code...")
        execution_result = self.execute_code(code)

        if not execution_result['success']:
            print(f"Execution failed: {execution_result['error']}")
            print(f"Traceback: {execution_result.get('traceback', 'N/A')}")
            return {
                'answer': f"I encountered an error: {execution_result['error']}. Code generated: {code[:200]}",
                'code_generated': code,
                'execution_result': execution_result,
                'error': execution_result['error']
            }

        print(f"Execution successful!")
        print(f"Result type: {type(execution_result['result'])}")
        print()

        # Step 3: Format result as natural language
        print("Step 3: Formatting response...")
        natural_response = self.format_result_as_text(
            execution_result['result'],
            user_question
        )

        print("Response generated!")
        print()

        return {
            'answer': natural_response,
            'code_generated': code,
            'execution_result': execution_result,
            'error': None
        }

    def chat(self, user_message: str) -> str:
        """
        Simple chat interface - returns just the answer text.

        Args:
            user_message: User's question

        Returns:
            Natural language answer
        """
        result = self.answer_question(user_message)
        return result['answer']


def demo_conversational_agent():
    """Demo the conversational agent with sample questions"""

    print("="*70)
    print("VALIO AI CONVERSATIONAL AGENT DEMO")
    print("="*70)
    print()

    agent = ConversationalAgent()

    # Sample questions
    questions = [
        "What are the top 5 replacement products for GTIN 6409460002724?",
        "How many products are in the catalog?",
        "Show me the first 3 products in the catalog",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"QUESTION {i}: {question}")
        print(f"{'='*70}")

        response = agent.answer_question(question)

        print(f"\n{'='*70}")
        print("ANSWER:")
        print(f"{'='*70}")
        print(response['answer'])
        print()

        if i < len(questions):
            input("\nPress Enter for next question...")


if __name__ == "__main__":
    demo_conversational_agent()
