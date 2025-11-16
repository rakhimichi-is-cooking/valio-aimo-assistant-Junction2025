"""
Web-based Chat Interface for Valio AI using Streamlit

Run with: streamlit run chat_web.py
"""

import streamlit as st
from backend.conversational_agent import ConversationalAgent
import time

# Page config
st.set_page_config(
    page_title="Valio AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'agent' not in st.session_state:
    st.session_state.agent = None

# Title
st.title("🤖 Valio AI - Supply Chain Assistant")
st.markdown("Ask me anything about products, forecasts, shortages, and replacements!")

# Sidebar with examples
with st.sidebar:
    st.header("Example Questions")
    st.markdown("""
    **Product Replacements:**
    - What replacements do we have for product 6409460002724?
    - Find substitutes for Forssan Potato Salad

    **Data Analysis:**
    - How many products are in the catalog?
    - Show me the top 5 products by category
    - What products are in category 17301?

    **Forecasting:**
    - Generate a 7-day forecast for product 400122
    - What's the predicted demand for next week?

    **Shortages:**
    - Show me recent shortage events
    - Which products have delivery issues?
    """)

    st.divider()

    st.markdown("**System Status:**")

    # Check LM Studio connection
    try:
        import requests
        response = requests.get("http://localhost:1234/v1/models", timeout=2)
        if response.status_code == 200:
            st.success("✅ LM Studio Connected")
        else:
            st.error("❌ LM Studio Error")
    except:
        st.error("❌ LM Studio Not Running")

    st.markdown("""
    **Tech Stack:**
    - LLM: Qwen 3-vl-30B
    - GNN: PyTorch Geometric
    - Backend: FastAPI
    """)

# Initialize agent
if st.session_state.agent is None:
    with st.spinner("Initializing AI agent..."):
        try:
            st.session_state.agent = ConversationalAgent()
            st.success("Agent initialized!", icon="✅")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show code if available
        if message.get("code"):
            with st.expander("View generated code"):
                st.code(message["code"], language="python")

# Chat input
if prompt := st.chat_input("Ask me anything about the supply chain..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.agent.answer_question(prompt)

                if response['error']:
                    st.error(f"Error: {response['error']}")
                    answer = f"I encountered an error: {response['error']}"
                    code = response.get('code_generated', '')
                else:
                    answer = response['answer']
                    code = response.get('code_generated', '')

                st.markdown(answer)

                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "code": code
                })

                # Show code expander
                if code:
                    with st.expander("View generated code"):
                        st.code(code, language="python")

            except Exception as e:
                st.error(f"Error: {e}")

# Clear chat button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Valio AI - Powered by Graph Neural Networks & LLM | Built for Junction 2025</small>
</div>
""", unsafe_allow_html=True)
