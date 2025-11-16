import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Valio Aimo Shortage Assistant", layout="wide")
st.title("Valio Aimo Shortage Assistant v0.1")

st.sidebar.header("Filters")
threshold = st.sidebar.slider("Risk threshold", 0.0, 1.0, 0.5, 0.05)
language = st.sidebar.selectbox("Customer message language", ["en", "fi", "sv"])

st.sidebar.markdown("---")
st.sidebar.write("Backend URL:")
st.sidebar.code(BACKEND_URL, language="bash")


@st.cache_data(ttl=60)
def fetch_shortages(threshold_value: float):
    resp = requests.get(
        f"{BACKEND_URL}/shortages",
        params={"threshold": threshold_value},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


shortages = []
error_msg = None
try:
    shortages = fetch_shortages(threshold)
except Exception as e:
    error_msg = str(e)

if error_msg:
    st.error(f"Error fetching shortages: {error_msg}")
else:
    st.subheader("Predicted Shortages")
    if not shortages:
        st.info("No shortages above the selected risk threshold.")
    else:
        import pandas as pd

        table_rows = []
        for idx, evt in enumerate(shortages):
            table_rows.append(
                {
                    "index": idx,
                    "Customer": evt["customer_name"],
                    "SKU": evt["sku"],
                    "Product": evt["product_name"],
                    "Ordered": evt["ordered_qty"],
                    "Delivered": evt["delivered_qty"],
                    "Risk": round(evt["risk_score"], 3),
                }
            )

        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True)

        st.markdown("### Event Detail & AI Messages")

        selected_index = st.number_input(
            "Select row index",
            min_value=0,
            max_value=len(shortages) - 1,
            value=0,
            step=1,
        )

        event = shortages[selected_index]
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Event data")
            st.json(event)

            st.markdown("#### Suggested substitutes")
            subs = event.get("suggested_substitutes") or []
            if not subs:
                st.write("No substitutes found.")
            else:
                st.table(subs)

        with col2:
            if st.button("Generate AI explanation and messages"):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/shortages/message",
                        params={"language": language},
                        json=event,
                        timeout=60,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.markdown("**Summary (for ops):**")
                    st.write(data["summary"] or "")

                    if data.get("internal_notes"):
                        st.markdown("**Internal notes:**")
                        st.write(data["internal_notes"])

                    st.markdown("**Customer message:**")
                    st.code(data["customer_message"] or "", language="text")

                    st.markdown("**Call script:**")
                    st.code(data["call_script"] or "", language="text")
                except Exception as e:
                    st.error(f"Error generating AI message: {e}")
