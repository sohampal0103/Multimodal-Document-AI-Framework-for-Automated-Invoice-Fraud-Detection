import streamlit as st
import tempfile, os
from src.pipeline import run_pipeline

st.set_page_config(page_title="Invoice Fraud Detection")

st.title("AI Invoice Fraud Detection System")

file = st.file_uploader("Drag & Drop Invoice", type=["png","jpg","jpeg"])

if file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.read())
        path = tmp.name

    st.image(path, use_column_width=True)

    if st.button("Analyze"):
        data, score, anomalies = run_pipeline(path)
        st.write("Extracted Data:", data)
        st.progress(int(score*100))
        st.write("Fraud Probability:", score)
        st.write("Anomalies:", anomalies)

    os.unlink(path)
