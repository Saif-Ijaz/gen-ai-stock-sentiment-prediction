# app/api.py
import streamlit as st
import joblib
from app.realtime import get_realtime_prediction

@st.cache_resource  # FIX: loads model.pkl ONCE, never reloads
def load_model_bundle():
    return joblib.load("models/model.pkl")

def predict_api(ticker: str, freq: str):
    try:
        result = get_realtime_prediction(ticker, freq)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}