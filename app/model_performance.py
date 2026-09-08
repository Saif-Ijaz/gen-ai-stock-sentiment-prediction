# app/model_performance.py
import streamlit as st
import plotly.graph_objects as go
import joblib
from datetime import datetime, timezone


@st.cache_resource
def load_model_metadata():
    bundle = joblib.load("models/model.pkl")
    return bundle.get("metadata", {})


def render_model_performance_page():
    st.markdown("## 🧠 Model Performance")

    meta = load_model_metadata()

    if not meta:
        st.warning("No model metadata found. Train the model first using `python main.py`.")
        return

    # -----------------------------------------------
    # Training info + staleness check
    # -----------------------------------------------
    trained_at = meta.get("trained_at", "Unknown")
    days_old = None
    try:
        dt = datetime.fromisoformat(trained_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        trained_display = dt.strftime("%B %d, %Y at %H:%M UTC")
        days_old = (datetime.now(timezone.utc) - dt).days
    except Exception:
        trained_display = trained_at

    st.caption(f"📅 Last trained: {trained_display}  |  📊 Features: {meta.get('n_features', 'N/A')}")

    # -----------------------------------------------
    # Metric cards
    # -----------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)

    acc  = meta.get("accuracy", 0) * 100
    f1   = meta.get("f1", 0) * 100
    prec = meta.get("precision", 0) * 100
    rec  = meta.get("recall", 0) * 100
    auc  = meta.get("roc_auc", 0) * 100

    c1.metric("Accuracy",  f"{acc:.1f}%")
    c2.metric("F1 Score",  f"{f1:.1f}%")
    c3.metric("Precision", f"{prec:.1f}%")
    c4.metric("Recall",    f"{rec:.1f}%")
    c5.metric("ROC-AUC",   f"{auc:.1f}%")

    # -----------------------------------------------
    # Cross validation stability
    # -----------------------------------------------
    cv_mean = meta.get("cv_accuracy_mean", 0) * 100
    cv_std  = meta.get("cv_accuracy_std", 0) * 100

    st.markdown(f"""
    <div style="background:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.2);border-radius:12px;padding:16px 18px;margin:14px 0">
        <div style="font-size:12px;color:var(--text-color);opacity:0.6;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:8px">
            Cross-Validation Stability (5-fold TimeSeriesSplit)
        </div>
        <div style="font-size:24px;font-weight:700;color:#4c9be8">
            {cv_mean:.1f}% <span style="font-size:14px;color:var(--text-color);opacity:0.6;font-weight:400">± {cv_std:.1f}%</span>
        </div>
        <div style="font-size:12px;color:var(--text-color);opacity:0.6;margin-top:6px">
            Low variance (±{cv_std:.1f}%) indicates a stable, reliable model across different time periods.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------------------------
    # Performance gauge
    # -----------------------------------------------
    col_gauge, col_bench = st.columns(2)

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=acc,
            number={"suffix": "%", "font": {"size": 32}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4c9be8"},
                "steps": [
                    {"range": [0, 50],  "color": "#ff4b4b"},
                    {"range": [50, 60], "color": "#f0a500"},
                    {"range": [60, 75], "color": "#4c9be8"},
                    {"range": [75, 100], "color": "#00c896"},
                ],
                "threshold": {"line": {"color": "rgba(128,128,128,0.6)", "width": 3}, "value": 50},
                "bgcolor": "rgba(0,0,0,0)",
            },
            title={"text": "Model Accuracy vs Random Baseline (50%)", "font": {"size": 13}},
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=None, height=260,
                          margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_bench:
        st.markdown("##### 📚 Benchmark Comparison")
        benchmarks = [
            ("Random Guess",       50.0, "#888"),
            ("Your Model",         acc,  "#4c9be8"),
        ]
        for label, val, color in benchmarks:
            is_yours = label == "Your Model"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
                <div style="width:140px;font-size:12px;color:var(--text-color);font-weight:{'700' if is_yours else '400'}">{label}</div>
                <div style="flex:1;background:var(--secondary-background-color);border-radius:4px;height:18px;position:relative">
                    <div style="background:{color};width:{val}%;height:100%;border-radius:4px"></div>
                </div>
                <div style="width:45px;font-size:12px;color:{color};font-weight:600;text-align:right">{val:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    # -----------------------------------------------
    # Feature importance
    # -----------------------------------------------
    st.markdown("##### 🔑 Feature Names Used")
    features = meta.get("feature_names", [])
    if features:
        feat_html = "".join([
            f"<span style='background:rgba(76,155,232,0.12);color:#4c9be8;padding:4px 10px;border-radius:6px;font-size:12px;margin:3px;display:inline-block'>{f}</span>"
            for f in features
        ])
        st.markdown(f"<div style='line-height:2.2'>{feat_html}</div>", unsafe_allow_html=True)

    # -----------------------------------------------
    # Honest interpretation
    # -----------------------------------------------
    st.markdown("##### 📝 Interpretation")
    if acc >= 60:
        interp = "Model shows strong predictive skill, meaningfully better than random chance."
        color = "#00c896"
    elif acc >= 55:
        interp = "Model shows modest predictive skill, consistent with academic research on short-term price prediction. Stock markets are highly efficient, making accuracy above 60% difficult to sustain."
        color = "#f0a500"
    else:
        interp = "Model performance is close to random — consider more features, more data, or a different target definition."
        color = "#ff4b4b"

    st.markdown(f"""
    <div style="background:var(--secondary-background-color);border-left:3px solid {color};border-radius:8px;padding:12px 16px;font-size:13px;color:var(--text-color)">
        {interp}
    </div>
    """, unsafe_allow_html=True)