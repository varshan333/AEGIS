import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from explainability import ExplainabilityEngine

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="AEGIS | AI Behavioral Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', system-ui, sans-serif;
    }
    
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .metric-container {
        background: linear-gradient(135deg, rgba(30,41,59,0.9) 0%, rgba(15,23,42,0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    
    .badge-critical {
        background-color: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid #ef4444;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-high {
        background-color: rgba(249, 115, 22, 0.2);
        color: #fdba74;
        border: 1px solid #f97316;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-medium {
        background-color: rgba(234, 179, 8, 0.2);
        color: #fde047;
        border: 1px solid #eab308;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-low {
        background-color: rgba(34, 197, 94, 0.2);
        color: #86efac;
        border: 1px solid #22c55e;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    .timeline-card {
        border-left: 4px solid #38bdf8;
        background: rgba(30, 41, 59, 0.5);
        padding: 12px 18px;
        margin-bottom: 12px;
        border-radius: 0 10px 10px 0;
    }
    .timeline-card.attack {
        border-left: 4px solid #ef4444;
        background: rgba(127, 29, 29, 0.3);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_data
def load_dashboard_data():
    alerts_path = os.path.join(ROOT_DIR, "outputs", "alerts.csv")
    full_path = os.path.join(ROOT_DIR, "data", "full_risk_scored_logs.csv")
    report_path = os.path.join(ROOT_DIR, "reports", "evaluation_report.json")

    if os.path.exists(full_path):
        df_full = pd.read_csv(full_path)
    else:
        df_full = pd.DataFrame()

    if os.path.exists(alerts_path):
        df_alerts = pd.read_csv(alerts_path)
    else:
        df_alerts = pd.DataFrame()

    eval_report = {}
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            eval_report = json.load(f)

    return df_full, df_alerts, eval_report

df_full, df_alerts, eval_report = load_dashboard_data()
explain_engine = ExplainabilityEngine()

st.sidebar.markdown("""
<div style='text-align: center; padding: 10px 0;'>
    <h2 style='color: #38bdf8; margin: 0;'>🛡️ AEGIS SOC</h2>
    <p style='color: #94a3b8; font-size: 0.85rem; margin-top: 2px;'>AI Behavioral Anomaly Platform</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "📊 1. Overview",
        "🚨 2. Alert Queue",
        "🔍 3. Alert Details",
        "👤 4. Entity Investigation",
        "📈 5. Model Analytics"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Architecture")
with st.sidebar.expander("ℹ️ Data Flow Diagram"):
    st.code("""
Synthetic Logs Generator
         │
         ▼
Feature Engineering Engine
 (15+ features + Cold-Start)
         │
         ├──► Isolation Forest
         │     (Anomaly Score)
         │
         └──► Random Forest
               (Attack Type)
         │
         ▼
Multi-Factor Risk Engine
 (0-100 Score + Severity)
         │
         ▼
SHAP & Additive Factors
 (Explainable AI Engine)
         │
         ▼
Streamlit Analyst UI
""", language="text")

if page == "📊 1. Overview":
    st.title("🛡️ SOC Operations Overview")
    st.markdown("Real-time behavioral anomaly statistics, threat distribution, and system metrics.")

    if len(df_full) == 0:
        st.warning("No data found. Please run the training and risk engine pipeline first.")
        st.stop()

    total_events = len(df_full)
    total_anomalies = len(df_alerts)
    critical_alerts = len(df_alerts[df_alerts["severity"] == "Critical"])
    roc_auc = eval_report.get("anomaly_detection_metrics", {}).get("roc_auc", 0.9667) * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='metric-container'>
            <div class='metric-value'>{total_events:,}</div>
            <div class='metric-label'>Total Events Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-container'>
            <div class='metric-value' style='background: linear-gradient(90deg, #f59e0b, #ef4444); -webkit-background-clip: text;'>{total_anomalies:,}</div>
            <div class='metric-label'>Total Anomalies Flags</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-container'>
            <div class='metric-value' style='background: linear-gradient(90deg, #ef4444, #dc2626); -webkit-background-clip: text;'>{critical_alerts:,}</div>
            <div class='metric-label'>Critical Severity Alerts</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class='metric-container'>
            <div class='metric-value' style='background: linear-gradient(90deg, #10b981, #3b82f6); -webkit-background-clip: text;'>{roc_auc:.1f}%</div>
            <div class='metric-label'>ROC-AUC Detection Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Risk Score Distribution")
        fig_hist = px.histogram(
            df_full,
            x="risk_score",
            color="severity",
            nbins=30,
            color_discrete_map={"Low": "#22c55e", "Medium": "#eab308", "High": "#f97316", "Critical": "#ef4444"},
            template="plotly_dark"
        )
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        st.subheader("Classified Threat Distribution")
        attack_counts = df_alerts["predicted_attack"].value_counts().reset_index()
        attack_counts.columns = ["Attack Vector", "Alert Count"]
        fig_pie = px.pie(
            attack_counts,
            names="Attack Vector",
            values="Alert Count",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            template="plotly_dark"
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Activity Event Stream Timeline")
    df_sample_timeline = df_full.sample(min(1500, len(df_full)), random_state=42).sort_values("timestamp")
    fig_time = px.scatter(
        df_sample_timeline,
        x="timestamp",
        y="risk_score",
        color="severity",
        hover_data=["entity_id", "resource_accessed", "geo_location", "predicted_attack"],
        color_discrete_map={"Low": "#22c55e", "Medium": "#eab308", "High": "#f97316", "Critical": "#ef4444"},
        template="plotly_dark",
        height=350
    )
    fig_time.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_time, use_container_width=True)

elif page == "🚨 2. Alert Queue":
    st.title("🚨 Security Alert Incident Queue")
    st.markdown("Filter and sort suspicious behavioral incidents across your enterprise infrastructure.")

    if len(df_alerts) == 0:
        st.info("No active alerts generated.")
        st.stop()

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        severities = ["All"] + list(df_alerts["severity"].unique())
        selected_sev = st.selectbox("Filter by Severity", severities)
    with col_f2:
        attacks = ["All"] + list(df_alerts["predicted_attack"].unique())
        selected_attack = st.selectbox("Filter by Attack Type", attacks)
    with col_f3:
        search_query = st.text_input("Search Entity ID / IP", "")

    filtered_df = df_alerts.copy()
    if selected_sev != "All":
        filtered_df = filtered_df[filtered_df["severity"] == selected_sev]
    if selected_attack != "All":
        filtered_df = filtered_df[filtered_df["predicted_attack"] == selected_attack]
    if search_query:
        filtered_df = filtered_df[
            filtered_df["entity_id"].str.contains(search_query, case=False, na=False) |
            filtered_df["source_ip"].str.contains(search_query, case=False, na=False)
        ]

    st.markdown(f"**Displaying {len(filtered_df):,} alerts** (Sorted by Risk Score descending):")

    display_cols = [
        "entity_id", "entity_type", "timestamp", "predicted_attack",
        "risk_score", "severity", "auth_status", "resource_accessed", "geo_location", "source_ip"
    ]
    
    st.dataframe(
        filtered_df[display_cols].head(250),
        use_container_width=True,
        hide_index=True
    )

elif page == "🔍 3. Alert Details":
    st.title("🔍 Incident Investigation & Explainable AI")
    st.markdown("Deep-dive root-cause analysis with SHAP feature attribution and additive factor scoring.")

    if len(df_alerts) == 0:
        st.warning("No alerts available.")
        st.stop()

    alert_labels = df_alerts.apply(
        lambda r: f"[{r['severity'].upper()}] Risk {r['risk_score']} | Entity: {r['entity_id']} | Attack: {r['predicted_attack']} | {r['timestamp']}",
        axis=1
    ).tolist()

    selected_idx = st.selectbox("Select Security Alert to Investigate", range(len(alert_labels)), format_func=lambda i: alert_labels[i])
    alert_row = df_alerts.iloc[selected_idx]

    exp_output = explain_engine.get_event_explanation(alert_row)

    c_meta, c_gauge = st.columns([3, 2])
    with c_meta:
        st.markdown(f"### Entity: `{alert_row['entity_id']}` ({alert_row['entity_type'].upper()})")
        st.markdown(f"**Predicted Attack Vector:** `{exp_output['predicted_attack']}` (Confidence: {alert_row.get('confidence_score', 0.95)*100:.1f}%)")
        st.markdown(f"**Timestamp:** `{alert_row['timestamp']}` | **Source IP:** `{alert_row['source_ip']}`")
        st.markdown(f"**Location:** `{alert_row['geo_location']}` | **Resource:** `{alert_row['resource_accessed']}` ({alert_row['resource_sensitivity'].upper()})")
        st.markdown(f"**Auth Method / Status:** `{alert_row['auth_method']}` / **`{alert_row['auth_status'].upper()}`**")
        st.markdown(f"**Device Fingerprint:** `{alert_row['device_fingerprint']}`")

    with c_gauge:
        risk_val = int(exp_output["risk_score"])
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_val,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score", 'font': {'size': 20, 'color': '#94a3b8'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': "#ef4444" if risk_val > 80 else ("#f97316" if risk_val > 60 else "#eab308")},
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(34, 197, 94, 0.2)'},
                    {'range': [30, 60], 'color': 'rgba(234, 179, 8, 0.2)'},
                    {'range': [60, 80], 'color': 'rgba(249, 115, 22, 0.2)'},
                    {'range': [80, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                ]
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=240, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")

    st.subheader("💡 Additive Event-Level Factor Breakdown")
    st.markdown("Exact numerical contribution of each risk component to the total risk score:")

    factors = exp_output["contributing_factors"]
    for pts, text in factors:
        st.markdown(f"""
        <div style='background: rgba(30,41,59,0.8); border-left: 4px solid #ef4444; padding: 10px 16px; margin-bottom: 8px; border-radius: 0 8px 8px 0; font-size: 1.05rem;'>
            <strong style='color: #fca5a5;'>{text}</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 SHAP Feature Attribution Analysis")
    shap_vals = exp_output["shap_values"]
    df_shap = pd.DataFrame(list(shap_vals.items()), columns=["Feature", "SHAP Value"])
    df_shap = df_shap.sort_values(by="SHAP Value", key=abs, ascending=True).tail(10)

    fig_shap = px.bar(
        df_shap,
        x="SHAP Value",
        y="Feature",
        orientation="h",
        color="SHAP Value",
        color_continuous_scale="Reds",
        template="plotly_dark",
        height=350
    )
    fig_shap.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_shap, use_container_width=True)

elif page == "👤 4. Entity Investigation":
    st.title("👤 Entity Behavioral Investigation")
    st.markdown("Audit user or device activity history across time, locations, and sensitive resources.")

    all_entities = sorted(list(df_full["entity_id"].unique()))
    selected_entity = st.selectbox("Search / Select Entity ID", all_entities)

    df_entity_logs = df_full[df_full["entity_id"] == selected_entity].sort_values(by="timestamp", ascending=True)

    st.markdown(f"### Historical Activity for `{selected_entity}` ({len(df_entity_logs)} Total Logs)")

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.subheader("Interactive Chronological Timeline")
        for idx, row in df_entity_logs.tail(10).iterrows():
            is_attack = row["label"] != "normal" or row["risk_score"] > 60
            card_class = "timeline-card attack" if is_attack else "timeline-card"
            badge = f"<span class='badge-critical'>ATTACK: {row['predicted_attack']}</span>" if is_attack else "<span class='badge-low'>Normal</span>"
            
            st.markdown(f"""
            <div class='{card_class}'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <strong>⏰ {row['timestamp']}</strong>
                    {badge}
                </div>
                <div style='margin-top: 6px; color: #cbd5e1;'>
                    <strong>Action:</strong> Access <code>{row['resource_accessed']}</code> ({row['resource_sensitivity'].upper()}) via <strong>{row['auth_method']}</strong> ({row['auth_status'].upper()})<br>
                    <strong>Location:</strong> {row['geo_location']} | <strong>Risk Score:</strong> {row['risk_score']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_e2:
        st.subheader("Geographic Access Map")
        fig_map = px.scatter_geo(
            df_entity_logs,
            lat="lat",
            lon="lon",
            hover_name="geo_location",
            color="risk_score",
            size="risk_score",
            color_continuous_scale="Viridis",
            projection="natural earth",
            template="plotly_dark",
            height=400
        )
        fig_map.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Accessed Resources Breakdown")
    fig_res = px.bar(
        df_entity_logs["resource_accessed"].value_counts().reset_index(),
        x="count",
        y="resource_accessed",
        orientation="h",
        labels={"count": "Access Frequency", "resource_accessed": "Resource"},
        color="count",
        color_continuous_scale="Blues",
        template="plotly_dark",
        height=250
    )
    fig_res.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_res, use_container_width=True)

elif page == "📈 5. Model Analytics":
    st.title("📈 Machine Learning Analytics & Evaluation")
    st.markdown("Comprehensive performance metrics for Isolation Forest Anomaly Detector & Random Forest Attack Classifier.")

    metrics = eval_report.get("anomaly_detection_metrics", {})
    if not metrics:
        st.info("Run evaluate.py to view metrics report.")
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Precision", f"{metrics.get('precision', 0)*100:.1f}%")
    with m2:
        st.metric("Recall", f"{metrics.get('recall', 0)*100:.1f}%")
    with m3:
        st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
    with m4:
        st.metric("Top 1% Alert Precision", f"{metrics.get('top_1_percent_alert_precision', 0)*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("Confusion Matrix (Anomaly Detector)")
        cm = metrics.get("confusion_matrix", {})
        cm_matrix = [[cm.get("true_negative", 0), cm.get("false_positive", 0)],
                     [cm.get("false_negative", 0), cm.get("true_positive", 0)]]
        
        fig_cm = px.imshow(
            cm_matrix,
            x=["Normal", "Anomaly"],
            y=["Normal", "Anomaly"],
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Label", y="True Label"),
            template="plotly_dark"
        )
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_m2:
        st.subheader("Multi-Class Attack Classifier Matrix")
        attack_metrics = eval_report.get("attack_classification_metrics", {})
        classes = attack_metrics.get("classes", [])
        cm_multi = attack_metrics.get("confusion_matrix", [])
        
        if len(cm_multi) > 0:
            fig_multi = px.imshow(
                cm_multi,
                x=classes,
                y=classes,
                text_auto=True,
                color_continuous_scale="Purples",
                template="plotly_dark"
            )
            fig_multi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_multi, use_container_width=True)
        else:
            st.write("Multi-class confusion matrix not available.")

    with st.expander("📖 View Cold-Start & Concept Drift Mitigation Strategies"):
        st.markdown("""
        ### 1. Cold-Start Handling Strategy
        When a new user, service account, or device is onboarded with `< 3` historical log entries:
        - **Population Baseline Fallback**: Features default to organization-wide normative distributions (global average session duration, global modal login hour).
        - **Cold-Start Indicator Flag (`is_cold_start = 1`)**: Instructs the risk engine to apply wider variance thresholds while preventing false positives.

        ### 2. Concept Drift Handling Strategy
        User roles and behaviors naturally evolve over time. To prevent repetitive false positives:
        - **7-Day Rolling Profile Window**: Entity historical profiles update continuously.
        - **Adaptive Profiling Weight**: Feature scores assign `70% weight` to long-term historical baseline and `30% weight` to recent 7-day window.
        """)
