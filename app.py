"""
FeedbackOps AI - Main Streamlit Application
Intelligent Feedback Operations & Pain Point Triage powered by Local Ollama LLM.
"""

import streamlit as st
import pandas as pd
import altair as alt
from typing import Dict, Any, List

from feedbackops.ollama_client import (
    OllamaClient,
    OllamaError,
    OllamaConnectionError,
    OllamaTimeoutError,
    OllamaModelNotFoundError,
    OllamaResponseError
)
from feedbackops.analyzer import FeedbackAnalyzer
from feedbackops.sample_data import get_sample_dataframe, SAMPLE_REVIEWS
from feedbackops import utils

# Page Configuration
st.set_page_config(
    page_title="FeedbackOps AI - Customer Feedback Operations",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
CUSTOM_CSS = """
<style>
    /* Card Container */
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-card p {
        margin: 4px 0 0 0;
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 600;
        border-radius: 12px;
        margin-right: 6px;
    }
    .badge-positive { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .badge-neutral { background-color: #e2e3e5; color: #383d41; border: 1px solid #d6d8db; }
    .badge-negative { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }

    .badge-critical { background-color: #dc3545; color: #ffffff; }
    .badge-high { background-color: #fd7e14; color: #ffffff; }
    .badge-medium { background-color: #ffc107; color: #212529; }
    .badge-low { background-color: #28a745; color: #ffffff; }

    /* Custom Alert Box */
    .setup-banner {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-left: 6px solid #ffc107;
        color: #856404;
        padding: 16px 20px;
        border-radius: 6px;
        margin-bottom: 24px;
    }
    .setup-banner h4 { margin-top: 0; color: #856404; font-weight: 700; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_session_state():
    """Initialize persistent Streamlit session state variables."""
    if "ollama_host" not in st.session_state:
        st.session_state.ollama_host = "http://localhost:11434"
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "llama3.2"
    if "ollama_timeout" not in st.session_state:
        st.session_state.ollama_timeout = 180
    if "single_review_text" not in st.session_state:
        st.session_state.single_review_text = ""
    if "single_analysis_result" not in st.session_state:
        st.session_state.single_analysis_result = None
    if "batch_analysis_results" not in st.session_state:
        st.session_state.batch_analysis_results = None


init_session_state()

# Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/color/96/flash-auto.png", width=64)
    st.title("FeedbackOps AI")
    st.caption("Local Ollama Feedback Operations Engine")
    st.markdown("---")

    st.subheader("⚙️ Ollama Configuration")
    
    ollama_host = st.text_input(
        "Ollama Host URL",
        value=st.session_state.ollama_host,
        help="Default local endpoint: http://localhost:11434"
    )
    st.session_state.ollama_host = ollama_host

    ollama_timeout = st.number_input(
        "Read Timeout (seconds)",
        min_value=10,
        max_value=600,
        value=st.session_state.ollama_timeout,
        step=10,
        help="Timeout limit for Ollama generation. Default is 180 seconds."
    )
    st.session_state.ollama_timeout = ollama_timeout

    client = OllamaClient(host=ollama_host, timeout=st.session_state.ollama_timeout)
    is_connected, conn_msg = client.check_connection()

    if is_connected:
        st.success("🟢 Ollama Service Connected")
        available_models = []
        try:
            available_models = client.list_models()
        except Exception:
            pass
        
        if available_models:
            model_options = list(set([m.split(":")[0] for m in available_models] + available_models))
            if st.session_state.selected_model not in model_options and model_options:
                st.session_state.selected_model = model_options[0]

            selected_model = st.selectbox(
                "Select LLM Model",
                options=model_options,
                index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
            )
            st.session_state.selected_model = selected_model
        else:
            selected_model = st.text_input(
                "Model Name",
                value=st.session_state.selected_model,
                help="E.g. llama3.2, mistral, gemma"
            )
            st.session_state.selected_model = selected_model
            st.info("💡 No models auto-detected. Ensure you ran `ollama pull llama3.2`.")
    else:
        st.error("🔴 Ollama Disconnected")
        st.caption(conn_msg)
        selected_model = st.text_input(
            "Model Name",
            value=st.session_state.selected_model
        )
        st.session_state.selected_model = selected_model

    if st.button("🔄 Refresh Connection"):
        st.rerun()

    st.markdown("---")
    st.markdown("""
    **Quick Checklist:**
    1. Install [Ollama](https://ollama.com)
    2. Run `ollama serve`
    3. Run `ollama pull llama3.2`
    """)


# Header Section
st.title("⚡ FeedbackOps AI")
st.markdown(
    "**Intelligent Customer Feedback Operations, Pain Point Triage & Response Automation** "
    "powered by local, private LLM inference with evidence validation and refund safety."
)

# Ollama Unavailable Banner
if not is_connected:
    st.markdown(
        f"""
        <div class="setup-banner">
            <h4>⚠️ Local Ollama Service Offline</h4>
            <p>FeedbackOps AI requires a running local <b>Ollama</b> instance for inference to ensure complete data privacy and zero API costs.</p>
            <p><b>How to resolve this in 3 quick steps:</b></p>
            <ol>
                <li>Download and install Ollama for Windows from <b><a href="https://ollama.com" target="_blank">ollama.com</a></b>.</li>
                <li>Open Command Prompt or PowerShell and execute: <code>ollama pull {st.session_state.selected_model}</code></li>
                <li>Ensure the Ollama application is running, then click <b>Refresh Connection</b> in the left sidebar.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )


# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Single Review Analysis",
    "📊 Batch CSV Analysis",
    "📁 Sample Review Library",
    "⚙️ Setup & Diagnostics"
])


# ==========================================
# TAB 1: SINGLE REVIEW ANALYSIS
# ==========================================
with tab1:
    st.subheader("Analyze Single Customer Feedback")
    st.caption("Paste customer support tickets, store reviews, or survey comments to generate structured operational insights.")

    col_quick, col_space = st.columns([3, 1])
    with col_quick:
        st.markdown("**Quick Load Presets:**")
        preset_cols = st.columns(3)
        if preset_cols[0].button("🚨 Payment Bug"):
            st.session_state.single_review_text = SAMPLE_REVIEWS[0]["review"]
        if preset_cols[1].button("⚠️ Missing Header UX"):
            st.session_state.single_review_text = SAMPLE_REVIEWS[1]["review"]
        if preset_cols[2].button("🌟 Biometric Praise"):
            st.session_state.single_review_text = SAMPLE_REVIEWS[2]["review"]

    review_input = st.text_area(
        "Customer Review / Feedback Text",
        value=st.session_state.single_review_text,
        height=140,
        placeholder="Paste customer review here (e.g. 'Checkout failed three times during flash sale...')"
    )
    st.session_state.single_review_text = review_input

    col_btn, col_count = st.columns([1, 4])
    with col_btn:
        analyze_clicked = st.button("🚀 Analyze Feedback", type="primary", use_container_width=True)
    with col_count:
        char_count = len(review_input.strip())
        st.caption(f"Character count: **{char_count}** / 5000 max")

    if analyze_clicked:
        if not is_connected:
            st.error("Cannot run AI analysis because local Ollama service is disconnected. Please start Ollama first.")
        else:
            is_valid, validation_msg = utils.validate_text_input(review_input)
            if not is_valid:
                st.warning(f"⚠️ {validation_msg}")
            else:
                st.session_state.single_analysis_result = None
                analyzer = FeedbackAnalyzer(client=client, model=st.session_state.selected_model)
                
                with st.spinner(f"Warming up and running AI analysis using '{st.session_state.selected_model}' (Timeout: {st.session_state.ollama_timeout}s)... Please wait."):
                    try:
                        result = analyzer.analyze_review(review_input, warmup=True)
                        st.session_state.single_analysis_result = result
                        st.success("Analysis complete!")
                    except OllamaTimeoutError as te:
                        st.error(
                            f"⏱️ **Ollama Generation Timeout Error**: {str(te)}\n\n"
                            f"**Recommended Fix**: Increase the 'Read Timeout' in the sidebar (currently {st.session_state.ollama_timeout}s) or pull a faster model (e.g. `ollama pull llama3.2`)."
                        )
                    except OllamaConnectionError as ce:
                        st.error(f"🔌 **Ollama Connection Error**: {str(ce)}")
                    except OllamaModelNotFoundError as mnfe:
                        st.error(f"📦 **Model Not Found Error**: {str(mnfe)}")
                    except OllamaResponseError as re_err:
                        st.error(f"⚠️ **Ollama Response Error**: {str(re_err)}")
                    except OllamaError as oe:
                        st.error(f"⚠️ **Ollama Error**: {str(oe)}")
                    except Exception as ex:
                        st.error(f"Unexpected error during analysis: {str(ex)}")

    # Display Single Analysis Results
    res = st.session_state.single_analysis_result
    if res:
        st.markdown("---")
        st.subheader("📋 Operational Analysis Report")

        if res.get("security_flag"):
            st.info(f"🛡️ **Security Notice:** {res['security_flag']}")

        # Top KPI Metrics
        m1, m2, m3, m4 = st.columns(4)
        
        sent = res["sentiment"]
        sent_color = "positive" if sent == "Positive" else ("negative" if sent == "Negative" else "neutral")
        m1.markdown(
            f"""
            <div class="metric-card">
                <p>Sentiment</p>
                <h3><span class="badge badge-{sent_color}">{sent}</span></h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        m2.markdown(
            f"""
            <div class="metric-card">
                <p>Polarity Score</p>
                <h3>{res["sentiment_score"]:+.2f}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        urg = res["overall_urgency"]
        urg_color = urg.lower()
        m3.markdown(
            f"""
            <div class="metric-card">
                <p>Urgency Level</p>
                <h3><span class="badge badge-{urg_color}">{urg}</span></h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        m4.markdown(
            f"""
            <div class="metric-card">
                <p>Priority Score</p>
                <h3>{res["urgency_score"]} / 10</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(f"**Executive Summary:** {res['summary']}")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### 🎯 Identified Pain Points (Evidence-Grounded)")
            pain_points = res.get("pain_points", [])
            if pain_points:
                formatted_pps = []
                for p in pain_points:
                    grounding_label = "✅ Verified" if p.get("is_grounded", False) else "⚠️ Unverified"
                    formatted_pps.append({
                        "Issue / Topic": p.get("issue", ""),
                        "Category": p.get("category", ""),
                        "Urgency": p.get("urgency", ""),
                        "Affected Area": p.get("affected_area", ""),
                        "Evidence Quote (from Review)": f"\"{p.get('evidence_quote', '')}\"",
                        "Evidence Grounding": grounding_label
                    })
                pp_df = pd.DataFrame(formatted_pps)
                st.dataframe(pp_df, use_container_width=True, hide_index=True)
            else:
                st.info("No specific pain points identified in this feedback.")

            st.markdown("#### 🛠️ Recommended Action Items (Advisory Only)")
            st.caption("⚠️ Recommended action playbooks for human team review. Not completed business actions.")
            actions = res.get("action_items", [])
            if actions:
                for act in actions:
                    st.markdown(f"- 📋 {act}")
            else:
                st.caption("No specific action items generated.")

        with col_right:
            st.markdown("#### ✉️ Drafted Customer Response (Advisory Draft)")
            if res.get("was_sanitized"):
                st.warning(
                    "🛡️ **Safety Guardrail Active:** Potential unauthorized commitments or unapproved promises "
                    "were intercepted and replaced with a compliant acknowledgment."
                )
            
            draft_text = res.get("draft_response", "")
            st.text_area(
                "Response Template (Editable — Requires Staff Authorization Before Sending)",
                value=draft_text,
                height=220
            )
            st.caption("💡 Generated draft for human review. Never sent automatically without staff authorization.")


# ==========================================
# TAB 2: BATCH CSV ANALYSIS
# ==========================================
with tab2:
    st.subheader("Batch Review Triage & CSV Analysis")
    st.caption("Upload a CSV file containing customer reviews to process entire feedback datasets at scale.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    df_batch = None
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded file with **{len(df_batch)}** rows.")
        except Exception as e:
            st.error(f"Failed to read CSV file: {str(e)}")

    if df_batch is not None:
        columns = list(df_batch.columns)
        default_col = columns[0]
        for candidate in ["review", "feedback", "comment", "text", "description"]:
            for col in columns:
                if candidate in col.lower():
                    default_col = col
                    break

        col_select, col_preview = st.columns([1, 2])
        with col_select:
            selected_col = st.selectbox("Select Column Containing Reviews", options=columns, index=columns.index(default_col))
            max_rows = st.number_input("Max Rows to Process (0 = All)", min_value=0, max_value=len(df_batch), value=min(10, len(df_batch)))

        with col_preview:
            st.markdown("**Dataset Preview:**")
            st.dataframe(df_batch.head(3), use_container_width=True, hide_index=True)

        if st.button("🚀 Run Batch Analysis", type="primary"):
            if not is_connected:
                st.error("Cannot run batch AI analysis because local Ollama service is disconnected.")
            else:
                valid_csv, csv_msg, csv_stats = utils.validate_csv_dataframe(df_batch, selected_col)
                if not valid_csv:
                    st.warning(f"⚠️ {csv_msg}")
                else:
                    reviews_to_analyze = df_batch[selected_col].dropna().astype(str).tolist()
                    if max_rows > 0:
                        reviews_to_analyze = reviews_to_analyze[:max_rows]

                    progress_bar = st.progress(0, text="Initializing batch analysis...")
                    analyzer = FeedbackAnalyzer(client=client, model=st.session_state.selected_model)

                    def update_progress(current, total, status_msg="Processing..."):
                        pct = int((current / total) * 100) if total > 0 else 0
                        progress_bar.progress(max(0.0, min(1.0, current / total if total else 0)), text=f"{status_msg} ({current}/{total} - {pct}%)")

                    try:
                        batch_results = analyzer.analyze_batch(reviews_to_analyze, progress_callback=update_progress, warmup=True)
                        st.session_state.batch_analysis_results = batch_results
                        progress_bar.empty()
                        st.success(f"Batch analysis complete for {len(batch_results)} reviews!")
                    except OllamaTimeoutError as te:
                        st.error(f"⏱️ Batch processing timeout: {str(te)}")
                    except OllamaConnectionError as ce:
                        st.error(f"🔌 Batch processing connection error: {str(ce)}")
                    except Exception as ex:
                        st.error(f"Batch processing error: {str(ex)}")

    # Display Batch Analysis Dashboard
    batch_res = st.session_state.batch_analysis_results
    if batch_res:
        st.markdown("---")
        st.subheader("📊 Batch Triage Summary Dashboard")

        metrics = utils.compute_aggregate_metrics(batch_res)

        bm1, bm2, bm3, bm4 = st.columns(4)
        bm1.metric("Total Submitted", metrics["total_submitted"])
        bm2.metric("Successfully Analyzed", f"{metrics['successful_count']} ({round(metrics['successful_count']/metrics['total_submitted']*100, 1) if metrics['total_submitted'] else 0}%)")
        bm3.metric("Negative Reviews", f"{metrics['sentiment_counts']['Negative']} ({metrics['sentiment_percentages']['Negative']}% of successful)")
        bm4.metric("Evidence Grounding", f"{metrics['grounded_pain_points_pct']}% verified")

        c_chart1, c_chart2 = st.columns(2)

        with c_chart1:
            st.markdown("##### Sentiment Distribution (Successfully Analyzed)")
            sent_df = pd.DataFrame([
                {"Sentiment": k, "Count": v} for k, v in metrics["sentiment_counts"].items()
            ])
            chart_sent = alt.Chart(sent_df).mark_bar().encode(
                x=alt.X("Sentiment:N", sort=None),
                y="Count:Q",
                color=alt.Color("Sentiment:N", scale=alt.Scale(domain=["Positive", "Neutral", "Negative"], range=["#28a745", "#6c757d", "#dc3545"]))
            ).properties(height=240)
            st.altair_chart(chart_sent, use_container_width=True)

        with c_chart2:
            st.markdown("##### Urgency Breakdown")
            urg_df = pd.DataFrame([
                {"Urgency": k, "Count": v} for k, v in metrics["urgency_counts"].items()
            ])
            chart_urg = alt.Chart(urg_df).mark_bar().encode(
                x=alt.X("Urgency:N", sort=["Low", "Medium", "High", "Critical"]),
                y="Count:Q",
                color=alt.Color("Urgency:N", scale=alt.Scale(domain=["Low", "Medium", "High", "Critical"], range=["#28a745", "#ffc107", "#fd7e14", "#dc3545"]))
            ).properties(height=240)
            st.altair_chart(chart_urg, use_container_width=True)

        # Filterable Results Table
        st.markdown("##### Detailed Review Results")
        
        table_rows = []
        for idx, r in enumerate(batch_res, start=1):
            status = "✅ Success" if r.get("is_success", True) and "error" not in r else f"❌ Failed"
            table_rows.append({
                "ID": idx,
                "Status": status,
                "Review Text": r.get("raw_text", "")[:80] + "...",
                "Sentiment": r.get("sentiment", "Neutral"),
                "Urgency": r.get("overall_urgency", "Low"),
                "Urgency Score": r.get("urgency_score", 1),
                "Summary": r.get("summary", ""),
                "Draft Response": r.get("draft_response", "")[:100] + "...",
                "Safety Sanitized": "Yes" if r.get("was_sanitized", False) else "No"
            })

        results_df = pd.DataFrame(table_rows)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Export Options
        st.markdown("##### 📥 Export Batch Results")
        exp_col1, exp_col2, _ = st.columns([1, 1, 2])
        
        csv_data = utils.export_results_to_csv(batch_res)
        json_data = utils.export_results_to_json(batch_res)

        exp_col1.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name="feedbackops_analysis_report.csv",
            mime="text/csv"
        )

        exp_col2.download_button(
            label="📦 Download JSON",
            data=json_data,
            file_name="feedbackops_analysis_report.json",
            mime="application/json"
        )


# ==========================================
# TAB 3: SAMPLE REVIEW LIBRARY
# ==========================================
with tab3:
    st.subheader("📁 Preset Sample Review Library")
    st.info(
        "ℹ️ **Sample Dataset (Preset)**: These raw customer reviews are included for testing and demonstration purposes. "
        "Click any **Analyze Review** button to load and test the item in the single review tab."
    )

    sample_df = get_sample_dataframe()
    for item in SAMPLE_REVIEWS:
        with st.expander(f"📌 [{item['id']}] {item['category']} — {item['customer']} ({item['source']})"):
            st.write(f"**Date:** {item['date']}")
            st.write(f"**Feedback:** \"{item['review']}\"")
            if st.button(f"Analyze [{item['id']}] in Single Tab", key=f"btn_sample_{item['id']}"):
                st.session_state.single_review_text = item['review']
                st.session_state.single_analysis_result = None
                st.success(f"Loaded {item['id']} into Single Review tab! Switch tabs above to analyze.")


# ==========================================
# TAB 4: SETUP & DIAGNOSTICS
# ==========================================
with tab4:
    st.subheader("⚙️ System Status & Ollama Installation Guide")

    st.markdown("#### Diagnostic Status")
    diag_col1, diag_col2 = st.columns(2)
    diag_col1.markdown(f"**Configured Host:** `{st.session_state.ollama_host}`")
    diag_col1.markdown(f"**Selected Model:** `{st.session_state.selected_model}`")
    diag_col1.markdown(f"**Configured Read Timeout:** `{st.session_state.ollama_timeout} seconds`")
    diag_col1.markdown(f"**Safety Guardrails:** `Active (Refund Protection, Evidence Validation, Injection Neutralizer)`")
    
    if is_connected:
        diag_col2.success("🟢 Connection Status: Active & Operational")
    else:
        diag_col2.error("🔴 Connection Status: Service Unreachable")

    st.markdown("---")
    st.markdown("#### 📖 Windows Setup & Launch Instructions")
    
    st.markdown("""
    ### 1. Install Ollama on Windows
    - Download the Windows installer from **[ollama.com/download](https://ollama.com/download)**.
    - Run the installer wizard and complete installation.

    ### 2. Pull a Supported Model
    Open Command Prompt (CMD) or Windows PowerShell and run:
    ```bash
    ollama pull llama3.2
    ```
    *(Alternative supported models: `ollama pull mistral`, `ollama pull gemma2`, `ollama pull phi3`)*

    ### 3. Launch FeedbackOps AI Locally
    Navigate to the project root folder in PowerShell and execute:
    ```bash
    streamlit run app.py
    ```

    ### 4. Cloud & Production Deployment Notes
    - **Streamlit Community Cloud**: Because Streamlit Community Cloud runs on remote servers, `localhost:11434` is not accessible directly unless you point `OLLAMA_HOST` to a publicly accessible Ollama server or tunnel (e.g. ngrok/Cloudflare tunnel), or use an API gateway proxy.
    - **Zero Data Exposure**: Running locally ensures customer feedback text never leaves your local network.
    """)
