from __future__ import annotations

import json
from datetime import datetime

import streamlit as st

from drug_safety_assistant.config import settings
from drug_safety_assistant.pipeline.orchestrator import DrugSafetyAssistant
from drug_safety_assistant.types import SafetyRequest, StructuredResponse

st.set_page_config(page_title="Drug Safety Assistant", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap');

      :root {
        --ink: #0e2231;
        --muted: #4f616d;
        --paper: #f4f8f7;
        --surface: #ffffff;
        --line: #d6e2e0;
        --accent: #0b7b81;
        --accent-strong: #045d62;
        --high: #ad3f00;
        --moderate: #8b6f00;
        --low: #21683d;
      }

      .stApp {
        background:
          radial-gradient(
            circle at 8% 10%,
            #d9f1ee 0%,
            rgba(217, 241, 238, 0.18) 35%,
            transparent 60%
          ),
          radial-gradient(
            circle at 95% 3%,
            #fff1dc 0%,
            rgba(255, 241, 220, 0.2) 25%,
            transparent 50%
          ),
          var(--paper);
      }

      .main .block-container {
        max-width: 1160px;
        padding-top: 1.25rem;
        padding-bottom: 2rem;
      }

      .hero-card {
        background: linear-gradient(125deg, #0f2d43 0%, #124d63 52%, #25706d 100%);
        border-radius: 18px;
        color: #f6feff;
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 1.2rem 1.3rem;
        margin-bottom: 1rem;
      }

      .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 0.01em;
      }

      .hero-subtitle {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.96rem;
        margin-top: 0.4rem;
        line-height: 1.45;
        color: #d8ecf2;
      }

      .risk-banner {
        border: 1px solid var(--line);
        background: var(--surface);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin: 0.2rem 0 1rem 0;
      }

      .risk-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.28rem 0.72rem;
        color: #fff;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-bottom: 0.35rem;
      }

      .risk-summary {
        color: var(--ink);
        font-size: 1.02rem;
        font-weight: 600;
      }

      .followup-card {
        background: #fff;
        border: 1px solid #f1d9b4;
        border-left: 5px solid #d0812a;
        border-radius: 10px;
        padding: 0.62rem 0.78rem;
        margin-bottom: 0.52rem;
      }

      .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
      }

      .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Sans', sans-serif;
        border-radius: 8px 8px 0 0;
        border: 1px solid #cfd9d7;
        padding: 0.35rem 0.7rem;
        background: #eaf0ef;
      }

      .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        border-bottom-color: #ffffff !important;
      }

      @media (max-width: 900px) {
        .hero-title {
          font-size: 1.28rem;
        }
        .hero-subtitle {
          font-size: 0.88rem;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _risk_color(level: str) -> str:
    if level == "High":
        return "var(--high)"
    if level == "Moderate":
        return "var(--moderate)"
    return "var(--low)"


def _risk_fraction(score: float) -> float:
    if score <= 3:
        return min(max(score / 3.0, 0.0), 1.0)
    return min(max(score / 100.0, 0.0), 1.0)


def _backend_statuses() -> tuple[str, str, str, str]:
    llm_backend = "Deterministic fallback"
    if settings.anthropic_api_key:
        llm_backend = f"Anthropic ({settings.anthropic_model})"
    elif settings.nvidia_api_key:
        llm_backend = f"NVIDIA ({settings.nvidia_model})"

    extract_backend = "Deterministic fallback"
    if settings.nvidia_api_key:
        extract_backend = f"NVIDIA ({settings.nvidia_extract_model})"

    medcpt_mode = "enabled" if settings.enable_medcpt_models else "hash fallback"
    persistent_mode = "enabled" if settings.use_persistent_index else "disabled"
    return llm_backend, extract_backend, medcpt_mode, persistent_mode


def _request_payload(
    question: str,
    drug: str,
    drug_a: str,
    drug_b: str,
    age_group: str,
    pregnancy_status: str,
    trimester: str,
    kidney_status: str,
    liver_status: str,
    current_meds_text: str,
) -> SafetyRequest:
    meds = [item.strip() for item in current_meds_text.split(",") if item.strip()]
    return SafetyRequest(
        question=question,
        drug=drug or None,
        drug_a=drug_a or None,
        drug_b=drug_b or None,
        age_group=age_group or None,
        pregnancy_status=pregnancy_status or None,
        trimester=trimester or None,
        kidney_status=kidney_status or None,
        liver_status=liver_status or None,
        current_meds=meds,
    )


def _push_history(question: str, response: StructuredResponse) -> None:
    history = st.session_state.setdefault("assessment_history", [])
    history.insert(
        0,
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "risk_level": response.risk_level.value,
            "risk_score": response.risk_score,
            "citations": len(response.evidence_sources),
        },
    )
    st.session_state["assessment_history"] = history[:8]


@st.cache_resource(show_spinner=False)
def _get_assistant() -> DrugSafetyAssistant:
    return DrugSafetyAssistant()


assistant = _get_assistant()
llm_backend, extract_backend, medcpt_mode, persistent_mode = _backend_statuses()

st.markdown(
    """
    <div class="hero-card">
      <div class="hero-title">Evidence-Grounded Drug Safety Decision Support</div>
      <div class="hero-subtitle">
        Patient-specific, citation-backed safety analysis using OpenFDA labels, PubMed literature,
        and FAERS post-marketing signals. The assistant asks only minimum necessary
        context questions and avoids PII collection.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("System Status")
st.sidebar.write(f"Generation backend: {llm_backend}")
st.sidebar.write(f"Intent extraction: {extract_backend}")
st.sidebar.write(f"MedCPT embeddings: {medcpt_mode}")
st.sidebar.write(f"Persistent index: {persistent_mode}")
st.sidebar.divider()
st.sidebar.caption("This tool supports clinical decision-making. It does not prescribe dosing.")

with st.form("drug_safety_form"):
    input_col, context_col = st.columns([1.25, 1.0], gap="large")
    with input_col:
        question = st.text_area(
            "Drug safety question",
            key="question_input",
            height=116,
            placeholder="Example: Can benzoyl peroxide topical be used with dapsone 7.5% gel?",
        )
        drug = st.text_input("Primary drug", key="primary_drug_input")
        pair_col1, pair_col2 = st.columns(2)
        with pair_col1:
            drug_a = st.text_input("Drug A (interaction)", key="drug_a_input")
        with pair_col2:
            drug_b = st.text_input("Drug B (interaction)", key="drug_b_input")
        current_meds_text = st.text_area(
            "Current medications (comma-separated)",
            key="meds_input",
            placeholder="metformin, lisinopril, aspirin",
            height=92,
        )

    with context_col:
        age_group = st.selectbox("Age group", ["", "pediatric", "adult", "over 65"], key="age")
        pregnancy_status = st.selectbox(
            "Pregnancy status",
            ["", "no", "yes", "unknown"],
            key="pregnancy",
        )
        trimester = st.selectbox("Trimester", ["", "1", "2", "3"], key="trimester")
        kidney_status = st.selectbox(
            "Kidney disease",
            ["", "none", "CKD", "dialysis"],
            key="kidney",
        )
        liver_status = st.selectbox(
            "Liver disease",
            ["", "none", "mild", "moderate", "severe"],
            key="liver",
        )

    submitted = st.form_submit_button("Assess Safety", use_container_width=True)

if submitted:
    if not question.strip():
        st.error("Enter a drug safety question to continue.")
    else:
        request = _request_payload(
            question=question,
            drug=drug,
            drug_a=drug_a,
            drug_b=drug_b,
            age_group=age_group,
            pregnancy_status=pregnancy_status,
            trimester=trimester,
            kidney_status=kidney_status,
            liver_status=liver_status,
            current_meds_text=current_meds_text,
        )

        with st.spinner("Retrieving evidence, validating claims, and scoring risk..."):
            response = assistant.assess(request)

        st.session_state["latest_request"] = request.model_dump()
        st.session_state["latest_response"] = response.model_dump()
        _push_history(question, response)

if "latest_response" in st.session_state:
    response = StructuredResponse(**st.session_state["latest_response"])
    risk_color = _risk_color(response.risk_level.value)

    if response.follow_up_questions:
        st.warning("Additional clinical context is required before evidence retrieval.")
        for followup in response.follow_up_questions:
            st.markdown(f"<div class='followup-card'>{followup}</div>", unsafe_allow_html=True)
    else:
        risk_banner = (
            "<div class='risk-banner'>"
            f"<span class='risk-pill' style='background:{risk_color};'>"
            f"{response.risk_level.value} risk</span>"
            f"<div class='risk-summary'>Risk score: {response.risk_score:.2f}</div>"
            "</div>"
        )
        st.markdown(
            risk_banner,
            unsafe_allow_html=True,
        )
        st.progress(
            _risk_fraction(response.risk_score),
            text=f"{response.risk_level.value} ({response.risk_score:.2f})",
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Risk Level", response.risk_level.value)
        m2.metric("Evidence Sources", str(len(response.evidence_sources)))
        m3.metric("Support Ratio", f"{response.guard_supported_ratio:.0%}")
        m4.metric("Intent", response.intent.value.replace("_", " ").title())

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Clinical Output", "Evidence Panel", "Traceability", "Export"]
        )

        with tab1:
            st.subheader("Safety Summary")
            st.write(response.safety_summary)
            st.subheader("Monitoring Recommendations")
            for item in response.monitoring_recommendations:
                st.write(f"- {item}")

        with tab2:
            if not response.evidence_sources:
                st.info("No evidence sources to display.")
            else:
                for citation in response.evidence_sources:
                    with st.expander(f"{citation.citation_id} - {citation.title}", expanded=False):
                        st.write(f"Source: {citation.source}")
                        st.write(citation.details)

        with tab3:
            st.subheader("Uncertainty Statement")
            st.write(response.uncertainty_statement)
            st.subheader("Guardrail Confidence")
            st.progress(
                min(max(response.guard_supported_ratio, 0.0), 1.0),
                text=f"Supported claim ratio: {response.guard_supported_ratio:.2%}",
            )
            if "latest_request" in st.session_state:
                st.subheader("Request Snapshot")
                st.json(st.session_state["latest_request"])

        with tab4:
            response_json = json.dumps(response.model_dump(), indent=2)
            response_md = response.to_markdown()
            st.download_button(
                label="Download JSON",
                data=response_json,
                file_name="drug_safety_response.json",
                mime="application/json",
                use_container_width=True,
            )
            st.download_button(
                label="Download Markdown",
                data=response_md,
                file_name="drug_safety_response.md",
                mime="text/markdown",
                use_container_width=True,
            )

if st.session_state.get("assessment_history"):
    st.sidebar.divider()
    st.sidebar.subheader("Recent Assessments")
    for item in st.session_state["assessment_history"]:
        st.sidebar.caption(
            f"{item['time']} | "
            f"{item['risk_level']} ({item['risk_score']:.2f}) | "
            f"{item['citations']} cites"
        )
        st.sidebar.write(item["question"][:80] + ("..." if len(item["question"]) > 80 else ""))
