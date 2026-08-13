"""
Phase 5: Streamlit dashboard for TEP fault detection + diagnosis.

Reads models produced by src/train.py (models/detection_<type>.joblib,
models/diagnosis_<type>.joblib) and the processed training parquet
(data/processed/tep_training.parquet), then rebuilds the same by-run
validation split used during training (split_by_run, same seed) so the
dashboard is always looking at data the selected models weren't fit on.

NOTE: streamlit and plotly aren't installed in the sandbox this was
written in (no network access to add them), so this file is written
carefully against documented APIs but hasn't been run end-to-end here --
same caveat as the rest of the repo until you run it against the real
data on your machine. Run with:

    streamlit run app.py

Four tabs:
  1. Detection Timeline -- actual vs. predicted fault_active over time
     for one simulation run, with the true onset marked.
  2. Diagnosis -- confusion matrix + per-fault metrics for the multiclass
     model, restricted to active-fault rows (matches evaluate_diagnosis_model).
  3. Feature Importance -- top sensors driving each model.
  4. Live Monitoring -- feeds one run's timesteps in one at a time to
     simulate a real-time monitoring view.
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent / "src"))
from features import SENSOR_COLUMNS, split_by_run  # noqa: E402

MODELS_DIR = Path("models")
DATA_PATH = Path("data/processed/tep_training.parquet")
SEED = 42  # must match the seed train.py used, so the val split lines up

st.set_page_config(page_title="TEP Fault Dashboard", layout="wide")


# ---------------------------------------------------------------- loaders --
@st.cache_resource
def load_models(model_type: str):
    det_path = MODELS_DIR / f"detection_{model_type}.joblib"
    diag_path = MODELS_DIR / f"diagnosis_{model_type}.joblib"
    if not det_path.exists() or not diag_path.exists():
        return None, None
    return joblib.load(det_path), joblib.load(diag_path)


@st.cache_data
def load_split():
    if not DATA_PATH.exists():
        return None, None
    df = pd.read_parquet(DATA_PATH)
    train_df, val_df = split_by_run(df, val_frac=0.2, seed=SEED)
    return train_df, val_df


def feature_columns_for(df: pd.DataFrame) -> list[str]:
    return [c for c in SENSOR_COLUMNS if c in df.columns]


# ------------------------------------------------------------- sidebar ----
st.sidebar.title("TEP Fault Dashboard")
model_type = st.sidebar.selectbox(
    "Model type", ["random_forest", "hist_gradient_boosting"],
    help="Must match a model_type you actually trained with src/train.py",
)

detection_model, diagnosis_model = load_models(model_type)
_, val_df = load_split()

if val_df is None:
    st.error(
        f"Couldn't find {DATA_PATH}. Run src/data_loader.py then "
        "src/train.py first to generate processed data and models."
    )
    st.stop()

if detection_model is None or diagnosis_model is None:
    st.error(
        f"No saved models for model_type='{model_type}' in {MODELS_DIR}/. "
        f"Run: python src/train.py --model-type {model_type} --class-weight balanced"
    )
    st.stop()

feature_columns = feature_columns_for(val_df)

fault_numbers = sorted(val_df["faultNumber"].unique())
selected_fault = st.sidebar.selectbox("Fault scenario", fault_numbers)

runs_for_fault = sorted(
    val_df.loc[val_df["faultNumber"] == selected_fault, "simulationRun"].unique()
)
selected_run = st.sidebar.selectbox("Simulation run", runs_for_fault)

run_df = (
    val_df[(val_df["faultNumber"] == selected_fault) & (val_df["simulationRun"] == selected_run)]
    .sort_values("sample")
    .reset_index(drop=True)
)

tab_detect, tab_diag, tab_importance, tab_live = st.tabs(
    ["Detection Timeline", "Diagnosis", "Feature Importance", "Live Monitoring"]
)


# ------------------------------------------------------ detection tab -----
with tab_detect:
    st.subheader(f"Fault {selected_fault}, run {selected_run} -- detection over time")

    preds = detection_model.predict(run_df[feature_columns])
    proba = None
    if hasattr(detection_model, "predict_proba"):
        proba = detection_model.predict_proba(run_df[feature_columns])[:, 1]

    onset_idx = None
    if run_df["fault_active"].any():
        onset_idx = int(run_df.index[run_df["fault_active"]][0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=run_df["sample"], y=run_df["fault_active"].astype(int),
        mode="lines", name="actual", line=dict(color="black", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=run_df["sample"], y=preds,
        mode="lines", name="predicted", line=dict(color="crimson", dash="dot"),
    ))
    if proba is not None:
        fig.add_trace(go.Scatter(
            x=run_df["sample"], y=proba,
            mode="lines", name="predicted probability",
            line=dict(color="crimson", width=1), opacity=0.4, yaxis="y",
        ))
    if onset_idx is not None:
        fig.add_vline(
            x=run_df.loc[onset_idx, "sample"], line_dash="dash", line_color="gray",
            annotation_text="true onset",
        )
    fig.update_layout(
        yaxis_title="fault_active (0/1) / probability", xaxis_title="sample",
        height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    correct = (preds == run_df["fault_active"].astype(int)).mean()
    col1, col2, col3 = st.columns(3)
    col1.metric("Timestep accuracy (this run)", f"{correct:.1%}")
    col2.metric("True onset sample", run_df.loc[onset_idx, "sample"] if onset_idx is not None else "n/a")
    first_flag = run_df.index[preds.astype(bool)]
    col3.metric(
        "First flagged sample",
        run_df.loc[first_flag[0], "sample"] if len(first_flag) else "never flagged",
    )


# --------------------------------------------------------- diagnosis tab --
with tab_diag:
    st.subheader(f"Diagnosis model ({model_type}) -- held-out validation, active-fault rows only")

    active_val = val_df[val_df["fault_active"]]
    diag_preds = diagnosis_model.predict(active_val[feature_columns])
    labels = sorted(active_val["faultNumber"].unique())

    cm = confusion_matrix(active_val["faultNumber"], diag_preds, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig_cm = px.imshow(
        cm_norm, x=[str(l) for l in labels], y=[str(l) for l in labels],
        color_continuous_scale="Blues", labels=dict(x="predicted", y="actual", color="row %"),
        text_auto=".0%", height=650,
    )
    fig_cm.update_xaxes(side="top")
    st.plotly_chart(fig_cm, use_container_width=True)
    st.caption(
        "Row-normalized. Faults 3, 9, and 15 are literature-documented as "
        "near-undetectable in TEP -- weak diagonal there is expected, not a bug."
    )


# ------------------------------------------------------- importance tab --
with tab_importance:
    st.subheader("Feature importance")

    def show_importance(model, X, y, title):
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=feature_columns)
        else:
            st.caption(f"{title}: computing permutation importance (no native "
                       "feature_importances_ on this model type) -- may take a moment.")
            sample = X.sample(min(2000, len(X)), random_state=SEED)
            result = permutation_importance(
                model, sample, y.loc[sample.index], n_repeats=5, random_state=SEED, n_jobs=-1
            )
            imp = pd.Series(result.importances_mean, index=feature_columns)
        top = imp.sort_values(ascending=False).head(20)
        fig = px.bar(top[::-1], orientation="h", labels={"value": "importance", "index": "feature"})
        fig.update_layout(showlegend=False, height=500, title=title)
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        show_importance(
            detection_model, val_df[feature_columns], val_df["fault_active"].astype(int),
            "Detection model",
        )
    with col_b:
        show_importance(
            diagnosis_model, active_val[feature_columns], active_val["faultNumber"],
            "Diagnosis model",
        )


# ------------------------------------------------------------ live tab ---
with tab_live:
    st.subheader(f"Live monitoring simulation -- fault {selected_fault}, run {selected_run}")
    st.caption(
        "Feeds this run's timesteps in one at a time, as if streaming from the "
        "process, and shows what the detection model would have flagged at "
        "each moment. Use Step for manual control, or Play for a walkthrough."
    )

    if "live_idx" not in st.session_state:
        st.session_state.live_idx = 0
    if "live_run_key" not in st.session_state or st.session_state.live_run_key != (selected_fault, selected_run):
        st.session_state.live_idx = 0
        st.session_state.live_run_key = (selected_fault, selected_run)

    n_samples = len(run_df)
    key_sensors = st.multiselect(
        "Sensors to plot", feature_columns, default=feature_columns[:2],
        help="Pick 1-3 sensors to watch on the live chart",
    )

    btn_cols = st.columns(4)
    if btn_cols[0].button("Reset"):
        st.session_state.live_idx = 0
    if btn_cols[1].button("Step"):
        st.session_state.live_idx = min(st.session_state.live_idx + 1, n_samples - 1)
    play = btn_cols[2].checkbox("Play")
    speed = btn_cols[3].slider("Speed (s/step)", 0.05, 1.0, 0.2, label_visibility="collapsed")

    idx = st.session_state.live_idx
    visible = run_df.iloc[: idx + 1]

    status_placeholder = st.empty()
    chart_placeholder = st.empty()
    table_placeholder = st.empty()

    def render(visible_df):
        current = visible_df.iloc[-1]
        pred = detection_model.predict(current[feature_columns].to_frame().T)[0]
        if pred:
            status_placeholder.error(f"Sample {current['sample']}: FAULT FLAGGED")
        else:
            status_placeholder.success(f"Sample {current['sample']}: normal")

        if key_sensors:
            fig_live = go.Figure()
            for sensor in key_sensors:
                fig_live.add_trace(go.Scatter(
                    x=visible_df["sample"], y=visible_df[sensor], mode="lines", name=sensor,
                ))
            fig_live.update_layout(height=350, xaxis_title="sample", yaxis_title="reading")
            chart_placeholder.plotly_chart(fig_live, use_container_width=True)

        table_placeholder.dataframe(
            visible_df[["sample", "fault_active"] + key_sensors].tail(5),
            use_container_width=True,
        )

    render(visible)

    if play:
        # Blocking walkthrough for the remaining samples. This locks up
        # other widget interaction until it finishes or hits the end --
        # fine for a portfolio demo, not a pattern to reuse in a real app.
        for i in range(idx + 1, n_samples):
            st.session_state.live_idx = i
            render(run_df.iloc[: i + 1])
            time.sleep(speed)
        st.rerun()