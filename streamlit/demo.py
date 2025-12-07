import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os

st.set_page_config(page_title="IncidentGrade", layout="wide")

DATA_PATH = "streamlit.feather"
MODEL_PATH = "XGB_small.json"
SEED = 42

ASSETS_DIR = "assets"
IMAGES = {
    2: os.path.join(ASSETS_DIR, "hacker.png"),   # class 2 -> TruePositive
    1: os.path.join(ASSETS_DIR, "admin.webp"),    # class 1 -> BenignPositive
    0: os.path.join(ASSETS_DIR, "user.png"),       # class 0 -> FalsePositive
}

LABEL_MAP = {
    2: "Hacker (2)",
    1: "Admin (1)",
    0: "User (0)"
}

st.title("🔎 IncidentGrade prediction")


@st.cache_data(show_spinner=False)
def load_data(data_path: str):
    df = pd.read_feather(data_path)
    return df

@st.cache_resource(show_spinner=False)
def load_xgb_model(path: str):
    booster = xgb.Booster()
    booster.load_model(path)
    return booster


df_sample = load_data(DATA_PATH)
xgb_model = load_xgb_model(MODEL_PATH)
feature_cols = [c for c in df_sample.columns if c != "IncidentGrade"]

with st.sidebar:
    st.markdown("### Controls")
    st.markdown(f"- dataset rows used: **{len(df_sample):,}**")
    st.markdown(f"- model: `{MODEL_PATH}`")
    st.markdown("---")
    if st.button("Pick a random row and predict 🎲"):
        do_pick = True
    else:
        do_pick = False


col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Prediction row")
    st.info("Press the button in the sidebar to pick a random row and predict.")
    truth_display = st.empty()
    pred_display = st.empty()
    diff_display = st.empty()
    df_preview = st.empty()

with col_right:
    st.subheader("Class visual")
    img_spot = st.empty()
    anim_spot = st.empty()


if "click_counter" not in st.session_state:
    st.session_state["click_counter"] = 0

if do_pick:
    st.session_state["click_counter"] += 1
    chosen = df_sample.sample(n=1,random_state=SEED + st.session_state["click_counter"]).iloc[0]
    true_label = int(chosen["IncidentGrade"])
    truth_display.markdown(f"**True IncidentGrade:** `{true_label}` — {LABEL_MAP.get(true_label, str(true_label))}")
    X_row = chosen[feature_cols].to_frame().T.copy()
    X_row = X_row.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    dmat = xgb.DMatrix(X_row)
    
    raw_preds = xgb_model.predict(dmat)
    if raw_preds.ndim == 1:
        pred_label = int((raw_preds > 0.5).astype(int)[0])
    else:
        pred_label = int(np.argmax(raw_preds, axis=1)[0])
    
    if pred_label is not None:
        pred_display.markdown(f"**Predicted IncidentGrade:** `{pred_label}` — {LABEL_MAP.get(pred_label, str(pred_label))}")
        
        img_path = IMAGES.get(pred_label)
        if img_path and os.path.exists(img_path):
            img_spot.image(img_path, use_container_width=True,
                            caption=LABEL_MAP.get(pred_label))
        else:
            img_spot.markdown(f"**{LABEL_MAP.get(pred_label)}**")
        
        correct = (pred_label == true_label)
        if correct:
            diff_display.success("✅ Prediction matches the true label!")
        else:
            diff_display.error("❌ Prediction does NOT match the true label.")
            shake_html = """
            <style>
            @keyframes shake {
                0% { transform: translateX(0); }
                20% { transform: translateX(-8px); }
                40% { transform: translateX(8px); }
                60% { transform: translateX(-4px); }
                80% { transform: translateX(4px); }
                100% { transform: translateX(0); }
            }
            .shake {
                display:inline-block;
                animation: shake 0.8s;
            }
            </style>
            <div class="shake">Prediction mismatch</div>
            """
            anim_spot.markdown(shake_html, unsafe_allow_html=True)


st.markdown("---")
st.markdown("### Development notebooks")
st.write("- Notebook on preprocessing:", '123')
st.write("- Notebook on modelling:", '123')
st.write("- Notebook on EDA:", '123')
st.markdown("---")
st.markdown("Made by **Viktor Korotkov**")

