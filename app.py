"""
app.py
Interactive Streamlit Web Application for Structural Damage Image Classification.
Mandatory Deliverable: Live Demonstration for UE24CS352A Review Session.
Problem Statement #39: Structural Damage Image Classification (Stanford CS229 Benchmark)
"""

import os
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import joblib

st.set_page_config(
    page_title="Structural Damage Classifier (Classical ML)",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Structural Damage Image Classifier")
st.markdown("""
**Course:** UE24CS352A - Machine Learning Mini-Project  
**Problem Statement #39:** Automated Structural Health Monitoring & Post-Earthquake Damage Recognition  
**Reference:** Stanford CS229 (*Stephanie Ho & Christopher Troncoso*)  
**Methodology:** Continuous Raw Pixel Vectors ($d = 2,304$ features in $[-1.0, 1.0]$) + Classical Supervised Learning
""")

# Model Candidate Locations
def find_model_path(filenames):
    candidates = []
    for fn in filenames:
        candidates.extend([
            os.path.join("outputs", "models", fn),
            os.path.join("models", fn),
            fn
        ])
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]

available_models = {
    "Support Vector Machine (RBF Kernel)": find_model_path(["SVM_RBF_Kernel.joblib", "SVM_(RBF_Kernel).joblib", "SVM__RBF_Kernel_.joblib"]),
    "Logistic Regression (L2)": find_model_path(["Logistic_Regression_L2.joblib", "Logistic_Regression_(L2).joblib", "Logistic_Regression__L2_.joblib"]),
    "Decision Tree (ID3 / Entropy)": find_model_path(["Decision_Tree_ID3___Entropy.joblib", "Decision_Tree_(ID3___Entropy).joblib", "Decision_Tree__ID3___Entropy_.joblib"])
}

# Sidebar Model Selection
st.sidebar.header("Model Selection & Settings")
selected_model_name = st.sidebar.selectbox("Active Inference Model", list(available_models.keys()))
model_path = available_models[selected_model_name]

# Helper raw pixel preprocessor
def get_pixel_vector(img_pil, target_size=(48, 48)):
    img = np.array(img_pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    # Continuous normalization to [-1.0, 1.0] matching Stanford CS229
    normalized = (resized.astype(np.float32) / 127.5) - 1.0
    return normalized.flatten().reshape(1, -1), resized

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Select or Upload Structural Image")

    # Sample folders
    sample_dirs = ["sample_images", "sample_dataset_images"]
    chosen_dir = "sample_images"
    for sd in sample_dirs:
        if os.path.exists(os.path.join(sd, "Damaged")):
            chosen_dir = sd
            break

    sample_options = ["Custom Upload"]
    peer_samples = {}

    dmg_dir = os.path.join(chosen_dir, "Damaged")
    if os.path.exists(dmg_dir):
        for f in sorted(os.listdir(dmg_dir)):
            if f.endswith((".jpg", ".png")):
                clean_title = f.replace(".jpg", "").replace(".png", "").replace("_", " ")
                name = f"🚨 Damaged: {clean_title}"
                peer_samples[name] = os.path.join(dmg_dir, f)
                sample_options.append(name)

    undmg_dir = os.path.join(chosen_dir, "Undamaged")
    if os.path.exists(undmg_dir):
        for f in sorted(os.listdir(undmg_dir)):
            if f.endswith((".jpg", ".png")):
                clean_title = f.replace(".jpg", "").replace(".png", "").replace("_", " ")
                name = f"✅ Undamaged: {clean_title}"
                peer_samples[name] = os.path.join(undmg_dir, f)
                sample_options.append(name)

    chosen_sample = st.selectbox("Choose a real benchmark example from PEER Hub, or upload custom photo:", sample_options)

    img_input = None
    if chosen_sample != "Custom Upload" and chosen_sample in peer_samples:
        img_input = Image.open(peer_samples[chosen_sample])
        st.image(img_input, caption=f"Selected Benchmark: {chosen_sample}", use_container_width=True)
    else:
        uploaded_file = st.file_uploader(
            "Upload a photo of civil infrastructure (concrete facade, pillar, wall)",
            type=["jpg", "jpeg", "png", "bmp"]
        )
        if uploaded_file is not None:
            img_input = Image.open(uploaded_file)
            st.image(img_input, caption="Uploaded Structural Photo", use_container_width=True)

with col2:
    st.subheader("2. Model Diagnostics & Classification")
    if img_input is not None:
        feat_vector, resized_gray = get_pixel_vector(img_input)

        st.write(f"**Normalized Pixel Matrix ($48 \\times 48 = {feat_vector.shape[1]}$ continuous features):**")
        st.image(resized_gray, caption="Preprocessed Grayscale Input", width=160)

        # Load all available models for multi-model consensus
        all_loaded = {}
        for m_name, p in available_models.items():
            if os.path.exists(p):
                try:
                    all_loaded[m_name] = joblib.load(p)
                except Exception:
                    pass

        if selected_model_name in all_loaded:
            model = all_loaded[selected_model_name]

            # Predict active model
            pred = model.predict(feat_vector)[0]
            confidence = 0.5

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(feat_vector)[0]
                confidence = proba[pred]
            elif hasattr(model, "decision_function"):
                df_val = model.decision_function(feat_vector)[0]
                prob1 = 1.0 / (1.0 + np.exp(-df_val))
                confidence = prob1 if pred == 1 else (1.0 - prob1)

            st.markdown("---")
            st.subheader(f"Active Model: {selected_model_name}")
            if pred == 1:
                st.error("🚨 **STRUCTURAL DAMAGE DETECTED (Class 1)**")
            else:
                st.success("✅ **NO STRUCTURAL DAMAGE - INTACT (Class 0)**")

            st.write(f"**Confidence / Posterior Probability:** `{confidence * 100:.2f}%`")
            st.progress(float(confidence))

            # Cross-model consensus summary
            st.markdown("##### Multi-Model Hard Voting Consensus:")
            res_cols = st.columns(3)
            votes = 0
            for idx, (m_name, m_inst) in enumerate(all_loaded.items()):
                m_pred = m_inst.predict(feat_vector)[0]
                votes += int(m_pred)
                m_prob = 0.5
                if hasattr(m_inst, "predict_proba"):
                    m_prob = m_inst.predict_proba(feat_vector)[0][1]
                elif hasattr(m_inst, "decision_function"):
                    df_v = m_inst.decision_function(feat_vector)[0]
                    m_prob = 1.0 / (1.0 + np.exp(-df_v))

                with res_cols[idx]:
                    short_name = m_name.split()[0]
                    if m_pred == 1:
                        st.metric(label=short_name, value="DAMAGED", delta=f"{m_prob*100:.1f}% dam", delta_color="inverse")
                    else:
                        st.metric(label=short_name, value="INTACT", delta=f"{(1-m_prob)*100:.1f}% intact", delta_color="normal")

            consensus_status = "🚨 DAMAGE DETECTED" if votes >= 2 else "✅ UNDAMAGED"
            st.info(f"**Ensemble Consensus Result:** {consensus_status} ({votes}/3 Models Agree)")

        else:
            st.warning(f"⚠️ Model weight not found at `{model_path}`.")
            st.info("Run `python3 main.py` to train all models and generate outputs.")
    else:
        st.info("👈 Choose a PEER dataset example from the dropdown or upload your own image.")

# --- Matplotlib Section: Dynamic ROC & AUC Analysis ---
st.markdown("---")
st.subheader("3. Model Performance Diagnostics: ROC Curves & AUC Analysis")
st.caption("Evaluates True Positive Rate (Sensitivity) vs False Positive Rate (1 - Specificity) across decision thresholds.")

roc_candidates = [
    "outputs/models/roc_data.joblib",
    "models/roc_data.joblib"
]
roc_path = next((p for p in roc_candidates if os.path.exists(p)), None)

if roc_path:
    roc_data = joblib.load(roc_path)
    fig, axes = plt.subplots(1, len(roc_data), figsize=(15, 4.2), dpi=150)
    if len(roc_data) == 1:
        axes = [axes]

    colors = {
        'Logistic Regression (L2)': '#1f77b4',
        'Decision Tree (ID3 - Entropy)': '#2ca02c',
        'Decision Tree (ID3 / Entropy)': '#2ca02c',
        'SVM (RBF Kernel)': '#d62728'
    }

    for ax, (name, data) in zip(axes, roc_data.items()):
        is_selected = (name == selected_model_name or name.split()[0] in selected_model_name)
        line_color = colors.get(name, '#1f77b4')
        line_width = 3.0 if is_selected else 2.0

        ax.plot(data['fpr'], data['tpr'], color=line_color, lw=line_width,
                label=f'ROC (AUC = {data["auc"]:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Random (AUC = 0.50)')

        title_suffix = " ★ (Active)" if is_selected else ""
        ax.set_title(f"{name}{title_suffix}\nAUC: {data['auc']:.3f}", fontsize=11, fontweight='bold')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=9)
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=9)
        ax.legend(loc='lower right', fontsize=8.5)
        ax.grid(True, linestyle=':', alpha=0.6)

        if is_selected:
            ax.set_facecolor('#fff8f8')

    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("💡 ROC curve data will appear here once models are trained via `python3 main.py`.")

st.caption("UE24CS352A Machine Learning Mini-Project | Performance Metrics: Sensitivity, Specificity, AUC, ROC")
