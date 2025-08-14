import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os

# --- File paths ---
REF_STONE_FILE = "reference_stones.csv"
CALIB_FILE = "calibration.csv"
TEST_STONE_FILE = "test_stones.csv"

# --- Initialize session state ---
if 'df_reference_stones' not in st.session_state:
    if os.path.exists(REF_STONE_FILE):
        st.session_state.df_reference_stones = pd.read_csv(REF_STONE_FILE)
    else:
        st.session_state.df_reference_stones = pd.DataFrame(columns=['Label','L','a','b'])

if 'df_test_stones' not in st.session_state:
    if os.path.exists(TEST_STONE_FILE):
        st.session_state.df_test_stones = pd.read_csv(TEST_STONE_FILE)
    else:
        st.session_state.df_test_stones = pd.DataFrame(columns=['TestLabel','L','a','b'])

if 'calibration_offset' not in st.session_state:
    if os.path.exists(CALIB_FILE):
        calib_df = pd.read_csv(CALIB_FILE)
        st.session_state.calibration_offset = {
            'L': float(calib_df.loc[0,'L']),
            'a': float(calib_df.loc[0,'a']),
            'b': float(calib_df.loc[0,'b'])
        }
    else:
        st.session_state.calibration_offset = {'L':0, 'a':0, 'b':0}

# --- Helper functions ---
def compute_lab_mean(image):
    lab_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2LAB)
    L = lab_image[:,:,0].mean() + st.session_state.calibration_offset['L']
    a = lab_image[:,:,1].mean() + st.session_state.calibration_offset['a']
    b = lab_image[:,:,2].mean() + st.session_state.calibration_offset['b']
    return L, a, b

def add_reference_stone(label, image):
    L, a, b = compute_lab_mean(image)
    st.session_state.df_reference_stones = pd.concat([
        st.session_state.df_reference_stones,
        pd.DataFrame({'Label':[label],'L':[L],'a':[a],'b':[b]})
    ], ignore_index=True)
    st.session_state.df_reference_stones.to_csv(REF_STONE_FILE, index=False)
    st.success(f"Reference stone '{label}' added!")

def add_test_stone(label, image):
    L, a, b = compute_lab_mean(image)
    st.session_state.df_test_stones = pd.concat([
        st.session_state.df_test_stones,
        pd.DataFrame({'TestLabel':[label],'L':[L],'a':[a],'b':[b]})
    ], ignore_index=True)
    st.session_state.df_test_stones.to_csv(TEST_STONE_FILE, index=False)
    st.success(f"Test stone '{label}' added!")

def grade_stone(L_test, a_test, b_test):
    if st.session_state.df_reference_stones.empty:
        return None, None
    df = st.session_state.df_reference_stones.copy()
    df['ΔE'] = np.sqrt((df['L']-L_test)**2 + (df['a']-a_test)**2 + (df['b']-b_test)**2)
    best_match = df.loc[df['ΔE'].idxmin()]
    return best_match['Label'], best_match['ΔE']

# --- Streamlit UI ---
st.title("💎 Stone Grading App")

# --- Tabs ---
tab1, tab2 = st.tabs(["Setup / Calibration", "Production / Grading"])

# --- Tab 1: Setup / Calibration ---
with tab1:
    st.header("🛠 Setup and Calibration")

    # Calibration
    st.subheader("Calibration")
    calib_image_file = st.file_uploader("Upload Calibration Stone Image", type=['png','jpg','jpeg'], key='calib')
    true_L = st.number_input("True L value", min_value=0.0, max_value=100.0, step=0.1)
    true_a = st.number_input("True a value", min_value=-128.0, max_value=127.0, step=0.1)
    true_b = st.number_input("True b value", min_value=-128.0, max_value=127.0, step=0.1)
    if st.button("Calibrate"):
        if calib_image_file is not None:
            image = Image.open(calib_image_file)
            lab_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2LAB)
            st.session_state.calibration_offset['L'] = true_L - lab_image[:,:,0].mean()
            st.session_state.calibration_offset['a'] = true_a - lab_image[:,:,1].mean()
            st.session_state.calibration_offset['b'] = true_b - lab_image[:,:,2].mean()
            pd.DataFrame([st.session_state.calibration_offset]).to_csv(CALIB_FILE, index=False)
            st.success("Calibration applied!")
        else:
            st.warning("Upload a calibration stone image first!")

    # Add Reference Stone
    st.subheader("Add Reference Stone")
    ref_label = st.text_input("Reference Stone Label", key='ref_label')
    ref_image_file = st.file_uploader("Upload Reference Stone Image", type=['png','jpg','jpeg'], key='ref_image')
    if st.button("Add Reference Stone"):
        if ref_image_file and ref_label:
            add_reference_stone(ref_label, Image.open(ref_image_file))
        else:
            st.warning("Provide both label and image.")
    st.dataframe(st.session_state.df_reference_stones)

    # Add Test Stone
    st.subheader("Add Test Stone")
    test_label = st.text_input("Test Stone Label", key='test_label')
    test_image_file = st.file_uploader("Upload Test Stone Image", type=['png','jpg','jpeg'], key='test_image')
    if st.button("Add Test Stone"):
        if test_image_file and test_label:
            add_test_stone(test_label, Image.open(test_image_file))
        else:
            st.warning("Provide both label and image.")
    st.dataframe(st.session_state.df_test_stones)

# --- Tab 2: Production / Grading ---
with tab2:
    st.header("💻 Production / Stone Grading")

    # Tolerance
    st.sidebar.header("Tolerance Setting")
    tolerance_deltaE = st.sidebar.number_input("Set ΔE tolerance", min_value=0.0, max_value=100.0, value=2.0, step=0.1)

    # Live Grading
    st.subheader("Live Grading via Webcam")
    img_file_buffer = st.camera_input("Capture Stone Image", key='live')
    if img_file_buffer:
        captured_image = Image.open(img_file_buffer)
        st.image(captured_image, caption="Captured Image", use_column_width=True)
        L, a, b = compute_lab_mean(captured_image)
        label, delta_e = grade_stone(L, a, b)
        if label:
            if delta_e <= tolerance_deltaE:
                st.success(f"Closest Reference: {label} | ΔE: {delta_e:.2f} ✅ Within Tolerance")
            else:
                st.error(f"Closest Reference: {label} | ΔE: {delta_e:.2f} ❌ Out of Tolerance")
        else:
            st.warning("No reference stones available for grading.")

    # Batch Grading for Uploaded Test Stones
    st.subheader("Batch Grading Uploaded Test Stones")
    if not st.session_state.df_test_stones.empty:
        results = []
        for idx, row in st.session_state.df_test_stones.iterrows():
            label, delta_e = grade_stone(row['L'], row['a'], row['b'])
            status = "✅ Within Tolerance" if delta_e <= tolerance_deltaE else "❌ Out of Tolerance"
            results.append({'Test Stone': row['TestLabel'], 'Closest Reference': label, 'ΔE': delta_e, 'Status': status})
        st.dataframe(pd.DataFrame(results))
    else:
        st.info("No test stones uploaded for batch grading.")
