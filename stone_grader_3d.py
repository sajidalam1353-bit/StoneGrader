import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os

# --- File paths ---
REF_STONE_FILE = "reference_stones.csv"
CALIB_FILE = "calibration.csv"

# --- Initialize session state ---
if 'df_reference_stones' not in st.session_state:
    if os.path.exists(REF_STONE_FILE):
        st.session_state.df_reference_stones = pd.read_csv(REF_STONE_FILE)
    else:
        st.session_state.df_reference_stones = pd.DataFrame(columns=['Label','L','a','b'])

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
def add_reference_stone(label, image):
    lab_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2LAB)
    L = lab_image[:,:,0].mean() + st.session_state.calibration_offset['L']
    a = lab_image[:,:,1].mean() + st.session_state.calibration_offset['a']
    b = lab_image[:,:,2].mean() + st.session_state.calibration_offset['b']
    
    st.session_state.df_reference_stones = pd.concat([
        st.session_state.df_reference_stones,
        pd.DataFrame({'Label':[label],'L':[L],'a':[a],'b':[b]})
    ], ignore_index=True)
    
    st.session_state.df_reference_stones.to_csv(REF_STONE_FILE, index=False)
    st.success(f"Reference stone '{label}' added!")

def grade_stone_live(image):
    if st.session_state.df_reference_stones.empty:
        return None, None
    lab_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2LAB)
    L_test = lab_image[:,:,0].mean() + st.session_state.calibration_offset['L']
    a_test = lab_image[:,:,1].mean() + st.session_state.calibration_offset['a']
    b_test = lab_image[:,:,2].mean() + st.session_state.calibration_offset['b']

    df = st.session_state.df_reference_stones.copy()
    df['ΔE'] = np.sqrt((df['L']-L_test)**2 + (df['a']-a_test)**2 + (df['b']-b_test)**2)
    best_match = df.loc[df['ΔE'].idxmin()]
    return best_match['Label'], best_match['ΔE']

# --- Streamlit UI ---
st.title("💎 Full Stone Grading App")

# --- Sidebar: Calibration ---
st.sidebar.header("Calibration")
calib_image_file = st.sidebar.file_uploader("Upload Calibration Stone Image", type=['png','jpg','jpeg'])
true_L = st.sidebar.number_input("True L value", min_value=0.0, max_value=100.0, step=0.1)
true_a = st.sidebar.number_input("True a value", min_value=-128.0, max_value=127.0, step=0.1)
true_b = st.sidebar.number_input("True b value", min_value=-128.0, max_value=127.0, step=0.1)

if st.sidebar.button("Calibrate"):
    if calib_image_file is not None:
        image = np.array(Image.open(calib_image_file))
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        st.session_state.calibration_offset['L'] = true_L - lab_image[:,:,0].mean()
        st.session_state.calibration_offset['a'] = true_a - lab_image[:,:,1].mean()
        st.session_state.calibration_offset['b'] = true_b - lab_image[:,:,2].mean()
        pd.DataFrame([st.session_state.calibration_offset]).to_csv(CALIB_FILE, index=False)
        st.success("Calibration applied!")
    else:
        st.warning("Upload a calibration stone image first!")

# --- Sidebar: Tolerance ---
st.sidebar.header("Tolerance Setting")
tolerance_deltaE = st.sidebar.number_input("Set ΔE tolerance", min_value=0.0, max_value=100.0, value=2.0, step=0.1)

# --- Reference Stone Management ---
st.header("Reference Stones")
ref_label = st.text_input("Reference Stone Label")
ref_image_file = st.file_uploader("Upload Reference Stone Image", type=['png','jpg','jpeg'], key='ref_upload')

if st.button("Add Reference Stone"):
    if ref_image_file and ref_label:
        add_reference_stone(ref_label, ref_image_file)
    else:
        st.warning("Provide both label and image.")

st.dataframe(st.session_state.df_reference_stones)

# --- Live Grading Section ---
st.header("💻 Live Grading via Webcam")
st.write("Point your webcam at a stone, the app will show closest reference and ΔE with tolerance check.")

img_file_buffer = st.camera_input("Capture Stone Image")
if img_file_buffer:
    captured_image = Image.open(img_file_buffer)
    st.image(captured_image, caption="Captured Image", use_column_width=True)
    
    label, delta_e = grade_stone_live(captured_image)
    if label:
        if delta_e <= tolerance_deltaE:
            st.success(f"Closest Reference: {label} | ΔE: {delta_e:.2f} ✅ Within Tolerance")
        else:
            st.error(f"Closest Reference: {label} | ΔE: {delta_e:.2f} ❌ Out of Tolerance")
    else:
        st.warning("No reference stones available for grading.")