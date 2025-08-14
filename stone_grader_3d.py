# stone_grader_app.py
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
from skimage import io, color

# ----------------------------
# Initialization
# ----------------------------
if 'ref_stones' not in st.session_state:
    st.session_state.ref_stones = pd.DataFrame(columns=['Label','L','a','b'])
if 'calibration_offset' not in st.session_state:
    st.session_state.calibration_offset = {'L':0,'a':0,'b':0}

# ----------------------------
# Helper Functions
# ----------------------------
def compute_lab_mean(image):
    img = np.array(image.convert('RGB')) / 255.0
    lab_img = color.rgb2lab(img)
    L_mean = np.mean(lab_img[:,:,0])
    a_mean = np.mean(lab_img[:,:,1])
    b_mean = np.mean(lab_img[:,:,2])
    return L_mean, a_mean, b_mean

def apply_calibration(L,a,b):
    L_adj = L + st.session_state.calibration_offset['L']
    a_adj = a + st.session_state.calibration_offset['a']
    b_adj = b + st.session_state.calibration_offset['b']
    return L_adj, a_adj, b_adj

def add_reference_stone(label, image):
    L,a,b = compute_lab_mean(image)
    L,a,b = apply_calibration(L,a,b)
    st.session_state.ref_stones = pd.concat([st.session_state.ref_stones,
                                             pd.DataFrame([{'Label':label,'L':L,'a':a,'b':b}])],
                                            ignore_index=True)

def grade_stone(L,a,b):
    L,a,b = apply_calibration(L,a,b)
    if st.session_state.ref_stones.empty:
        return None, None
    test_color = LabColor(L,a,b)
    min_delta = None
    closest_label = None
    for _, row in st.session_state.ref_stones.iterrows():
        ref_color = LabColor(row['L'], row['a'], row['b'])
        delta = delta_e_cie2000(test_color, ref_color)
        if (min_delta is None) or (delta < min_delta):
            min_delta = delta
            closest_label = row['Label']
    return closest_label, min_delta

# ----------------------------
# Streamlit App Tabs
# ----------------------------
tab1, tab2 = st.tabs(["Calibration & Reference Stones", "Grading Test Stones"])

# ----------------------------
# Tab 1: Calibration & Reference Stones
# ----------------------------
with tab1:
    st.header("Calibration & Reference Stones")

    # Calibration Image Upload
    calib_file = st.file_uploader("Upload Calibration Image (White/Gray)", type=['png','jpg','jpeg'], key='calib_upload')
    if calib_file:
        calib_img = Image.open(calib_file)
        st.image(calib_img, caption="Calibration Image", use_column_width=True)
        L_calib, a_calib, b_calib = compute_lab_mean(calib_img)
        L_ideal, a_ideal, b_ideal = 100,0,0
        st.session_state.calibration_offset = {
            'L': L_ideal - L_calib,
            'a': a_ideal - a_calib,
            'b': b_ideal - b_calib
        }
        st.success(f"Calibration applied: ΔL={st.session_state.calibration_offset['L']:.2f}, Δa={st.session_state.calibration_offset['a']:.2f}, Δb={st.session_state.calibration_offset['b']:.2f}")

    # Add Reference Stone
    st.subheader("Add Reference Stone")
    ref_label = st.text_input("Stone Label", key='ref_label')
    ref_file = st.file_uploader("Upload Reference Stone Image", type=['png','jpg','jpeg'], key='ref_file')
    if st.button("Add Reference Stone"):
        if ref_label and ref_file:
            img = Image.open(ref_file)
            add_reference_stone(ref_label, img)
            st.success(f"Reference stone '{ref_label}' added.")
        else:
            st.error("Please provide label and image.")

    # Show Reference Stones Table
    if not st.session_state.ref_stones.empty:
        st.dataframe(st.session_state.ref_stones)

    # Upload Test Stone for Preview
    st.subheader("Upload Test Stone (Optional)")
    test_file = st.file_uploader("Upload Test Stone Image", type=['png','jpg','jpeg'], key='test_upload_tab1')
    if test_file:
        test_img = Image.open(test_file)
        st.image(test_img, caption="Test Stone", use_column_width=True)
        L_test, a_test, b_test = compute_lab_mean(test_img)
        label, delta = grade_stone(L_test,a_test,b_test)
        if label:
            st.info(f"Closest Reference Stone: {label} (ΔE={delta:.2f})")
        else:
            st.warning("No reference stones added yet.")

# ----------------------------
# Tab 2: Grading Test Stones
# ----------------------------
with tab2:
    st.header("Grading Test Stones")

    # Live Camera Input
    st.subheader("Capture Image with Camera")
    cam_file = st.camera_input("Take a picture of the stone")
    if cam_file:
        cam_img = Image.open(cam_file)
        st.image(cam_img, caption="Captured Stone", use_column_width=True)
        L_cam,a_cam,b_cam = compute_lab_mean(cam_img)
        label, delta = grade_stone(L_cam,a_cam,b_cam)
        if label:
            st.info(f"Closest Reference Stone: {label} (ΔE={delta:.2f})")
        else:
            st.warning("No reference stones added yet.")

    # Upload Test Stone
    st.subheader("Upload Test Stone Image")
    test_file2 = st.file_uploader("Upload Test Stone", type=['png','jpg','jpeg'], key='test_upload_tab2')
    if test_file2:
        test_img2 = Image.open(test_file2)
        st.image(test_img2, caption="Test Stone", use_column_width=True)
        L_test2,a_test2,b_test2 = compute_lab_mean(test_img2)
        label, delta = grade_stone(L_test2,a_test2,b_test2)
        if label:
            st.info(f"Closest Reference Stone: {label} (ΔE={delta:.2f})")
        else:
            st.warning("No reference stones added yet.")
