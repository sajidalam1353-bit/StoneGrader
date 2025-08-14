import streamlit as st
import pandas as pd
import numpy as np
from skimage import io, color
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
from PIL import Image
import io as sysio

# --------------------------------
# INITIAL SESSION STATE
# --------------------------------
if "reference_stones" not in st.session_state:
    st.session_state.reference_stones = pd.DataFrame(columns=["Label", "L", "a", "b", "Image"])
if "tolerance" not in st.session_state:
    st.session_state.tolerance = 2.0

# --------------------------------
# FUNCTIONS
# --------------------------------
def image_to_lab(image):
    img_array = np.array(image)
    lab = color.rgb2lab(img_array / 255.0)
    L = np.mean(lab[:, :, 0])
    a = np.mean(lab[:, :, 1])
    b = np.mean(lab[:, :, 2])
    return L, a, b

def add_reference_stone(label, image):
    L, a, b = image_to_lab(image)
    img_bytes = sysio.BytesIO()
    image.save(img_bytes, format="PNG")
    new_row = pd.DataFrame([{
        "Label": label,
        "L": L,
        "a": a,
        "b": b,
        "Image": img_bytes.getvalue()
    }])
    st.session_state.reference_stones = pd.concat(
        [st.session_state.reference_stones, new_row], ignore_index=True
    )

def grade_stone(L_test, a_test, b_test):
    if st.session_state.reference_stones.empty:
        return None, None
    test_color = LabColor(L_test, a_test, b_test)
    min_delta = float("inf")
    best_label = None
    for _, row in st.session_state.reference_stones.iterrows():
        ref_color = LabColor(row["L"], row["a"], row["b"])
        delta_val = delta_e_cie2000(test_color, ref_color)
        if hasattr(delta_val, "item"):
            delta = float(delta_val.item())
        else:
            delta = float(delta_val)
        if delta < min_delta:
            min_delta = delta
            best_label = row["Label"]
    return best_label, min_delta

# --------------------------------
# STREAMLIT UI
# --------------------------------
st.set_page_config(page_title="Stone Grading System", layout="wide")
st.title("💎 Stone Grading Device")

tab1, tab2 = st.tabs(["⚙️ Calibration & Reference Stones", "🔍 Grading Stones (Production)"])

# --------------------------------
# TAB 1 - CALIBRATION
# --------------------------------
with tab1:
    st.subheader("Add Reference Stone")
    ref_label = st.text_input("Enter Stone Label")
    col1, col2 = st.columns(2)
    with col1:
        ref_file = st.file_uploader("Upload Reference Stone Image", type=["jpg", "jpeg", "png"])
    with col2:
        ref_cam = st.camera_input("Capture Reference Stone with Camera")

    if st.button("Add Reference Stone"):
        img = None
        if ref_file:
            img = Image.open(ref_file).convert("RGB")
        elif ref_cam:
            img = Image.open(ref_cam).convert("RGB")
        if img and ref_label:
            add_reference_stone(ref_label, img)
            st.success(f"Reference stone '{ref_label}' added successfully!")
        else:
            st.warning("Please provide both label and image.")

    st.slider("Tolerance (ΔE) for Grading", min_value=0.5, max_value=10.0, value=st.session_state.tolerance,
              step=0.1, key="tolerance")

    if not st.session_state.reference_stones.empty:
        st.subheader("Reference Stones")
        for idx, row in st.session_state.reference_stones.iterrows():
            st.image(row["Image"], caption=f"{row['Label']} (L={row['L']:.2f}, a={row['a']:.2f}, b={row['b']:.2f})",
                     width=150)

# --------------------------------
# TAB 2 - GRADING
# --------------------------------
with tab2:
    st.subheader("Grading Test Stones")
    test_file = st.file_uploader("Upload Test Stone Image", type=["jpg", "jpeg", "png"])
    test_cam = st.camera_input("Capture Test Stone with Camera")

    if st.button("Grade Stone"):
        img = None
        if test_file:
            img = Image.open(test_file).convert("RGB")
        elif test_cam:
            img = Image.open(test_cam).convert("RGB")

        if img:
            L_test, a_test, b_test = image_to_lab(img)
            label, delta = grade_stone(L_test, a_test, b_test)
            if label is None:
                st.error("No reference stones available for grading.")
            else:
                if delta <= st.session_state.tolerance:
                    st.success(f"Matched with '{label}' ✅\nΔE = {delta:.2f}")
                else:
                    st.warning(f"No close match found ❌\nClosest: {label} (ΔE = {delta:.2f})")
        else:
            st.warning("Please upload or capture a test stone image.")

