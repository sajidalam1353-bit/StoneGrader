import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from skimage import color
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000

# ---------- Initialize Session State ----------
if "reference_stones" not in st.session_state:
    st.session_state.reference_stones = pd.DataFrame(columns=["Label", "L", "a", "b", "Image"])

if "tolerance" not in st.session_state:
    st.session_state.tolerance = 2.0

# ---------- Helper Functions ----------
def image_to_lab(image):
    image = image.convert("RGB")
    img_array = np.array(image) / 255.0
    lab_image = color.rgb2lab(img_array)
    L = np.mean(lab_image[:, :, 0])
    a = np.mean(lab_image[:, :, 1])
    b = np.mean(lab_image[:, :, 2])
    return L, a, b

def grade_stone(L_test, a_test, b_test):
    if st.session_state.reference_stones.empty:
        return None, None

    test_color = LabColor(L_test, a_test, b_test)
    min_delta = float("inf")
    best_label = None

    for _, row in st.session_state.reference_stones.iterrows():
        ref_color = LabColor(row["L"], row["a"], row["b"])
        delta = float(delta_e_cie2000(test_color, ref_color))
        if delta < min_delta:
            min_delta = delta
            best_label = row["Label"]

    return best_label, min_delta

# ---------- UI Layout ----------
st.set_page_config(page_title="Stone Grading Device", layout="wide")
tab1, tab2 = st.tabs(["⚙️ Calibration & Reference Stones", "🔍 Grading Stones (Production)"])

# ---------- TAB 1 ----------
with tab1:
    st.header("Add & Calibrate Reference Stones")

    st.session_state.tolerance = st.slider(
        "Set Grading Tolerance (ΔE CIE2000)",
        min_value=0.1,
        max_value=10.0,
        value=st.session_state.tolerance,
        step=0.1
    )

    ref_label = st.text_input("Enter Reference Stone Label")
    ref_upload = st.file_uploader("Upload Reference Stone Image", type=["jpg", "jpeg", "png"])
    ref_camera = st.camera_input("Capture Reference Stone with Camera")

    if st.button("Add Reference Stone"):
        img_source = ref_upload or ref_camera
        if img_source and ref_label.strip():
            image = Image.open(img_source)
            L, a, b = image_to_lab(image)

            img_bytes = BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes = img_bytes.getvalue()

            new_row = pd.DataFrame([{
                "Label": ref_label,
                "L": L,
                "a": a,
                "b": b,
                "Image": img_bytes
            }])
            st.session_state.reference_stones = pd.concat(
                [st.session_state.reference_stones, new_row],
                ignore_index=True
            )

            st.success(f"Reference stone '{ref_label}' added successfully!")
        else:
            st.error("Please provide both label and image.")

    if not st.session_state.reference_stones.empty:
        st.subheader("Stored Reference Stones")
        for _, row in st.session_state.reference_stones.iterrows():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(row["Image"], caption=row["Label"], use_container_width=True)
            with col2:
                st.write(f"**Label:** {row['Label']}")
                st.write(f"L: {row['L']:.2f}, a: {row['a']:.2f}, b: {row['b']:.2f}")

# ---------- TAB 2 ----------
with tab2:
    st.header("Grading Test Stones")

    test_upload = st.file_uploader("Upload Test Stone Image", type=["jpg", "jpeg", "png"])
    test_camera = st.camera_input("Capture Test Stone with Camera")

    if st.button("Grade Test Stone"):
        img_source = test_upload or test_camera
        if img_source:
            test_image = Image.open(img_source)
            L_test, a_test, b_test = image_to_lab(test_image)

            label, delta = grade_stone(L_test, a_test, b_test)
            if label:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(test_image, caption="Test Stone", use_container_width=True)
                with col2:
                    ref_row = st.session_state.reference_stones[
                        st.session_state.reference_stones["Label"] == label
                    ].iloc[0]
                    st.image(ref_row["Image"], caption=f"Reference: {label}", use_container_width=True)

                st.write(f"**Closest Match:** {label}")
                st.write(f"**ΔE (CIE2000):** {delta:.2f}")

                if delta <= st.session_state.tolerance:
                    st.success("✅ Within tolerance limit")
                else:
                    st.error("❌ Out of tolerance limit")
            else:
                st.warning("No reference stones available for grading.")
        else:
            st.error("Please upload or capture a test stone image.")
