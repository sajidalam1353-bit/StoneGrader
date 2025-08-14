import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
from skimage import color

# ----------------------------
# Helper Functions
# ----------------------------
def rgb_to_lab(image):
    """Convert RGB PIL image to LAB values (mean of pixels)."""
    rgb = np.array(image.convert('RGB')) / 255.0
    lab = color.rgb2lab(rgb)
    L = np.mean(lab[:,:,0])
    a = np.mean(lab[:,:,1])
    b = np.mean(lab[:,:,2])
    return L, a, b

def grade_stone(test_lab, reference_df, tolerance):
    """Compare test stone with all reference stones and return closest match."""
    min_delta = None
    best_label = None
    best_row = None
    test_color = LabColor(*test_lab)
    
    for idx, row in reference_df.iterrows():
        ref_color = LabColor(row['L'], row['a'], row['b'])
        delta = delta_e_cie2000(test_color, ref_color).item()
        if (min_delta is None) or (delta < min_delta):
            min_delta = delta
            best_label = row['Label']
            best_row = row
    
    status = "Within Tolerance ✅" if min_delta <= tolerance else "Above Tolerance ❌"
    color_code = "green" if min_delta <= tolerance else "red"
    return best_label, min_delta, status, color_code, best_row

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Stone Grader", layout="wide")
st.title("💎 Stone Grading System")

# Initialize session state
if 'reference_stones' not in st.session_state:
    st.session_state.reference_stones = pd.DataFrame(columns=['Label','L','a','b','Image'])

if 'tolerance' not in st.session_state:
    st.session_state.tolerance = 2.0  # default ΔE tolerance

# Tabs
tab1, tab2 = st.tabs(["🛠 Calibration & Reference Stones", "🎯 Grading Stones"])

# ----------------------------
# Tab 1: Calibration & Reference Stones
# ----------------------------
with tab1:
    st.header("Add Reference Stone")
    ref_label = st.text_input("Enter Stone Label/Name")
    ref_image_file = st.file_uploader("Upload Reference Stone Image", type=['jpg','jpeg','png'])
    
    st.subheader("Tolerance Setting")
    st.session_state.tolerance = st.slider("Set ΔE Tolerance", min_value=0.0, max_value=10.0, value=st.session_state.tolerance, step=0.1)
    
    if st.button("Add Reference Stone"):
        if ref_label and ref_image_file:
            image = Image.open(ref_image_file)
            L,a,b = rgb_to_lab(image)
            # Save image bytes for later display
            img_bytes = np.array(image.convert('RGB'))
            st.session_state.reference_stones = st.session_state.reference_stones.append(
                {'Label': ref_label, 'L': L, 'a': a, 'b': b, 'Image': img_bytes},
                ignore_index=True
            )
            st.success(f"Reference Stone '{ref_label}' added successfully!")
        else:
            st.warning("Please provide both label and image.")

    st.subheader("Current Reference Stones")
    st.dataframe(st.session_state.reference_stones[['Label','L','a','b']])

# ----------------------------
# Tab 2: Grading Stones (Production)
# ----------------------------
with tab2:
    st.header("Grading Test Stone")

    st.subheader("Upload Test Stone Image")
    test_image_file = st.file_uploader("Upload Test Stone", type=['jpg','jpeg','png'], key="test_upload")
    
    st.subheader("Or Capture Test Stone via Camera")
    captured_image = st.camera_input("Take a picture of the stone")

    test_image = None
    if test_image_file:
        test_image = Image.open(test_image_file)
    elif captured_image:
        test_image = Image.open(captured_image)

    if test_image:
        st.image(test_image, caption="Test Stone", use_container_width=True)
        L_test,a_test,b_test = rgb_to_lab(test_image)

        if not st.session_state.reference_stones.empty:
            label, delta, status, color_code, best_row = grade_stone(
                (L_test,a_test,b_test), st.session_state.reference_stones, st.session_state.tolerance
            )

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Test Stone")
                st.image(test_image, use_container_width=True)
            with col2:
                st.subheader(f"Matched Reference Stone: {label}")
                st.image(best_row['Image'], use_container_width=True)

            st.markdown(f"<h3 style='color:{color_code};'>Status: {status}</h3>", unsafe_allow_html=True)
            st.metric(label="ΔE (CIE2000)", value=f"{delta:.2f}", delta=f"{st.session_state.tolerance} tolerance")

        else:
            st.warning("Add at least one reference stone in Tab 1 to grade.")

