# Save as stone_grader_3d.py
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
from PIL import Image
import plotly.express as px

REF_FILE = "reference_stones.csv"

# Automatic ROI detection
def detect_stone_roi(image):
    image_np = np.array(image.convert('RGB'))
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return x, y, w, h

# Convert image (PIL) to Lab average within ROI
def get_lab_from_image(image, roi=None):
    image_np = np.array(image.convert('RGB'))
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    if roi:
        x, y, w, h = roi
        image_bgr = image_bgr[y:y+h, x:x+w]
    lab_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab_image)
    return np.mean(L), np.mean(a), np.mean(b)

# Add reference stone
def add_reference_stone(label, image):
    roi = detect_stone_roi(image)
    if roi is None:
        st.error("No stone detected. Try again.")
        return
    L, a, b = get_lab_from_image(image, roi)
    df = pd.read_csv(REF_FILE) if pd.io.common.file_exists(REF_FILE) else pd.DataFrame(columns=['Label','L','a','b'])
    df = df.append({'Label': label, 'L': L, 'a': a, 'b': b}, ignore_index=True)
    df.to_csv(REF_FILE, index=False)
    st.success(f"Reference '{label}' added successfully.")

# Grade test stone
def grade_test_stone(image):
    roi = detect_stone_roi(image)
    if roi is None:
        st.error("No stone detected. Try again.")
        return None, None
    L_test, a_test, b_test = get_lab_from_image(image, roi)
    test_color = LabColor(L_test, a_test, b_test)
    
    if not pd.io.common.file_exists(REF_FILE):
        st.error("No reference stones found. Add references first.")
        return None, None
    
    df = pd.read_csv(REF_FILE)
    min_de = float('inf')
    best_match = None
    
    for _, row in df.iterrows():
        ref_color = LabColor(row['L'], row['a'], row['b'])
        de = delta_e_cie2000(test_color, ref_color)
        if de < min_de:
            min_de = de
            best_match = row['Label']
    
    st.success(f"Best match: {best_match} (ΔE00 = {min_de:.2f})")
    
    # Add test stone to dataframe for plotting
    df_plot = df.copy()
    df_plot = df_plot.append({'Label':'Test Stone','L':L_test,'a':a_test,'b':b_test}, ignore_index=True)
    
    fig = px.scatter_3d(df_plot, x='a', y='b', z='L', color='Label', text='Label', size=[10]*len(df_plot))
    fig.update_traces(marker=dict(size=8), selector=dict(mode='markers+text'))
    st.plotly_chart(fig, use_container_width=True)
    
    return best_match, min_de

# Streamlit UI
st.title("💎 Stone Grading with 3D Color Visualization")

tab1, tab2 = st.tabs(["Add Reference Stone", "Grade Test Stone"])

with tab1:
    st.header("Add Reference Stone")
    ref_label = st.text_input("Reference Stone Label")
    ref_image = st.camera_input("Capture Reference Stone")
    
    if st.button("Add Reference") and ref_label and ref_image:
        image = Image.open(ref_image)
        add_reference_stone(ref_label, image)

with tab2:
    st.header("Grade Test Stone")
    test_image = st.camera_input("Capture Test Stone", key="test")
    
    if st.button("Grade Stone") and test_image:
        image = Image.open(test_image)
        grade_test_stone(image)