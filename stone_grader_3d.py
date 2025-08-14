import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Color Stone Grader", layout="wide")
st.title("💎 Color Stone Grading App")

# --- Global DataFrame for reference stones ---
try:
    df_reference_stones
except NameError:
    df_reference_stones = pd.DataFrame(columns=['Label', 'L', 'a', 'b'])

# --- Helper function to convert uploaded image to OpenCV format ---
def load_image(uploaded_file):
    image = Image.open(uploaded_file)
    image = np.array(image)
    if image.shape[-1] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif image.shape[-1] == 3:  # RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

# --- Add Reference Stone ---
def add_reference_stone(label, image):
    global df_reference_stones
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L = lab_image[:, :, 0].mean()
    a = lab_image[:, :, 1].mean()
    b = lab_image[:, :, 2].mean()
    new_row = pd.DataFrame([{'Label': label, 'L': L, 'a': a, 'b': b}])
    df_reference_stones = pd.concat([df_reference_stones, new_row], ignore_index=True)
    st.success(f"Reference stone '{label}' added successfully!")

# --- Grade Test Stone ---
def grade_test_stone(image):
    global df_reference_stones
    if df_reference_stones.empty:
        st.error("No reference stones added yet!")
        return None
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L_test = lab_image[:, :, 0].mean()
    a_test = lab_image[:, :, 1].mean()
    b_test = lab_image[:, :, 2].mean()
    
    distances = []
    for idx, row in df_reference_stones.iterrows():
        L_ref, a_ref, b_ref = row['L'], row['a'], row['b']
        dist = np.sqrt((L_test - L_ref)**2 + (a_test - a_ref)**2 + (b_test - b_ref)**2)
        distances.append(dist)
    
    min_idx = np.argmin(distances)
    closest_label = df_reference_stones.iloc[min_idx]['Label']
    closest_distance = distances[min_idx]
    
    st.success(f"Test stone closest to reference: '{closest_label}' (Distance: {closest_distance:.2f})")
    return {'Label': closest_label, 'Distance': closest_distance}

# --- Sidebar: Upload or Capture Reference Stone ---
st.sidebar.header("Add Reference Stone")
ref_label = st.sidebar.text_input("Stone Label")
ref_file = st.sidebar.file_uploader("Upload Reference Stone Image", type=['png','jpg','jpeg'], key="ref_upload")
ref_camera = st.sidebar.camera_input("Or Capture Reference Stone", key="ref_camera")
if st.sidebar.button("Add Reference Stone"):
    if ref_label and (ref_file or ref_camera):
        img = load_image(ref_file) if ref_file else load_image(ref_camera)
        add_reference_stone(ref_label, img)
    else:
        st.sidebar.error("Please enter a label and provide an image or capture.")

# --- Sidebar: Upload or Capture Test Stone ---
st.sidebar.header("Grade Test Stone")
test_file = st.sidebar.file_uploader("Upload Test Stone Image", type=['png','jpg','jpeg'], key="test_upload")
test_camera = st.sidebar.camera_input("Or Capture Test Stone", key="test_camera")
if st.sidebar.button("Grade Test Stone"):
    if test_file or test_camera:
        img_test = load_image(test_file) if test_file else load_image(test_camera)
        grade_test_stone(img_test)
    else:
        st.sidebar.error("Please provide an image or capture a test stone.")

# --- Display Reference Stones Table ---
if not df_reference_stones.empty:
    st.subheader("Reference Stones Database")
    st.dataframe(df_reference_stones)