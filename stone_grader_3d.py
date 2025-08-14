import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Color Stone Grader", layout="wide")
st.title("💎 Color Stone Grading App with Persistent Reference Stones")

# --- File to store reference stones ---
DATA_FILE = "reference_stones.csv"

# --- Load existing reference stones if file exists ---
if os.path.exists(DATA_FILE):
    df_reference_stones = pd.read_csv(DATA_FILE)
else:
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
    df_reference_stones.to_csv(DATA_FILE, index=False)  # Save permanently
    st.success(f"Reference stone '{label}' added successfully!")

# --- Grade Test Stone with Tolerance ---
def grade_test_stone(image, tolerance):
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
    
    if closest_distance <= tolerance:
        st.success(f"✅ Test stone matches reference: '{closest_label}' (Distance: {closest_distance:.2f})")
    else:
        st.warning(f"⚠️ No reference stone within tolerance. Closest: '{closest_label}' (Distance: {closest_distance:.2f})")
    
    return {'Label': closest_label, 'Distance': closest_distance, 'L': L_test, 'a': a_test, 'b': b_test}

# --- Plot LAB Space ---
def plot_lab_space(test_stone=None):
    fig = go.Figure()

    # Reference stones
    if not df_reference_stones.empty:
        fig.add_trace(go.Scatter3d(
            x=df_reference_stones['L'],
            y=df_reference_stones['a'],
            z=df_reference_stones['b'],
            mode='markers+text',
            marker=dict(size=8, color='blue'),
            text=df_reference_stones['Label'],
            name='Reference Stones'
        ))

    # Test stone
    if test_stone is not None:
        fig.add_trace(go.Scatter3d(
            x=[test_stone['L']],
            y=[test_stone['a']],
            z=[test_stone['b']],
            mode='markers+text',
            marker=dict(size=10, color='red'),
            text=['Test Stone'],
            name='Test Stone'
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='L',
            yaxis_title='a',
            zaxis_title='b'
        ),
        width=700,
        height=500,
        margin=dict(l=0, r=0, b=0, t=0)
    )

    st.plotly_chart(fig)

# --- Sidebar: Tolerance Slider ---
st.sidebar.header("Settings")
tolerance = st.sidebar.slider("Set LAB Distance Tolerance", min_value=1, max_value=50, value=10, step=1)

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
        result = grade_test_stone(img_test, tolerance)
        if result is not None:
            plot_lab_space(result)  # Show 3D LAB plot
    else:
        st.sidebar.error("Please provide an image or capture a test stone.")

# --- Display Reference Stones Table ---
if not df_reference_stones.empty:
    st.subheader("Reference Stones Database")
    st.dataframe(df_reference_stones)