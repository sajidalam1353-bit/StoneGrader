import os
import json
import numpy as np

# --- PATCH for numpy.asscalar removal (NumPy >= 1.25) ---
if not hasattr(np, "asscalar"):
    np.asscalar = lambda a: a.item()

import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
from skimage import color
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000

# ------------------ Files for persistence ------------------
REFS_CSV = "reference_stones.csv"   # Label, L, a, b
CONF_JSON = "config.json"           # {"tolerance": float}

# ------------------ Page / basic styles ------------------
st.set_page_config(page_title="Stone Color Grader", page_icon="💎", layout="wide")
st.markdown(
    """
    <style>
      .result-title { font-size: 28px; font-weight: 800; margin: 0.25rem 0; }
      .badge-ok { background:#e8f7ee; color:#137333; padding:.45rem .7rem; border-radius:999px; display:inline-block; font-weight:700; }
      .badge-bad { background:#fde8e8; color:#b00020; padding:.45rem .7rem; border-radius:999px; display:inline-block; font-weight:700; }
      .card { border:1px solid #eee; border-radius:16px; padding:16px; }
      .muted { color:#666; }
      .gallery { display:grid; grid-template-columns: repeat(auto-fill, minmax(180px,1fr)); gap:12px; }
      .thumb { border:1px solid #eee; border-radius:12px; padding:8px; text-align:center;}
      .label { font-weight:700; margin-top:.25rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Persistence helpers ------------------
def load_tolerance() -> float:
    if os.path.exists(CONF_JSON):
        try:
            with open(CONF_JSON, "r") as f:
                return float(json.load(f).get("tolerance", 2.0))
        except Exception:
            return 2.0
    return 2.0

def save_tolerance(val: float):
    with open(CONF_JSON, "w") as f:
        json.dump({"tolerance": float(val)}, f)

def load_refs() -> pd.DataFrame:
    if os.path.exists(REFS_CSV):
        try:
            df = pd.read_csv(REFS_CSV)
            need = {"Label", "L", "a", "b"}
            if need.issubset(df.columns):
                return df[["Label", "L", "a", "b"]].copy()
        except Exception:
            pass
    return pd.DataFrame(columns=["Label", "L", "a", "b"])

def save_refs(df: pd.DataFrame):
    df[["Label", "L", "a", "b"]].to_csv(REFS_CSV, index=False)

# ------------------ Session init ------------------
if "tolerance" not in st.session_state:
    st.session_state.tolerance = load_tolerance()

# Keep images only in-session (not persisted in CSV)
if "refs" not in st.session_state:
    base = load_refs()
    base["Image"] = None  # bytes (PNG) or None
    st.session_state.refs = base

# For duplicate modal flow
if "pending_replace" not in st.session_state:
    st.session_state.pending_replace = None  # (label:str, image:PIL.Image)

# ------------------ Color helpers ------------------
def pil_to_lab_mean(image: Image.Image):
    rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    lab = color.rgb2lab(rgb)
    L, a, b = lab.reshape(-1, 3).mean(axis=0)
    return float(L), float(a), float(b)

def lab_to_swatch(L, a, b, size=200) -> Image.Image:
    lab_img = np.zeros((size, size, 3), dtype=np.float32)
    lab_img[..., 0] = L; lab_img[..., 1] = a; lab_img[..., 2] = b
    rgb = color.lab2rgb(lab_img)  # 0..1
    rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb8, mode="RGB")

# ------------------ Reference CRUD ------------------
def add_or_replace_reference(label: str, image: Image.Image, replace=False):
    L, a, b = pil_to_lab_mean(image)
    # prepare row with optional image bytes
    buf = BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    img_bytes = buf.getvalue()

    if replace:
        # replace row
        st.session_state.refs.loc[st.session_state.refs["Label"] == label, ["L", "a", "b"]] = [L, a, b]
        st.session_state.refs.loc[st.session_state.refs["Label"] == label, "Image"] = [img_bytes]
    else:
        # append new
        new_row = pd.DataFrame([{"Label": label, "L": L, "a": a, "b": b, "Image": img_bytes}])
        st.session_state.refs = pd.concat([st.session_state.refs, new_row], ignore_index=True)

    save_refs(st.session_state.refs)     # persist LAB + label
    save_tolerance(st.session_state.tolerance)

def delete_references(labels: list[str]):
    if not labels:
        return
    st.session_state.refs = st.session_state.refs[~st.session_state.refs["Label"].isin(labels)].reset_index(drop=True)
    save_refs(st.session_state.refs)
    save_tolerance(st.session_state.tolerance)

# ------------------ Grading ------------------
def find_best_label(Lt, at, bt):
    """Return (label, within_tolerance, matched_row) — no ΔE shown."""
    if st.session_state.refs.empty:
        return None, False, None

    test = LabColor(Lt, at, bt)
    best_label, best_row, best_delta = None, None, None

    for _, row in st.session_state.refs.iterrows():
        ref = LabColor(row["L"], row["a"], row["b"])
        d = delta_e_cie2000(test, ref)
        d = float(d.item() if hasattr(d, "item") else d)
        if (best_delta is None) or (d < best_delta):
            best_delta, best_label, best_row = d, row["Label"], row

    return best_label, (best_delta is not None and best_delta <= st.session_state.tolerance), best_row

# ------------------ Small UI helpers ------------------
def show_ref_thumb(row):
    st.markdown('<div class="thumb">', unsafe_allow_html=True)
    if isinstance(row["Image"], (bytes, bytearray)):
        st.image(row["Image"], use_container_width=True)
    else:
        st.image(lab_to_swatch(row["L"], row["a"], row["b"], size=160), use_container_width=True)
    st.markdown(f'<div class="label">{row["Label"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="muted">L={row["L"]:.1f}, a={row["a"]:.1f}, b={row["b"]:.1f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Tabs ------------------
tab1, tab2 = st.tabs(["🛠️ Calibration & Reference Stones", "🎯 Grading (Production)"])

# ================== TAB 1 ==================
with tab1:
    st.subheader("Reference Library & Tolerance")

    # Tolerance (persists)
    st.session_state.tolerance = st.slider(
        "ΔE Tolerance (affects grading)",
        min_value=0.5, max_value=15.0, step=0.1, value=float(st.session_state.tolerance)
    )
    save_tolerance(st.session_state.tolerance)

    st.divider()
    st.subheader("Add Reference Stone")

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        ref_label = st.text_input("Stone Label", placeholder="e.g., Ruby Grade A")
    with c2:
        ref_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    with c3:
        ref_cam = st.camera_input("Or capture via camera")

    # Decide the image to use
    new_ref_image = None
    if ref_cam is not None:
        new_ref_image = Image.open(ref_cam)
    elif ref_file is not None:
        new_ref_image = Image.open(ref_file)

    if st.button("➕ Add / Update Reference", use_container_width=True):
        if not ref_label or new_ref_image is None:
            st.error("Please enter a label and provide an image.")
        else:
            label = ref_label.strip()
            exists = label in list(st.session_state.refs["Label"].values)
            if exists:
                # queue modal confirmation
                st.session_state.pending_replace = (label, new_ref_image)
            else:
                add_or_replace_reference(label, new_ref_image, replace=False)
                st.success(f"Added reference: {label}")

    # Delete / Backup / Restore
    st.divider()
    st.subheader("Manage / Backup")

    if st.session_state.refs.empty:
        st.info("No reference stones saved yet.")
    else:
        del_labels = st.multiselect("Select references to delete", st.session_state.refs["Label"].tolist())
        coldel1, coldl2, coldl3 = st.columns([1, 1, 2])
        with coldel1:
            if st.button("🗑️ Delete Selected", use_container_width=True):
                delete_references(del_labels)
                st.success("Selected references deleted.")

        # Downloads
        refs_csv_bytes = st.session_state.refs[["Label", "L", "a", "b"]].to_csv(index=False).encode("utf-8")
        conf_bytes = json.dumps({"tolerance": float(st.session_state.tolerance)}, indent=2).encode("utf-8")

        with coldl2:
            st.download_button("⬇️ reference_stones.csv", refs_csv_bytes, "reference_stones.csv", "text/csv", use_container_width=True)
        with coldl3:
            st.download_button("⬇️ config.json", conf_bytes, "config.json", "application/json", use_container_width=True)

        # Restore
        restore = st.file_uploader("Restore references from CSV", type=["csv"], key="restore_csv")
        if restore is not None:
            try:
                df_new = pd.read_csv(restore)
                need = {"Label", "L", "a", "b"}
                if not need.issubset(df_new.columns):
                    st.error("CSV must contain columns: Label, L, a, b")
                else:
                    df_new = df_new[["Label", "L", "a", "b"]].copy()
                    df_new["Image"] = None
                    st.session_state.refs = df_new.reset_index(drop=True)
                    save_refs(st.session_state.refs)
                    st.success("Reference stones restored from CSV.")
            except Exception as e:
                st.error(f"Restore failed: {e}")

    # Gallery
    st.divider()
    st.subheader("Reference Gallery")
    if st.session_state.refs.empty:
        st.caption("Add references to see them here.")
    else:
        st.markdown('<div class="gallery">', unsafe_allow_html=True)
        for _, r in st.session_state.refs.iterrows():
            show_ref_thumb(r)
        st.markdown('</div>', unsafe_allow_html=True)

    # -------- Modal-style duplicate confirmation --------
    if st.session_state.pending_replace is not None:
        label, pil_img = st.session_state.pending_replace

        st.markdown("""
        <style>
        .modal-background {
            position: fixed; top: 0; left: 0;
            width: 100%; height: 100%;
            background-color: rgba(0,0,0,0.55);
            z-index: 9998;
        }
        .modal-container {
            position: fixed; top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            background-color: white; padding: 22px;
            border-radius: 12px; box-shadow: 0 8px 40px rgba(0,0,0,0.35);
            z-index: 9999; max-width: 420px; text-align: center;
        }
        .modal-title { font-size: 20px; font-weight: 800; margin-bottom: 6px; }
        .modal-text { color:#444; }
        </style>
        """, unsafe_allow_html=True)

        # Backdrop
        st.markdown("<div class='modal-background'></div>", unsafe_allow_html=True)
        # Content
        st.markdown(
            f"""
            <div class='modal-container'>
                <div class='modal-title'>⚠️ Duplicate Reference</div>
                <div class='modal-text'>
                    Reference stone '<b>{label}</b>' already exists.<br>
                    Do you want to replace it with the new one?
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        # Buttons (cannot render inside HTML; place right after)
        m1, m2 = st.columns(2)
        with m1:
            if st.button("✅ Replace", key="replace_modal_btn"):
                add_or_replace_reference(label, pil_img, replace=True)
                st.session_state.pending_replace = None
                st.success(f"Reference stone '{label}' replaced.")
        with m2:
            if st.button("❌ Cancel", key="cancel_modal_btn"):
                st.session_state.pending_replace = None
                st.info("Operation cancelled.")

# ================== TAB 2 ==================
with tab2:
    st.subheader("Upload or Capture Test Stone (instant grading)")
    g1, g2 = st.columns(2)
    with g1:
        test_file = st.file_uploader("Upload Test Stone", type=["jpg", "jpeg", "png"], key="test_upload")
    with g2:
        test_cam = st.camera_input("Or capture via camera", key="test_cam")

    # latest available image (camera > file)
    test_img = None
    if test_cam is not None:
        test_img = Image.open(test_cam)
    elif test_file is not None:
        test_img = Image.open(test_file)

    st.divider()
    if test_img is None:
        st.info("Upload or capture a test stone to grade.")
    elif st.session_state.refs.empty:
        st.warning("No reference stones available. Add them in the Calibration tab.")
        st.image(test_img, caption="Test Stone", use_container_width=True)
    else:
        # grade instantly
        Lt, at, bt = pil_to_lab_mean(test_img)
        best_label, within_tol, best_row = find_best_label(Lt, at, bt)

        colA, colB = st.columns([1, 1])
        with colA:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<div class='muted'>Test Stone</div>", unsafe_allow_html=True)
            st.image(test_img, use_container_width=True)
            st.image(lab_to_swatch(Lt, at, bt, size=80), caption="Test Color", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with colB:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # BIG, BOLD GRADE NAME
            if best_label is None:
                st.markdown("<div class='result-title'>No Match</div>", unsafe_allow_html=True)
                st.markdown('<span class="badge-bad">❌ OUT OF TOLERANCE</span>', unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result-title'>{best_label}</div>", unsafe_allow_html=True)
                st.markdown(
                    '<span class="badge-ok">✅ MATCH</span>' if within_tol else
                    '<span class="badge-bad">❌ OUT OF TOLERANCE</span>',
                    unsafe_allow_html=True
                )
                # Show matched reference image if present, else swatch
                if isinstance(best_row["Image"], (bytes, bytearray)):
                    st.image(best_row["Image"], caption="Matched Reference", use_container_width=True)
                else:
                    st.image(lab_to_swatch(best_row["L"], best_row["a"], best_row["b"], size=160),
                             caption="Matched Reference (Color)", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.caption(f"Tolerance: ΔE ≤ {st.session_state.tolerance:.1f} (set in Calibration tab)")